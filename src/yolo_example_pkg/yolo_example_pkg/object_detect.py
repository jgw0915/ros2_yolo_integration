import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory
import torch

try:
    from sensor_msgs_py import point_cloud2
except Exception:
    point_cloud2 = None

using_yolo_det_model = True
using_yolo_seg_model = True

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__("yolo_detection_node")

        self.declare_parameter("target_labels", ["bear", "knob"])
        self.create_subscription(String, "/target_label", self._target_label_callback, 10)
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("det_model_name", "best_object_detection.pt")
        self.declare_parameter("seg_model_name", "best_segmentation.pt")
        self.declare_parameter("enable_segmentation", True)
        self.declare_parameter("road_seg_labels", ["road"])
        self.declare_parameter("bridge_seg_labels", ["bridge"])
        self.declare_parameter("segmentation_connection_dilation_pixels", 17)
        self.declare_parameter("target_lock_enabled", True)
        self.declare_parameter("target_lock_max_pixel_distance", 180.0)
        self.declare_parameter("target_lock_max_depth_delta", 0.8)
        self.declare_parameter("target_lock_miss_limit", 10)
        self.declare_parameter("target_lock_strict_on_miss", False)
        self.declare_parameter("target_lock_hold_last_on_miss", True)
        self.declare_parameter("target_lock_iou_bonus", 250.0)
        self.declare_parameter("target_lock_min_iou_for_reassociate", 0.08)
        self.declare_parameter("target_lock_reassociate_pixel_distance", 85.0)
        self.declare_parameter("bridge_geometry_camera_info_topic", "/camera/camera_info")
        self.declare_parameter("bridge_geometry_max_depth_age_seconds", 0.20)
        self.declare_parameter("bridge_geometry_min_valid_depth", 0.15)
        self.declare_parameter("bridge_geometry_max_valid_depth", 8.0)
        self.declare_parameter("bridge_geometry_depth_patch_radius", 3)
        self.declare_parameter("bridge_geometry_edge_sample_count", 12)
        self.declare_parameter("bridge_camera_fx", 0.0)
        self.declare_parameter("bridge_camera_fy", 0.0)
        self.declare_parameter("bridge_camera_cx", 0.0)
        self.declare_parameter("bridge_camera_cy", 0.0)
        self.declare_parameter("bridge_geometry_camera_frame", "")
        self.declare_parameter("bridge_entry_contact_min_pixels", 20)
        self.declare_parameter("bridge_entry_contact_min_width_pixels", 25)
        self.declare_parameter("bridge_entry_contact_center_margin_ratio", 0.40)
        self.declare_parameter("bridge_entry_contact_lower_y_ratio", 0.35)
        self.declare_parameter("bridge_entry_gate_expand_pixels", 15)
        self.declare_parameter("bridge_entry_fallback_allowed_for_turn_only", True)
        self.declare_parameter("bridge_side_inward_sample_pixels", 5)
        self.declare_parameter("bridge_side_depth_max_row_difference_m", 0.35)
        self.declare_parameter("bridge_side_min_pair_width_m", 0.25)
        self.declare_parameter("bridge_side_max_pair_width_m", 2.5)
        self.declare_parameter("bridge_side_projection_min_valid_pairs", 4)
        self.declare_parameter("bridge_side_depth_patch_radius", 4)
        self.declare_parameter("drivable_corridor_enable", True)
        self.declare_parameter("drivable_corridor_bottom_y_ratio", 0.88)
        self.declare_parameter("drivable_corridor_lower_y_ratio", 0.70)
        self.declare_parameter("drivable_corridor_center_y_ratio", 0.50)
        self.declare_parameter("drivable_corridor_center_band_height_ratio", 0.06)
        self.declare_parameter("drivable_corridor_min_bottom_width_ratio", 0.12)
        self.declare_parameter("drivable_corridor_min_center_width_ratio", 0.08)
        self.declare_parameter("drivable_corridor_min_continuous_score", 0.55)
        self.declare_parameter("drivable_corridor_center_tolerance_pixels", 45.0)
        self.declare_parameter("drivable_corridor_max_slope_pixels", 140.0)

        # 初始化 cv_bridge
        self.bridge = CvBridge()

        self.latest_depth_image_raw = None
        self.latest_depth_image_compressed = None
        self.latest_depth_stamp_raw = None
        self.latest_depth_stamp_compressed = None
        self.latest_depth_frame_raw = ""
        self.latest_depth_frame_compressed = ""

        self.allowed_labels = {
            label.strip().lower()
            for label in self.get_parameter("target_labels")
            .get_parameter_value()
            .string_array_value
            if label.strip()
        }
        self.conf_threshold = (
            self.get_parameter("conf_threshold").get_parameter_value().double_value
        )
        self.enable_segmentation = (
            self.get_parameter("enable_segmentation").get_parameter_value().bool_value
        )
        self.road_seg_labels = self._string_set_param("road_seg_labels")
        self.bridge_seg_labels = self._string_set_param("bridge_seg_labels")
        self.segmentation_connection_dilation_pixels = (
            self.get_parameter("segmentation_connection_dilation_pixels")
            .get_parameter_value()
            .integer_value
        )
        self.drivable_corridor_enable = (
            self.get_parameter("drivable_corridor_enable")
            .get_parameter_value()
            .bool_value
        )
        self.drivable_corridor_bottom_y_ratio = self._double_param(
            "drivable_corridor_bottom_y_ratio"
        )
        self.drivable_corridor_lower_y_ratio = self._double_param(
            "drivable_corridor_lower_y_ratio"
        )
        self.drivable_corridor_center_y_ratio = self._double_param(
            "drivable_corridor_center_y_ratio"
        )
        self.drivable_corridor_center_band_height_ratio = self._double_param(
            "drivable_corridor_center_band_height_ratio"
        )
        self.drivable_corridor_min_bottom_width_ratio = self._double_param(
            "drivable_corridor_min_bottom_width_ratio"
        )
        self.drivable_corridor_min_center_width_ratio = self._double_param(
            "drivable_corridor_min_center_width_ratio"
        )
        self.drivable_corridor_min_continuous_score = self._double_param(
            "drivable_corridor_min_continuous_score"
        )
        self.drivable_corridor_center_tolerance_pixels = self._double_param(
            "drivable_corridor_center_tolerance_pixels"
        )
        self.drivable_corridor_max_slope_pixels = self._double_param(
            "drivable_corridor_max_slope_pixels"
        )
        self.target_lock_enabled = (
            self.get_parameter("target_lock_enabled").get_parameter_value().bool_value
        )
        self.target_lock_max_pixel_distance = (
            self.get_parameter("target_lock_max_pixel_distance")
            .get_parameter_value()
            .double_value
        )
        self.target_lock_max_depth_delta = (
            self.get_parameter("target_lock_max_depth_delta")
            .get_parameter_value()
            .double_value
        )
        self.target_lock_miss_limit = (
            self.get_parameter("target_lock_miss_limit")
            .get_parameter_value()
            .integer_value
        )
        self.target_lock_strict_on_miss = (
            self.get_parameter("target_lock_strict_on_miss")
            .get_parameter_value()
            .bool_value
        )
        self.target_lock_hold_last_on_miss = (
            self.get_parameter("target_lock_hold_last_on_miss")
            .get_parameter_value()
            .bool_value
        )
        self.target_lock_iou_bonus = (
            self.get_parameter("target_lock_iou_bonus")
            .get_parameter_value()
            .double_value
        )
        self.target_lock_min_iou_for_reassociate = (
            self.get_parameter("target_lock_min_iou_for_reassociate")
            .get_parameter_value()
            .double_value
        )
        self.target_lock_reassociate_pixel_distance = (
            self.get_parameter("target_lock_reassociate_pixel_distance")
            .get_parameter_value()
            .double_value
        )
        self.bridge_geometry_camera_info_topic = (
            self.get_parameter("bridge_geometry_camera_info_topic")
            .get_parameter_value()
            .string_value
        )
        self.bridge_geometry_max_depth_age = (
            self.get_parameter("bridge_geometry_max_depth_age_seconds")
            .get_parameter_value()
            .double_value
        )
        self.bridge_geometry_min_valid_depth = (
            self.get_parameter("bridge_geometry_min_valid_depth")
            .get_parameter_value()
            .double_value
        )
        self.bridge_geometry_max_valid_depth = (
            self.get_parameter("bridge_geometry_max_valid_depth")
            .get_parameter_value()
            .double_value
        )
        self.bridge_geometry_depth_patch_radius = (
            self.get_parameter("bridge_geometry_depth_patch_radius")
            .get_parameter_value()
            .integer_value
        )
        self.bridge_geometry_edge_sample_count = (
            self.get_parameter("bridge_geometry_edge_sample_count")
            .get_parameter_value()
            .integer_value
        )
        self.bridge_camera_fx = (
            self.get_parameter("bridge_camera_fx").get_parameter_value().double_value
        )
        self.bridge_camera_fy = (
            self.get_parameter("bridge_camera_fy").get_parameter_value().double_value
        )
        self.bridge_camera_cx = (
            self.get_parameter("bridge_camera_cx").get_parameter_value().double_value
        )
        self.bridge_camera_cy = (
            self.get_parameter("bridge_camera_cy").get_parameter_value().double_value
        )
        self.bridge_geometry_camera_frame = (
            self.get_parameter("bridge_geometry_camera_frame")
            .get_parameter_value()
            .string_value
        )
        self.bridge_entry_contact_min_pixels = (
            self.get_parameter("bridge_entry_contact_min_pixels")
            .get_parameter_value()
            .integer_value
        )
        self.bridge_entry_contact_min_width_pixels = (
            self.get_parameter("bridge_entry_contact_min_width_pixels")
            .get_parameter_value()
            .integer_value
        )
        self.bridge_entry_contact_center_margin_ratio = (
            self.get_parameter("bridge_entry_contact_center_margin_ratio")
            .get_parameter_value()
            .double_value
        )
        self.bridge_entry_contact_lower_y_ratio = (
            self.get_parameter("bridge_entry_contact_lower_y_ratio")
            .get_parameter_value()
            .double_value
        )
        self.bridge_entry_gate_expand_pixels = (
            self.get_parameter("bridge_entry_gate_expand_pixels")
            .get_parameter_value()
            .integer_value
        )
        self.bridge_entry_fallback_allowed_for_turn_only = (
            self.get_parameter("bridge_entry_fallback_allowed_for_turn_only")
            .get_parameter_value()
            .bool_value
        )
        self.bridge_side_inward_sample_pixels = (
            self.get_parameter("bridge_side_inward_sample_pixels")
            .get_parameter_value()
            .integer_value
        )
        self.bridge_side_depth_max_row_difference = (
            self.get_parameter("bridge_side_depth_max_row_difference_m")
            .get_parameter_value()
            .double_value
        )
        self.bridge_side_min_pair_width = (
            self.get_parameter("bridge_side_min_pair_width_m")
            .get_parameter_value()
            .double_value
        )
        self.bridge_side_max_pair_width = (
            self.get_parameter("bridge_side_max_pair_width_m")
            .get_parameter_value()
            .double_value
        )
        self.bridge_side_projection_min_valid_pairs = (
            self.get_parameter("bridge_side_projection_min_valid_pairs")
            .get_parameter_value()
            .integer_value
        )
        self.bridge_side_depth_patch_radius = (
            self.get_parameter("bridge_side_depth_patch_radius")
            .get_parameter_value()
            .integer_value
        )
        self.locked_target = None
        self.lock_misses = 0
        self.current_frame_target = None
        self.active_target_label = None
        self.last_selected_target_class = None
        self.camera_info_received = False
        self.warned_missing_camera_info = False
        self.valid_depth_samples = 0
        self.rejected_depth_samples = 0
        self.bridge_edge_points_published = 0
        self.bridge_boundary_points_published = 0
        self.bridge_entry_points_published = 0
        self.bridge_edge_points_last_published_count = 0
        self.bridge_boundary_points_last_published_count = 0
        self.bridge_entry_point_published_last = False
        self.bridge_mask_found_last = False
        self.bridge_entry_confirmed_last = False
        self.last_bridge_geometry_reason = "not published yet"
        self.last_bridge_geometry_debug_time = None

        # 使用 yolo detection model 位置
        if using_yolo_det_model:
            det_model_name = (
                self.get_parameter("det_model_name").get_parameter_value().string_value
            )
            det_model_path = os.path.join(
                get_package_share_directory("yolo_example_pkg"),
                "models",
                det_model_name,
            )

        # 使用 yolo segmentation model 位置
        if using_yolo_seg_model and self.enable_segmentation:
            seg_model_name = (
                self.get_parameter("seg_model_name").get_parameter_value().string_value
            )
            seg_model_path = os.path.join(
                get_package_share_directory("yolo_example_pkg"),
                "models",
                seg_model_name,
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device : ", device)

        # 初始化 YOLO detection 模型
        if using_yolo_det_model:
            self.det_model = YOLO(det_model_path)
            self.det_model.to(device)

        # 初始化 YOLO segmentation 模型
        if using_yolo_seg_model and self.enable_segmentation:
            self.seg_model = YOLO(seg_model_path)
            self.seg_model.to(device)

        # 訂閱影像 Topic
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 1
        )

        # 訂閱 **無壓縮** 深度圖 Topic
        self.depth_sub_raw = self.create_subscription(
            Image, "/camera/depth/image_raw", self.depth_callback_raw, 1
        )

        # 訂閱 **壓縮** 深度圖 Topic
        self.depth_sub_compressed = self.create_subscription(
            CompressedImage,
            "/camera/depth/compressed",
            self.depth_callback_compressed,
            1,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.bridge_geometry_camera_info_topic,
            self.camera_info_callback,
            1,
        )

        # 發佈處理後的影像 Topic
        if using_yolo_det_model:
            self.det_image_pub = self.create_publisher(
                CompressedImage, "/yolo/detection/compressed", 10
            )

        if using_yolo_seg_model and self.enable_segmentation:
            self.seg_image_pub = self.create_publisher(
                CompressedImage, "/yolo/segmentation/compressed", 10
            )
            self.segmentation_info_pub = self.create_publisher(
                Float32MultiArray, "/yolo/segmentation_info", 10
            )
            self.drivable_corridor_pub = self.create_publisher(
                Float32MultiArray, "/yolo/drivable_corridor_info", 10
            )
            self.drivable_corridor_debug_pub = self.create_publisher(
                String, "/yolo/drivable_corridor_debug", 10
            )

        # 發布 目標檢測數據 (是否找到目標 + 距離)
        self.target_pub = self.create_publisher(
            Float32MultiArray, "/yolo/target_info", 10
        )
        self.target_bbox_pub = self.create_publisher(
            Float32MultiArray, "/yolo/target_bbox", 10
        )
        self.target_surface_pub = self.create_publisher(
            Float32MultiArray, "/yolo/target_surface_info", 10
        )
        self.bridge_edge_points_pub = self.create_publisher(
            PointCloud2, "/yolo/bridge_edge_points", 10
        )
        self.bridge_boundary_points_pub = self.create_publisher(
            PointCloud2, "/yolo/bridge_boundary_points", 10
        )
        self.bridge_entry_point_pub = self.create_publisher(
            PointStamped, "/yolo/bridge_entry_point", 10
        )
        self.bridge_pre_entry_point_pub = self.create_publisher(
            PointStamped, "/yolo/bridge_pre_entry_point", 10
        )
        self.bridge_geometry_debug_pub = self.create_publisher(
            String, "/yolo/bridge_geometry_debug", 10
        )

        self.x_multi_depth_pub = self.create_publisher(
            Float32MultiArray, "/camera/x_multi_depth_values", 10
        )

        # 相機畫面中央高度上切成 n 個等距水平點。
        self.x_num_splits = 20
        self.det_publish_count = 0
        self.seg_publish_count = 0
        self.seg_no_mask_count = 0

        self.get_logger().info(
            "YOLO target labels: "
            + (", ".join(sorted(self.allowed_labels)) if self.allowed_labels else "all")
        )
        self.get_logger().info(
            "Segmentation enabled: "
            + ("yes" if self.enable_segmentation else "no")
            + ", topic: /yolo/segmentation/compressed"
        )
        self.bridge_geometry_debug_timer = self.create_timer(
            1.0, self._publish_bridge_geometry_heartbeat
        )

    def _string_set_param(self, name):
        return {
            label.strip().lower()
            for label in self.get_parameter(name)
            .get_parameter_value()
            .string_array_value
            if label.strip()
        }

    def _double_param(self, name):
        return self.get_parameter(name).get_parameter_value().double_value

    def _normalize_target_label(self, label):
        label = (label or "").strip().lower()
        if label == "door_knob":
            return "knob"
        return label

    def _target_label_callback(self, msg):
        label = self._normalize_target_label(msg.data)
        if label and label != self.active_target_label:
            self.active_target_label = label
            self.locked_target = None
            self.lock_misses = 0
            self.current_frame_target = None
            self.last_selected_target_class = None
            self.get_logger().info(f"Active YOLO target switched to: {label}")

    def camera_info_callback(self, msg):
        if len(msg.k) >= 9 and msg.k[0] > 0.0 and msg.k[4] > 0.0:
            self.bridge_camera_fx = float(msg.k[0])
            self.bridge_camera_fy = float(msg.k[4])
            self.bridge_camera_cx = float(msg.k[2])
            self.bridge_camera_cy = float(msg.k[5])
            if msg.header.frame_id:
                self.bridge_geometry_camera_frame = msg.header.frame_id
            self.camera_info_received = True

    def depth_callback_raw(self, msg):
        """接收 **無壓縮** 深度圖"""
        try:
            self.latest_depth_image_raw = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            self.latest_depth_stamp_raw = msg.header.stamp
            self.latest_depth_frame_raw = msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f"Could not convert raw depth image: {e}")

    def depth_callback_compressed(self, msg):
        """接收 **壓縮** 深度圖（當無壓縮深度圖不可用時使用）"""
        try:
            # 自行強制使用 cv2.IMREAD_UNCHANGED 解碼，避開 cv_bridge 的潛在雷區
            np_arr = np.frombuffer(msg.data, np.uint8)
            depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                self.latest_depth_image_compressed = depth_img
                self.latest_depth_stamp_compressed = msg.header.stamp
                self.latest_depth_frame_compressed = msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f"Could not convert compressed depth image: {e}")

    def image_callback(self, msg):
        """接收影像並進行物體檢測"""
        self.current_frame_target = None
        # 將 ROS 影像消息轉換為 OpenCV 格式
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        if using_yolo_det_model:
            # 使用 YOLO Detection 模型檢測物體
            try:
                det_results = self.det_model(cv_image, conf=self.conf_threshold, verbose=False)
            except Exception as e:
                self.get_logger().error(f"Error during YOLO detection: {e}")
                det_results = None
            
            # 繪製 Bounding Box
            if det_results is not None:
                det_image = self.draw_bounding_boxes(cv_image, det_results)
                
                # 取得影像中心深度並發布
                self.publish_x_multi_depths(det_image)
                
                # 發佈 Detection 影像
                self.publish_det_image(det_image, msg.header)

        if using_yolo_seg_model and self.enable_segmentation:
            # 使用 YOLO Segmentation 模型檢測物體
            try:
                seg_results = self.seg_model(cv_image, conf=self.conf_threshold, verbose=False)
            except Exception as e:
                self.get_logger().error(f"Error during YOLO segmentation: {e}")
                self.publish_seg_image(cv_image, msg.header)
                return

            # 繪製 Mask
            seg_image, has_mask = self.draw_masks(cv_image, seg_results, msg.header)
            if not has_mask:
                self.seg_no_mask_count += 1
                if self.seg_no_mask_count % 60 == 0:
                    self.get_logger().warn(
                        "Segmentation produced no masks for 60 frames. "
                        "Check seg_model_name/weights and conf_threshold."
                    )
            else:
                self.seg_no_mask_count = 0
            
            # 發佈 Segmentation 影像
            self.publish_seg_image(seg_image, msg.header)

    def draw_cross(self, image):
        # 回傳繪製十字架的影像和畫面正中間的像素座標
        height, width = image.shape[:2]
        cx_center = width // 2
        cy_center = height // 2
        # 繪製橫線
        cv2.line(image, (0, cy_center), (width, cy_center), (0, 0, 255), 2)

        # 繪製直線
        cv2.line(
            image,
            (cx_center, cy_center - 10),
            (cx_center, cy_center + 10),
            (0, 0, 255),
            2,
        )

        cv2.line(
            image,
            (cx_center, cy_center - 10),
            (cx_center, cy_center + 10),
            (0, 0, 255),
            2,
        )

        # 計算橫線上的 n 個等分點
        segment_length = width // self.x_num_splits
        points = [
            (i * segment_length, cy_center) for i in range(self.x_num_splits + 1)
        ]  # 11 個點表示 10 段區間的端點

        # 在每個等分點繪製垂直的短黑線
        for x, y in points:
            cv2.line(image, (x, y - 10), (x, y + 10), (0, 0, 0), 2)  # 黑色垂直線

        return image, points

    def draw_bounding_boxes(self, image, results):
        """在影像上繪製 YOLO 檢測到的 Bounding Box"""
        # 一開始預設沒找到目標
        found_target = 0
        target_distance = 0.0
        delta_x = 0.0
        det_image = image.copy()
        det_image, _ = self.draw_cross(det_image)
        height, width = image.shape[:2]
        image_center_x = width // 2
        target_candidates = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls[0])
                class_name = str(self.det_model.names[class_id]).strip()
                normalized_class_name = class_name.lower()

                # 只保留設定內的標籤；target_labels 空陣列時代表不過濾
                if self.allowed_labels and normalized_class_name not in self.allowed_labels:
                    continue

                # 計算 Bounding Box 正中心點
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 優先使用無壓縮的深度圖
                depth_value = self.get_depth_at(cx, cy)
                depth_text = f"{depth_value:.2f}m" if depth_value > 0.0 else "N/A"
                delta_x = cx - image_center_x

                # 根據 class_id 產生隨機但固定的顏色 (B, G, R)
                rng = np.random.RandomState(class_id)
                color = tuple(int(c) for c in rng.randint(0, 256, 3))

                # 繪製框和標籤
                cv2.rectangle(det_image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f} Depth: {depth_text}"

                cv2.putText(
                    det_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                # Only the active task target can drive /yolo/target_*.
                # If /target_label has not arrived yet, keep startup behavior.
                if (
                    self.active_target_label is not None
                    and normalized_class_name != self.active_target_label
                ):
                    continue

                target_candidate = {
                    "class_name": normalized_class_name,
                    "confidence": conf,
                    "depth": depth_value,
                    "delta_x": delta_x,
                    "center_error": abs(delta_x),
                    "has_valid_depth": depth_value > 0.0,
                    "center_x": cx,
                    "center_y": cy,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": max(0, x2 - x1),
                    "height": max(0, y2 - y1),
                    "image_width": width,
                    "image_height": height,
                }
                target_candidates.append(target_candidate)

        best_target = self._select_target(target_candidates)
        if best_target is not None:
            found_target = 1
            target_distance = best_target["depth"]
            delta_x = best_target["delta_x"]
            self.current_frame_target = dict(best_target)
            selected_class = best_target.get("class_name", "")
            if selected_class and selected_class != self.last_selected_target_class:
                self.last_selected_target_class = selected_class
                self.get_logger().info(
                    f"Selected YOLO target for /yolo/target_info: {selected_class}"
                )
        else:
            self.current_frame_target = None

        self.publish_target_info(found_target, target_distance, delta_x)
        self.publish_target_bbox(found_target, best_target)
        return det_image

    def _select_target(self, candidates):
        if not candidates:
            self._register_lock_miss()
            return None

        if self.target_lock_enabled and self.locked_target is not None:
            locked_candidates = [
                candidate
                for candidate in candidates
                if self._matches_locked_target(candidate)
            ]
            if locked_candidates:
                iou_candidates = [
                    candidate
                    for candidate in locked_candidates
                    if self._locked_bbox_iou(candidate)
                    >= self.target_lock_min_iou_for_reassociate
                ]
                if iou_candidates:
                    selected = min(iou_candidates, key=self._locked_target_score)
                else:
                    close_candidates = [
                        candidate
                        for candidate in locked_candidates
                        if self._locked_center_distance(candidate)
                        <= self.target_lock_reassociate_pixel_distance
                    ]
                    if not close_candidates:
                        self._register_lock_miss()
                        if self.locked_target is not None:
                            if self.target_lock_hold_last_on_miss:
                                return dict(self.locked_target)
                            if self.target_lock_strict_on_miss:
                                return None
                        close_candidates = locked_candidates
                    selected = min(close_candidates, key=self._locked_target_score)
                self._update_locked_target(selected)
                return selected

            self._register_lock_miss()
            if self.locked_target is not None:
                if self.target_lock_hold_last_on_miss:
                    return dict(self.locked_target)
                if self.target_lock_strict_on_miss:
                    return None
                selected = min(candidates, key=self._locked_target_score)
                self._update_locked_target(selected)
                return selected

        selected = None
        for candidate in candidates:
            if self._is_better_target(candidate, selected):
                selected = candidate
        self._update_locked_target(selected)
        return selected

    def _matches_locked_target(self, candidate):
        locked_class = self.locked_target.get("class_name")
        candidate_class = candidate.get("class_name")
        if locked_class and candidate_class and locked_class != candidate_class:
            return False

        center_distance = self._locked_center_distance(candidate)
        if center_distance > self.target_lock_max_pixel_distance:
            return False

        iou = self._locked_bbox_iou(candidate)
        if (
            iou < self.target_lock_min_iou_for_reassociate
            and center_distance > self.target_lock_reassociate_pixel_distance
        ):
            return False

        locked_depth = self.locked_target.get("depth", -1.0)
        candidate_depth = candidate["depth"]
        if locked_depth > 0.0 and candidate_depth > 0.0:
            return abs(candidate_depth - locked_depth) <= self.target_lock_max_depth_delta
        return True

    def _locked_target_score(self, candidate):
        depth_score = 0.0
        locked_depth = self.locked_target.get("depth", -1.0)
        if locked_depth > 0.0 and candidate["depth"] > 0.0:
            depth_score = abs(candidate["depth"] - locked_depth) * 50.0
        iou_bonus = self._locked_bbox_iou(candidate) * self.target_lock_iou_bonus
        return self._locked_center_distance(candidate) + depth_score - iou_bonus

    def _locked_center_distance(self, candidate):
        dx = candidate["center_x"] - self.locked_target["center_x"]
        dy = candidate["center_y"] - self.locked_target["center_y"]
        return float(np.hypot(dx, dy))

    def _locked_bbox_iou(self, candidate):
        if self.locked_target is None:
            return 0.0
        required_keys = ("x1", "y1", "x2", "y2")
        if not all(key in self.locked_target for key in required_keys):
            return 0.0

        inter_x1 = max(candidate["x1"], self.locked_target["x1"])
        inter_y1 = max(candidate["y1"], self.locked_target["y1"])
        inter_x2 = min(candidate["x2"], self.locked_target["x2"])
        inter_y2 = min(candidate["y2"], self.locked_target["y2"])
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        candidate_area = max(0, candidate["x2"] - candidate["x1"]) * max(
            0, candidate["y2"] - candidate["y1"]
        )
        locked_area = max(0, self.locked_target["x2"] - self.locked_target["x1"]) * max(
            0, self.locked_target["y2"] - self.locked_target["y1"]
        )
        union_area = candidate_area + locked_area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _update_locked_target(self, target):
        if target is None:
            return
        self.locked_target = dict(target)
        self.lock_misses = 0

    def _register_lock_miss(self):
        if self.locked_target is None:
            return
        self.lock_misses += 1
        if self.lock_misses > self.target_lock_miss_limit:
            self.locked_target = None
            self.lock_misses = 0

    def _is_better_target(self, candidate, current):
        if current is None:
            return True
        if candidate["has_valid_depth"] != current["has_valid_depth"]:
            return candidate["has_valid_depth"]
        if candidate["center_error"] != current["center_error"]:
            return candidate["center_error"] < current["center_error"]
        return candidate["confidence"] > current["confidence"]

    def draw_masks(self, image, results, header=None):
        """在影像上繪製 YOLO 檢測到的 Mask"""
        height, width = image.shape[:2]
        mask_image = image.copy()  # 從原始影像複製一份來繪製 Mask
        road_mask = np.zeros((height, width), dtype=bool)
        bridge_mask = np.zeros((height, width), dtype=bool)
        has_mask = False

        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                for i, mask in enumerate(masks):
                    has_mask = True
                    # Create a boolean mask and assign color
                    mask_resized = cv2.resize(mask, (width, height))
                    mask_bool = mask_resized > 0.5
                    
                    # 根據 class_id 產生隨機但固定的顏色 (B, G, R)
                    class_id = -1
                    class_name = ""
                    if boxes is not None and boxes.cls is not None and i < len(boxes.cls):
                        class_id = int(boxes.cls[i])
                        class_name = str(self.seg_model.names[class_id]).strip().lower()
                    rng = np.random.RandomState(class_id)
                    color = tuple(int(c) for c in rng.randint(0, 256, 3))

                    if class_name in self.road_seg_labels:
                        road_mask |= mask_bool
                    if class_name in self.bridge_seg_labels:
                        bridge_mask |= mask_bool
                    
                    # Blend the mask for better visibility
                    mask_colored = np.zeros_like(mask_image)
                    mask_colored[mask_bool] = color
                    mask_image = cv2.addWeighted(mask_image, 1, mask_colored, 0.5, 0)

        corridor = self._compute_drivable_corridor(road_mask, bridge_mask)
        if self.drivable_corridor_enable:
            self.publish_drivable_corridor_info(corridor)
            self.publish_drivable_corridor_debug(corridor)
            mask_image = self._draw_drivable_corridor_overlay(mask_image, corridor)
        self.publish_segmentation_info(road_mask, bridge_mask, header=header)
        self.publish_target_surface_info(self.current_frame_target, bridge_mask)
        self.publish_bridge_geometry(road_mask, bridge_mask, header)
        return mask_image, has_mask

    def get_depth_at(self, x, y):
        """
        取得指定像素的深度值，轉換為米 (m)
        若深度出問題，回傳 -1
        """
        # **優先使用無壓縮的深度圖**
        depth_image = (
            self.latest_depth_image_raw
            if self.latest_depth_image_raw is not None
            else self.latest_depth_image_compressed
        )

        if depth_image is None:
            return -1.0

        # 如果深度影像為三通道，那只取第一個數值
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]

        try:
            depth_value = depth_image[y, x]
            if depth_value < 0.0001 or depth_value == 0.0:  # 無效深度
                return -1.0
            return depth_value / 1000.0  # 16-bit 深度圖通常單位為 mm，轉換為 m
        except IndexError:
            return -1.0

    def publish_det_image(self, image, header=None):
        """將 Detection 影像轉換並發佈到 ROS"""
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
                image, dst_format="jpeg"
            )
            if header is not None:
                compressed_msg.header = header
            self.det_image_pub.publish(compressed_msg)
            self.det_publish_count += 1
        except Exception as e:
            self.get_logger().error(f"Could not publish detection image: {e}")

    def publish_seg_image(self, image, header=None):
        """將 Segmentation 影像轉換並發佈到 ROS"""
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
                image, dst_format="jpeg"
            )
            if header is not None:
                compressed_msg.header = header
            self.seg_image_pub.publish(compressed_msg)
            self.seg_publish_count += 1
            if self.seg_publish_count % 120 == 0:
                self.get_logger().info(
                    f"Published segmentation frames: {self.seg_publish_count}"
                )
        except Exception as e:
            self.get_logger().error(f"Could not publish segmentation image: {e}")

    def _compute_drivable_corridor(self, road_mask, bridge_mask):
        height, width = road_mask.shape
        center_x = width * 0.5
        empty = {
            "corridor_valid": False,
            "bottom_connected": False,
            "centerline_reached": False,
            "upper_mid_reached": False,
            "continuous_score": 0.0,
            "bottom_width_ratio": 0.0,
            "center_width_ratio": 0.0,
            "corridor_center_x_bottom": 0.0,
            "corridor_center_x_mid": 0.0,
            "corridor_center_x_centerline": 0.0,
            "corridor_error_x_pixels": 0.0,
            "corridor_slope_pixels": 0.0,
            "road_ratio_in_corridor": 0.0,
            "bridge_ratio_in_corridor": 0.0,
            "drivable_type": 0.0,
            "side_view_likely": False,
            "image_width": float(width),
            "image_height": float(height),
            "reason_code": 1.0,
            "reason": "no drivable mask",
            "component_mask": np.zeros((height, width), dtype=bool),
            "bands": {},
        }
        if height <= 0 or width <= 0:
            return empty

        road_mask = road_mask.astype(bool)
        bridge_mask = bridge_mask.astype(bool)
        drivable_mask = road_mask | bridge_mask
        if int(np.count_nonzero(drivable_mask)) <= 0:
            return empty

        labels, label_img, stats, _ = cv2.connectedComponentsWithStats(
            drivable_mask.astype(np.uint8), 8
        )
        bottom_y = self._clamped_row(height, self.drivable_corridor_bottom_y_ratio)
        lower_y = self._clamped_row(height, self.drivable_corridor_lower_y_ratio)
        center_y = self._clamped_row(height, self.drivable_corridor_center_y_ratio)
        band_half = max(
            2,
            int(height * max(0.01, self.drivable_corridor_center_band_height_ratio) * 0.5),
        )
        center_band = (max(0, center_y - band_half), min(height, center_y + band_half + 1))
        bottom_band = (bottom_y, height)
        lower_band = (lower_y, min(height, max(lower_y + 1, bottom_y + band_half)))
        upper_mid_band = (
            max(0, int(height * 0.35)),
            min(height, max(int(height * 0.35) + 1, int(height * 0.48))),
        )

        best = None
        min_area = max(20, int(width * height * 0.0015))
        for label in range(1, labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            component = label_img == label
            bottom = self._drivable_band_stats(component, *bottom_band)
            lower = self._drivable_band_stats(component, *lower_band)
            center = self._drivable_band_stats(component, *center_band)
            upper_mid = self._drivable_band_stats(component, *upper_mid_band)
            touches_bottom = bottom["width"] > 0.0
            reaches_center = center["width"] > 0.0
            center_overlap_score = max(
                0.0,
                1.0 - abs((bottom["center_x"] or lower["center_x"] or center["center_x"]) - center_x)
                / max(1.0, width * 0.45),
            )
            vertical_span = int(stats[label, cv2.CC_STAT_HEIGHT])
            score = (
                area
                + (width * height * 0.25 if touches_bottom else 0.0)
                + (width * height * 0.18 if reaches_center else 0.0)
                + vertical_span * width * 0.08
                + center_overlap_score * width * height * 0.10
            )
            if best is None or score > best["score"]:
                best = {
                    "label": label,
                    "component": component,
                    "score": score,
                    "area": area,
                    "bottom": bottom,
                    "lower": lower,
                    "center": center,
                    "upper_mid": upper_mid,
                }

        if best is None:
            result = dict(empty)
            result["reason"] = "no usable drivable component"
            result["reason_code"] = 1.0
            return result

        component = best["component"]
        bottom = best["bottom"]
        lower = best["lower"]
        center = best["center"]
        upper_mid = best["upper_mid"]
        bottom_connected = bottom["width"] > 0.0
        centerline_reached = center["width"] > 0.0
        upper_mid_reached = upper_mid["width"] > 0.0
        bottom_center = bottom["center_x"] if bottom["width"] > 0.0 else lower["center_x"]
        if bottom_center <= 0.0:
            bottom_center = center["center_x"]
        mid_center = lower["center_x"] if lower["width"] > 0.0 else bottom_center
        centerline_center = center["center_x"] if center["width"] > 0.0 else mid_center
        error_x = centerline_center - center_x
        slope = centerline_center - bottom_center
        row_has_pixels = np.any(component, axis=1)
        ys = np.flatnonzero(row_has_pixels)
        continuous_score = 0.0
        if len(ys) > 0:
            span_start = min(int(ys[0]), center_band[0])
            span_end = max(int(ys[-1]) + 1, bottom_band[1])
            span_rows = max(1, span_end - span_start)
            continuous_score = float(
                np.count_nonzero(row_has_pixels[span_start:span_end]) / span_rows
            )

        component_area = max(1, int(np.count_nonzero(component)))
        road_ratio = float(np.count_nonzero(component & road_mask) / component_area)
        bridge_ratio = float(np.count_nonzero(component & bridge_mask) / component_area)
        if road_ratio >= 0.20 and bridge_ratio >= 0.20:
            drivable_type = 3.0
        elif bridge_ratio > road_ratio:
            drivable_type = 2.0
        elif road_ratio > 0.0:
            drivable_type = 1.0
        else:
            drivable_type = 0.0

        side_view_likely = bool(
            abs(error_x) > width * 0.42
            or abs(slope) > self.drivable_corridor_max_slope_pixels
            or (
                bottom_connected
                and bottom["width_ratio"] < self.drivable_corridor_min_bottom_width_ratio * 0.7
                and abs(bottom_center - center_x) > width * 0.25
            )
            or (
                centerline_reached
                and center["width_ratio"] < self.drivable_corridor_min_center_width_ratio * 0.7
                and abs(centerline_center - center_x) > width * 0.25
            )
        )
        valid = (
            bottom_connected
            and centerline_reached
            and continuous_score >= self.drivable_corridor_min_continuous_score
            and bottom["width_ratio"] >= self.drivable_corridor_min_bottom_width_ratio
            and center["width_ratio"] >= self.drivable_corridor_min_center_width_ratio
            and not side_view_likely
        )
        reason = "valid drivable corridor"
        reason_code = 0.0
        if not bottom_connected:
            reason = "drivable mask does not touch lower frame"
            reason_code = 2.0
        elif not centerline_reached:
            reason = "drivable component does not reach center line"
            reason_code = 3.0
        elif bottom["width_ratio"] < self.drivable_corridor_min_bottom_width_ratio:
            reason = "bottom corridor too narrow"
            reason_code = 4.0
        elif center["width_ratio"] < self.drivable_corridor_min_center_width_ratio:
            reason = "center corridor too narrow"
            reason_code = 4.0
        elif continuous_score < self.drivable_corridor_min_continuous_score:
            reason = "corridor is not vertically continuous"
            reason_code = 5.0
        elif side_view_likely:
            reason = "side-view or sharply slanted corridor likely"
            reason_code = 6.0

        return {
            "corridor_valid": bool(valid),
            "bottom_connected": bool(bottom_connected),
            "centerline_reached": bool(centerline_reached),
            "upper_mid_reached": bool(upper_mid_reached),
            "continuous_score": float(continuous_score),
            "bottom_width_ratio": float(bottom["width_ratio"]),
            "center_width_ratio": float(center["width_ratio"]),
            "corridor_center_x_bottom": float(bottom_center),
            "corridor_center_x_mid": float(mid_center),
            "corridor_center_x_centerline": float(centerline_center),
            "corridor_error_x_pixels": float(error_x),
            "corridor_slope_pixels": float(slope),
            "road_ratio_in_corridor": float(road_ratio),
            "bridge_ratio_in_corridor": float(bridge_ratio),
            "drivable_type": float(drivable_type),
            "side_view_likely": bool(side_view_likely),
            "image_width": float(width),
            "image_height": float(height),
            "reason_code": float(reason_code),
            "reason": reason,
            "component_mask": component,
            "bands": {
                "bottom": bottom_band,
                "lower": lower_band,
                "center": center_band,
                "upper_mid": upper_mid_band,
            },
        }

    def _clamped_row(self, height, ratio):
        if height <= 0:
            return 0
        ratio = max(0.0, min(0.99, float(ratio)))
        return max(0, min(height - 1, int(height * ratio)))

    def _drivable_band_stats(self, mask, y_start, y_end):
        height, width = mask.shape
        y_start = max(0, min(height, int(y_start)))
        y_end = max(y_start, min(height, int(y_end)))
        if y_end <= y_start or width <= 0:
            return {"center_x": 0.0, "width": 0.0, "width_ratio": 0.0}
        xs = []
        row_widths = []
        min_pixels_per_row = max(2, int(width * 0.006))
        for row in mask[y_start:y_end, :]:
            row_xs = np.flatnonzero(row)
            if len(row_xs) < min_pixels_per_row:
                continue
            xs.extend(row_xs.tolist())
            row_widths.append(float(row_xs[-1] - row_xs[0] + 1))
        if not xs or not row_widths:
            return {"center_x": 0.0, "width": 0.0, "width_ratio": 0.0}
        width_px = float(np.median(row_widths))
        return {
            "center_x": float(np.median(xs)),
            "width": width_px,
            "width_ratio": float(width_px / max(1, width)),
        }

    def publish_drivable_corridor_info(self, corridor):
        if not hasattr(self, "drivable_corridor_pub"):
            return
        msg = Float32MultiArray()
        msg.data = [
            1.0 if corridor["corridor_valid"] else 0.0,
            1.0 if corridor["bottom_connected"] else 0.0,
            1.0 if corridor["centerline_reached"] else 0.0,
            1.0 if corridor["upper_mid_reached"] else 0.0,
            corridor["continuous_score"],
            corridor["bottom_width_ratio"],
            corridor["center_width_ratio"],
            corridor["corridor_center_x_bottom"],
            corridor["corridor_center_x_mid"],
            corridor["corridor_center_x_centerline"],
            corridor["corridor_error_x_pixels"],
            corridor["corridor_slope_pixels"],
            corridor["road_ratio_in_corridor"],
            corridor["bridge_ratio_in_corridor"],
            corridor["drivable_type"],
            1.0 if corridor["side_view_likely"] else 0.0,
            corridor["image_width"],
            corridor["image_height"],
            corridor["reason_code"],
        ]
        self.drivable_corridor_pub.publish(msg)

    def publish_drivable_corridor_debug(self, corridor):
        if not hasattr(self, "drivable_corridor_debug_pub"):
            return
        msg = String()
        msg.data = (
            f"valid={corridor['corridor_valid']} "
            f"bottom_connected={corridor['bottom_connected']} "
            f"centerline_reached={corridor['centerline_reached']} "
            f"score={corridor['continuous_score']:.2f} "
            f"bottom_width={corridor['bottom_width_ratio']:.2f} "
            f"center_width={corridor['center_width_ratio']:.2f} "
            f"error_x={corridor['corridor_error_x_pixels']:.1f} "
            f"slope={corridor['corridor_slope_pixels']:.1f} "
            f"type={corridor['drivable_type']:.0f} "
            f"side_view={corridor['side_view_likely']} "
            f"reason={corridor['reason']}"
        )
        self.drivable_corridor_debug_pub.publish(msg)

    def _draw_drivable_corridor_overlay(self, image, corridor):
        overlay = image.copy()
        height, width = image.shape[:2]
        component = corridor.get("component_mask")
        if component is not None and component.shape[:2] == (height, width):
            color = (0, 210, 255) if corridor["drivable_type"] != 2.0 else (255, 180, 0)
            component_layer = np.zeros_like(image)
            component_layer[component.astype(bool)] = color
            overlay = cv2.addWeighted(overlay, 1.0, component_layer, 0.35, 0.0)

        bands = corridor.get("bands", {})
        for name, band_color in (
            ("bottom", (0, 255, 255)),
            ("lower", (255, 255, 0)),
            ("center", (0, 255, 0)),
        ):
            if name not in bands:
                continue
            y0, y1 = bands[name]
            cv2.rectangle(overlay, (0, int(y0)), (width - 1, max(int(y0), int(y1) - 1)), band_color, 1)

        points = []
        for key, y_key in (
            ("corridor_center_x_bottom", "bottom"),
            ("corridor_center_x_mid", "lower"),
            ("corridor_center_x_centerline", "center"),
        ):
            x = corridor.get(key, 0.0)
            band = bands.get(y_key)
            if x > 0.0 and band is not None:
                y = int((band[0] + band[1]) * 0.5)
                point = (int(round(x)), y)
                points.append(point)
                cv2.circle(overlay, point, 5, (0, 0, 255), -1)
        for first, second in zip(points, points[1:]):
            cv2.line(overlay, first, second, (0, 0, 255), 2)

        cv2.line(overlay, (width // 2, 0), (width // 2, height - 1), (255, 255, 255), 1)
        text_lines = [
            f"corridor valid={corridor['corridor_valid']} type={corridor['drivable_type']:.0f}",
            f"err={corridor['corridor_error_x_pixels']:.0f}px score={corridor['continuous_score']:.2f}",
            corridor["reason"],
        ]
        for index, text in enumerate(text_lines):
            cv2.putText(
                overlay,
                text,
                (12, 26 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        return overlay

    def publish_segmentation_info(self, road_mask, bridge_mask, header=None):
        """
        data layout:
        [road_found, road_delta_x, road_area_ratio, road_bottom_coverage,
         bridge_found, bridge_delta_x, bridge_area_ratio, bridge_bottom_coverage,
         image_width, image_height,
         road_center_x, road_center_y_ratio, road_top_y_ratio, road_bottom_y_ratio,
         bridge_center_x, bridge_center_y_ratio, bridge_top_y_ratio, bridge_bottom_y_ratio,
         connected, connection_score, connection_delta_x, connection_gap_y_ratio,
         road_bottom_center_x, road_top_center_x, bridge_bottom_center_x, bridge_mid_center_x,
         central_connection_score, central_connection_delta_x,
         road_bottom_left_x, road_bottom_right_x, road_mid_left_x, road_mid_right_x,
         road_bottom_width_ratio,
         bridge_bottom_left_x, bridge_bottom_right_x, bridge_mid_left_x, bridge_mid_right_x,
         bridge_bottom_width_ratio,
         bridge_entry_u, bridge_entry_v, bridge_entry_confidence,
         bridge_entry_from_road_connection, bridge_centerline_slope_pixels,
         bridge_frontalness_score, bridge_entry_depth,
         bridge_target_u, bridge_target_v, bridge_target_confidence,
         bridge_entry_confirmed, bridge_entry_gate_left_u, bridge_entry_gate_right_u,
         bridge_entry_gate_center_u, bridge_entry_gate_v,
         bridge_entry_gate_width_pixels,
        bridge_pre_entry_u, bridge_pre_entry_v, bridge_pre_entry_confidence,
         bridge_pre_entry_depth, bridge_entry_gate_confirmed,
         bridge_ramp_valid, bridge_ramp_confidence, bridge_ramp_lower_present,
         bridge_ramp_continuous, bridge_side_view_score,
         bridge_vertical_coverage_score, bridge_ramp_reason_code]
        """
        msg = Float32MultiArray()
        height, width = road_mask.shape
        road = self._segmentation_stats(road_mask)
        bridge = self._segmentation_stats(bridge_mask)
        connection = self._segmentation_connection_stats(road_mask, bridge_mask)
        entry = self._bridge_entry_stats(road_mask, bridge_mask, bridge, header)
        ramp = self._compute_bridge_ramp_quality(bridge_mask, road_mask)
        msg.data = [
            road["found"],
            road["delta_x"],
            road["area_ratio"],
            road["bottom_coverage"],
            bridge["found"],
            bridge["delta_x"],
            bridge["area_ratio"],
            bridge["bottom_coverage"],
            float(width),
            float(height),
            road["center_x"],
            road["center_y_ratio"],
            road["top_y_ratio"],
            road["bottom_y_ratio"],
            bridge["center_x"],
            bridge["center_y_ratio"],
            bridge["top_y_ratio"],
            bridge["bottom_y_ratio"],
            connection["connected"],
            connection["score"],
            connection["delta_x"],
            connection["gap_y_ratio"],
            road["bottom_center_x"],
            road["top_center_x"],
            bridge["bottom_center_x"],
            bridge["mid_center_x"],
            connection["central_score"],
            connection["central_delta_x"],
            road["bottom_left_x"],
            road["bottom_right_x"],
            road["mid_left_x"],
            road["mid_right_x"],
            road["bottom_width_ratio"],
            bridge["bottom_left_x"],
            bridge["bottom_right_x"],
            bridge["mid_left_x"],
            bridge["mid_right_x"],
            bridge["bottom_width_ratio"],
            entry["u"],
            entry["v"],
            entry["confidence"],
            entry["from_road_connection"],
            entry["centerline_slope_pixels"],
            entry["frontalness"],
            entry["depth"],
            entry["target_u"],
            entry["target_v"],
            entry["target_confidence"],
            entry["entry_confirmed"],
            entry["gate_left_u"],
            entry["gate_right_u"],
            entry["gate_center_u"],
            entry["gate_v"],
            entry["gate_width_pixels"],
            entry["pre_entry_u"],
            entry["pre_entry_v"],
            entry["pre_entry_confidence"],
            entry["pre_entry_depth"],
            entry["entry_confirmed"],
            1.0 if ramp["ramp_valid"] else 0.0,
            ramp["ramp_confidence"],
            1.0 if ramp["lower_present"] else 0.0,
            1.0 if ramp["continuous_lower_to_upper"] else 0.0,
            ramp["side_view_score"],
            ramp["vertical_coverage_score"],
            ramp["reason_code"],
        ]
        self.segmentation_info_pub.publish(msg)

    def _segmentation_stats(self, mask):
        height, width = mask.shape
        area = int(np.count_nonzero(mask))
        if area <= 0:
            return {
                "found": 0.0,
                "delta_x": 0.0,
                "area_ratio": 0.0,
                "bottom_coverage": 0.0,
                "center_x": 0.0,
                "center_y_ratio": 0.0,
                "top_y_ratio": 0.0,
                "bottom_y_ratio": 0.0,
                "bottom_center_x": 0.0,
                "top_center_x": 0.0,
                "mid_center_x": 0.0,
                "bottom_left_x": 0.0,
                "bottom_right_x": 0.0,
                "mid_left_x": 0.0,
                "mid_right_x": 0.0,
                "bottom_width_ratio": 0.0,
            }

        ys, xs = np.nonzero(mask)
        bottom_start = int(height * 0.65)
        bottom_roi = mask[bottom_start:, :]
        bottom_area = max(1, bottom_roi.size)
        top_end = int(height * 0.45)
        mid_start = int(height * 0.40)
        mid_end = int(height * 0.75)
        bottom_left_x, bottom_right_x, bottom_width_ratio = self._mask_roi_bounds(
            mask, bottom_start, height
        )
        mid_left_x, mid_right_x, _ = self._mask_roi_bounds(mask, mid_start, mid_end)
        return {
            "found": 1.0,
            "delta_x": float(np.mean(xs) - (width / 2.0)),
            "area_ratio": float(area / max(1, mask.size)),
            "bottom_coverage": float(np.count_nonzero(bottom_roi) / bottom_area),
            "center_x": float(np.mean(xs)),
            "center_y_ratio": float(np.mean(ys) / max(1, height)),
            "top_y_ratio": float(np.min(ys) / max(1, height)),
            "bottom_y_ratio": float(np.max(ys) / max(1, height)),
            "bottom_center_x": self._mask_roi_center_x(mask, bottom_start, height),
            "top_center_x": self._mask_roi_center_x(mask, 0, top_end),
            "mid_center_x": self._mask_roi_center_x(mask, mid_start, mid_end),
            "bottom_left_x": bottom_left_x,
            "bottom_right_x": bottom_right_x,
            "mid_left_x": mid_left_x,
            "mid_right_x": mid_right_x,
            "bottom_width_ratio": bottom_width_ratio,
        }

    def _mask_roi_center_x(self, mask, y_start, y_end):
        height, _ = mask.shape
        y_start = max(0, min(height, int(y_start)))
        y_end = max(y_start, min(height, int(y_end)))
        if y_end <= y_start:
            return 0.0
        roi = mask[y_start:y_end, :]
        _, xs = np.nonzero(roi)
        if len(xs) <= 0:
            return 0.0
        return float(np.mean(xs))

    def _mask_roi_bounds(self, mask, y_start, y_end):
        height, width = mask.shape
        y_start = max(0, min(height, int(y_start)))
        y_end = max(y_start, min(height, int(y_end)))
        if y_end <= y_start or width <= 0:
            return 0.0, 0.0, 0.0

        min_pixels_per_row = max(3, int(width * 0.01))
        left_edges = []
        right_edges = []
        for row in mask[y_start:y_end, :]:
            xs = np.flatnonzero(row)
            if len(xs) < min_pixels_per_row:
                continue
            left_edges.append(float(xs[0]))
            right_edges.append(float(xs[-1]))

        if len(left_edges) < 3:
            return 0.0, 0.0, 0.0

        left_x = float(np.median(left_edges))
        right_x = float(np.median(right_edges))
        if right_x <= left_x:
            return 0.0, 0.0, 0.0
        return left_x, right_x, float((right_x - left_x + 1.0) / width)

    def _compute_bridge_ramp_quality(self, bridge_mask, road_mask=None):
        height, width = bridge_mask.shape
        empty = {
            "ramp_valid": False,
            "ramp_confidence": 0.0,
            "lower_present": False,
            "centered": False,
            "continuous_lower_to_upper": False,
            "centerline_slope_pixels": 0.0,
            "bottom_width_ratio": 0.0,
            "mid_width_ratio": 0.0,
            "top_width_ratio": 0.0,
            "vertical_coverage_score": 0.0,
            "side_view_score": 1.0,
            "reason": "no bridge mask",
            "reason_code": 1.0,
        }
        if height <= 0 or width <= 0 or int(np.count_nonzero(bridge_mask)) <= 0:
            return empty

        center_x = width * 0.5
        center_half_width = max(12, int(width * 0.18))
        lower_start = int(height * 0.68)
        mid_start = int(height * 0.45)
        mid_end = int(height * 0.72)
        top_end = int(height * 0.45)

        lower_roi = bridge_mask[lower_start:, :]
        center_lower_roi = bridge_mask[lower_start:, max(0, int(center_x) - center_half_width): min(width, int(center_x) + center_half_width)]
        lower_present = (
            int(np.count_nonzero(lower_roi)) >= max(12, int(width * 0.025))
            and int(np.count_nonzero(center_lower_roi)) >= max(4, int(width * 0.006))
        )

        labels, label_img, stats, _ = cv2.connectedComponentsWithStats(
            bridge_mask.astype(np.uint8), 8
        )
        best = None
        for label in range(1, labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(20, int(width * height * 0.002)):
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            touches_lower = y + h >= lower_start
            overlaps_center = x <= center_x + center_half_width and x + w >= center_x - center_half_width
            score = area + (1500 if touches_lower else 0) + (1200 if overlaps_center else 0) + h * 8
            if best is None or score > best["score"]:
                best = {
                    "label": label,
                    "score": score,
                    "area": area,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }

        if best is None:
            result = dict(empty)
            result.update({"lower_present": lower_present, "reason": "no usable bridge component", "reason_code": 2.0})
            return result

        component = label_img == best["label"]
        row_centers = []
        row_widths = []
        row_ys = []
        min_pixels_per_row = max(3, int(width * 0.008))
        for y in range(max(0, best["y"]), min(height, best["y"] + best["h"])):
            xs = np.flatnonzero(component[y, :])
            if len(xs) < min_pixels_per_row:
                continue
            row_ys.append(float(y))
            row_centers.append(float(np.median(xs)))
            row_widths.append(float(xs[-1] - xs[0] + 1))

        if len(row_ys) < 4:
            result = dict(empty)
            result.update({"lower_present": lower_present, "reason": "bridge component has too few rows", "reason_code": 3.0})
            return result

        row_ys_np = np.asarray(row_ys, dtype=np.float32)
        row_centers_np = np.asarray(row_centers, dtype=np.float32)
        row_widths_np = np.asarray(row_widths, dtype=np.float32)
        lower_rows = row_ys_np >= lower_start
        mid_rows = (row_ys_np >= mid_start) & (row_ys_np <= mid_end)
        top_rows = row_ys_np <= top_end

        bottom_center = float(np.median(row_centers_np[lower_rows])) if np.any(lower_rows) else float(row_centers_np[-1])
        mid_center = float(np.median(row_centers_np[mid_rows])) if np.any(mid_rows) else float(np.median(row_centers_np))
        top_center = float(np.median(row_centers_np[top_rows])) if np.any(top_rows) else mid_center
        centerline_slope = top_center - bottom_center
        bottom_width_ratio = (
            float(np.median(row_widths_np[lower_rows]) / max(1, width))
            if np.any(lower_rows)
            else 0.0
        )
        mid_width_ratio = (
            float(np.median(row_widths_np[mid_rows]) / max(1, width))
            if np.any(mid_rows)
            else float(np.median(row_widths_np) / max(1, width))
        )
        top_width_ratio = (
            float(np.median(row_widths_np[top_rows]) / max(1, width))
            if np.any(top_rows)
            else 0.0
        )

        span = max(1.0, float(np.max(row_ys_np) - np.min(row_ys_np) + 1.0))
        vertical_coverage_score = min(1.0, len(row_ys) / span)
        upward_extent_score = min(1.0, max(0.0, (height - np.min(row_ys_np)) / max(1.0, height * 0.75)))
        centered_error = abs(0.65 * (bottom_center - center_x) + 0.35 * (mid_center - center_x))
        centered_score = max(0.0, 1.0 - centered_error / max(1.0, width * 0.22))
        centered = centered_error <= width * 0.16
        slope_score = max(0.0, 1.0 - abs(centerline_slope) / max(1.0, width * 0.28))
        lower_score = 1.0 if lower_present else 0.0
        width_ok = 0.08 <= bottom_width_ratio <= 0.78 and 0.05 <= mid_width_ratio <= 0.85
        width_score = 1.0 if width_ok else max(0.0, min(1.0, bottom_width_ratio / 0.16))
        continuous_lower_to_upper = (
            lower_present
            and vertical_coverage_score >= 0.45
            and upward_extent_score >= 0.55
            and bool(np.any(mid_rows))
        )
        continuity_score = 0.65 * vertical_coverage_score + 0.35 * upward_extent_score
        side_view_score = min(
            1.0,
            0.35 * (1.0 - centered_score)
            + 0.25 * (1.0 - slope_score)
            + 0.25 * (1.0 - continuity_score)
            + 0.15 * (0.0 if width_ok else 1.0),
        )
        ramp_confidence = min(
            1.0,
            0.28 * lower_score
            + 0.24 * centered_score
            + 0.24 * continuity_score
            + 0.14 * slope_score
            + 0.10 * width_score,
        )

        reason = "valid frontal bridge ramp"
        reason_code = 0.0
        if not lower_present:
            reason = "bridge ramp missing lower centered mask"
            reason_code = 4.0
        elif not centered:
            reason = "bridge ramp not centered"
            reason_code = 5.0
        elif not continuous_lower_to_upper:
            reason = "bridge ramp not vertically continuous"
            reason_code = 6.0
        elif not width_ok:
            reason = "bridge ramp width implausible"
            reason_code = 7.0
        elif side_view_score > 0.55:
            reason = "bridge side-view likely"
            reason_code = 8.0

        ramp_valid = (
            lower_present
            and centered
            and continuous_lower_to_upper
            and width_ok
            and ramp_confidence >= 0.55
            and side_view_score <= 0.55
        )
        return {
            "ramp_valid": bool(ramp_valid),
            "ramp_confidence": float(ramp_confidence),
            "lower_present": bool(lower_present),
            "centered": bool(centered),
            "continuous_lower_to_upper": bool(continuous_lower_to_upper),
            "centerline_slope_pixels": float(centerline_slope),
            "bottom_width_ratio": float(bottom_width_ratio),
            "mid_width_ratio": float(mid_width_ratio),
            "top_width_ratio": float(top_width_ratio),
            "vertical_coverage_score": float(vertical_coverage_score),
            "side_view_score": float(side_view_score),
            "reason": reason,
            "reason_code": float(reason_code),
        }

    def _bridge_entry_stats(self, road_mask, bridge_mask, bridge_stats, header=None):
        height, width = bridge_mask.shape
        target_u = float(bridge_stats.get("bottom_center_x", 0.0))
        if target_u <= 0.0:
            target_u = float(bridge_stats.get("center_x", 0.0))
        target_v = float(bridge_stats.get("bottom_y_ratio", 0.0)) * height
        target_confidence = 0.45 if target_u > 0.0 and target_v > 0.0 else 0.0
        empty = {
            "u": 0.0,
            "v": 0.0,
            "confidence": 0.0,
            "from_road_connection": 0.0,
            "centerline_slope_pixels": 0.0,
            "frontalness": 0.0,
            "depth": 0.0,
            "target_u": target_u,
            "target_v": target_v,
            "target_confidence": target_confidence,
            "entry_confirmed": 0.0,
            "gate_left_u": 0.0,
            "gate_right_u": 0.0,
            "gate_center_u": 0.0,
            "gate_v": 0.0,
            "gate_width_pixels": 0.0,
            "pre_entry_u": 0.0,
            "pre_entry_v": 0.0,
            "pre_entry_confidence": 0.0,
            "pre_entry_depth": 0.0,
        }
        if int(np.count_nonzero(bridge_mask)) <= 0:
            self.bridge_mask_found_last = False
            self.bridge_entry_confirmed_last = False
            self.last_bridge_geometry_reason = "no bridge mask"
            return empty
        self.bridge_mask_found_last = True

        kernel_size = max(1, int(self.segmentation_connection_dilation_pixels))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        road_dilated = cv2.dilate(road_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        contact = road_dilated & bridge_mask

        center_margin = int(width * max(0.0, min(0.48, self.bridge_entry_contact_center_margin_ratio)))
        lower_start = int(height * max(0.0, min(0.95, self.bridge_entry_contact_lower_y_ratio)))
        roi = contact[lower_start:, center_margin : width - center_margin]
        component = self._best_bridge_contact_component(
            roi, width, height, x_offset=center_margin, y_offset=lower_start
        )
        from_connection = component is not None

        u = 0.0
        v = 0.0
        confidence = 0.0
        gate_left = 0.0
        gate_right = 0.0
        gate_center = 0.0
        gate_v = 0.0
        gate_width = 0.0
        pre_entry_u = 0.0
        pre_entry_v = 0.0
        pre_entry_confidence = 0.0
        pre_entry_depth = 0.0
        if component is not None:
            xs = component["xs"] + center_margin
            ys = component["ys"] + lower_start
            expand = max(0, int(self.bridge_entry_gate_expand_pixels))
            gate_left = float(max(0, int(np.min(xs)) - expand))
            gate_right = float(min(width - 1, int(np.max(xs)) + expand))
            gate_width = max(0.0, gate_right - gate_left)
            gate_center = float(np.median(xs))
            gate_v = float(np.percentile(ys, 65.0))
            u = gate_center
            v = gate_v
            confidence = min(
                1.0,
                component["count"] / max(20.0, width * 0.08)
                + gate_width / max(1.0, width) * 0.5,
            )
            pre_entry = self._bridge_pre_entry_from_contact(
                road_mask, gate_center, gate_v, gate_width
            )
            if pre_entry is not None:
                pre_entry_u = pre_entry["u"]
                pre_entry_v = pre_entry["v"]
                pre_entry_confidence = pre_entry["confidence"]
                depth = self._depth_patch_meters(pre_entry_u, pre_entry_v, header)
                pre_entry_depth = depth if depth > 0.0 else 0.0
            self.bridge_entry_confirmed_last = True
            self.last_bridge_geometry_reason = "confirmed road-bridge contact"
        else:
            self.bridge_entry_confirmed_last = False
            self.last_bridge_geometry_reason = "no confirmed road-bridge contact"

        bottom_center = float(bridge_stats.get("bottom_center_x", 0.0))
        mid_center = float(bridge_stats.get("mid_center_x", 0.0))
        centerline_slope = mid_center - bottom_center if bottom_center > 0.0 and mid_center > 0.0 else 999.0
        bottom_width_ratio = float(bridge_stats.get("bottom_width_ratio", 0.0))
        edges_visible = (
            float(bridge_stats.get("bottom_left_x", 0.0)) > 0.0
            and float(bridge_stats.get("bottom_right_x", 0.0)) > 0.0
            and float(bridge_stats.get("mid_left_x", 0.0)) > 0.0
            and float(bridge_stats.get("mid_right_x", 0.0)) > 0.0
        )
        alignment_score = max(0.0, 1.0 - abs(centerline_slope) / max(1.0, width * 0.18))
        width_score = max(0.0, min(1.0, (bottom_width_ratio - 0.12) / 0.28))
        edge_score = 1.0 if edges_visible else 0.25
        connection_score = 1.0 if from_connection else 0.35
        frontalness = float(min(1.0, alignment_score * 0.35 + width_score * 0.25 + edge_score * 0.25 + connection_score * 0.15))
        depth = self._depth_patch_meters(u, v, header) if confidence > 0.0 else -1.0

        return {
            "u": u,
            "v": v,
            "confidence": float(confidence),
            "from_road_connection": 1.0 if from_connection else 0.0,
            "centerline_slope_pixels": float(centerline_slope),
            "frontalness": frontalness,
            "depth": depth if depth > 0.0 else 0.0,
            "target_u": target_u,
            "target_v": target_v,
            "target_confidence": target_confidence,
            "entry_confirmed": 1.0 if from_connection else 0.0,
            "gate_left_u": gate_left,
            "gate_right_u": gate_right,
            "gate_center_u": gate_center,
            "gate_v": gate_v,
            "gate_width_pixels": gate_width,
            "pre_entry_u": pre_entry_u,
            "pre_entry_v": pre_entry_v,
            "pre_entry_confidence": pre_entry_confidence,
            "pre_entry_depth": pre_entry_depth,
        }

    def _bridge_pre_entry_from_contact(self, road_mask, gate_center_u, gate_v, gate_width):
        height, width = road_mask.shape
        if gate_center_u <= 0.0 or gate_v <= 0.0:
            return None
        center = int(round(gate_center_u))
        half_width = max(8, int(max(gate_width * 0.5, width * 0.05)))
        y_start = min(height - 1, int(round(gate_v)) + 3)
        y_end = min(height, y_start + max(15, int(height * 0.22)))
        if y_start >= y_end:
            return None

        candidates = []
        for row_y in range(y_start, y_end):
            x1 = max(0, center - half_width)
            x2 = min(width, center + half_width + 1)
            xs = np.flatnonzero(road_mask[row_y, x1:x2])
            if len(xs) <= 0:
                continue
            full_xs = xs + x1
            candidates.append((row_y, full_xs))

        if not candidates:
            return None
        # Prefer a row visibly on the road side, close to the robot but still near the gate.
        row_y, xs = candidates[min(len(candidates) - 1, max(0, len(candidates) // 2))]
        return {
            "u": float(np.median(xs)),
            "v": float(row_y),
            "confidence": min(1.0, len(xs) / max(8.0, half_width * 0.6)),
        }

    def _best_bridge_contact_component(self, roi, full_width, full_height, x_offset=0, y_offset=0):
        min_pixels = max(1, int(self.bridge_entry_contact_min_pixels))
        min_width = max(1, int(self.bridge_entry_contact_min_width_pixels))
        if roi.size <= 0 or int(np.count_nonzero(roi)) < min_pixels:
            return None

        labels, label_img, stats, _ = cv2.connectedComponentsWithStats(
            roi.astype(np.uint8), 8
        )
        best = None
        image_center = full_width * 0.5
        for label in range(1, labels):
            count = int(stats[label, cv2.CC_STAT_AREA])
            if count < min_pixels:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if w < min_width or h < 2:
                continue
            ys, xs = np.nonzero(label_img == label)
            if len(xs) <= 0:
                continue
            full_x = xs + x_offset
            full_y = ys + y_offset
            center_score = 1.0 - min(1.0, abs(float(np.median(full_x)) - image_center) / max(1.0, image_center))
            lower_score = min(1.0, float(np.median(full_y)) / max(1.0, full_height))
            score = count + w * 2.0 + center_score * 50.0 + lower_score * 25.0
            candidate = {"score": score, "count": count, "xs": xs, "ys": ys}
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best

    def _stamp_to_seconds(self, stamp):
        if stamp is None:
            return None
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _depth_source(self):
        if self.latest_depth_image_raw is not None:
            return self.latest_depth_image_raw, self.latest_depth_stamp_raw, self.latest_depth_frame_raw
        if self.latest_depth_image_compressed is not None:
            return (
                self.latest_depth_image_compressed,
                self.latest_depth_stamp_compressed,
                self.latest_depth_frame_compressed,
            )
        return None, None, ""

    def _depth_patch_meters(self, u, v, header=None, radius_override=None):
        depth_image, depth_stamp, _ = self._depth_source()
        if depth_image is None:
            self.rejected_depth_samples += 1
            return -1.0
        if header is not None and depth_stamp is not None:
            rgb_time = self._stamp_to_seconds(header.stamp)
            depth_time = self._stamp_to_seconds(depth_stamp)
            if (
                rgb_time is not None
                and depth_time is not None
                and abs(rgb_time - depth_time) > self.bridge_geometry_max_depth_age
            ):
                self.rejected_depth_samples += 1
                return -1.0

        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        height, width = depth_image.shape[:2]
        u = int(round(float(u)))
        v = int(round(float(v)))
        if u < 0 or v < 0 or u >= width or v >= height:
            self.rejected_depth_samples += 1
            return -1.0
        radius = max(
            0,
            int(
                self.bridge_geometry_depth_patch_radius
                if radius_override is None
                else radius_override
            ),
        )
        x1 = max(0, u - radius)
        x2 = min(width, u + radius + 1)
        y1 = max(0, v - radius)
        y2 = min(height, v + radius + 1)
        patch = depth_image[y1:y2, x1:x2].astype(np.float32).reshape(-1)
        patch = patch[np.isfinite(patch)]
        if len(patch) == 0:
            self.rejected_depth_samples += 1
            return -1.0
        if np.nanmedian(patch) > 50.0:
            patch = patch / 1000.0
        valid = patch[
            (patch >= self.bridge_geometry_min_valid_depth)
            & (patch <= self.bridge_geometry_max_valid_depth)
        ]
        if len(valid) == 0:
            self.rejected_depth_samples += 1
            return -1.0
        self.valid_depth_samples += 1
        return float(np.median(valid))

    def _camera_intrinsics_valid(self):
        return self.bridge_camera_fx > 0.0 and self.bridge_camera_fy > 0.0

    def _project_pixel_to_camera(self, u, v, header=None):
        if not self._camera_intrinsics_valid():
            self.last_bridge_geometry_reason = "missing camera intrinsics"
            if not self.warned_missing_camera_info:
                self.warned_missing_camera_info = True
                self.get_logger().warn(
                    "Bridge metric projection disabled: no CameraInfo or valid "
                    "bridge_camera_fx/fy fallback parameters."
                )
            return None
        depth = self._depth_patch_meters(u, v, header)
        if depth <= 0.0:
            self.last_bridge_geometry_reason = "missing or stale valid depth"
            return None
        x = (float(u) - self.bridge_camera_cx) * depth / self.bridge_camera_fx
        y = (float(v) - self.bridge_camera_cy) * depth / self.bridge_camera_fy
        return (float(x), float(y), float(depth))

    def _project_pixel_to_camera_with_depth(self, u, v, depth):
        if not self._camera_intrinsics_valid() or depth <= 0.0:
            return None
        x = (float(u) - self.bridge_camera_cx) * depth / self.bridge_camera_fx
        y = (float(v) - self.bridge_camera_cy) * depth / self.bridge_camera_fy
        return (float(x), float(y), float(depth))

    def _segmentation_connection_stats(self, road_mask, bridge_mask):
        height, width = road_mask.shape
        road_area = int(np.count_nonzero(road_mask))
        bridge_area = int(np.count_nonzero(bridge_mask))
        if road_area <= 0 or bridge_area <= 0:
            return {
                "connected": 0.0,
                "score": 0.0,
                "delta_x": 0.0,
                "gap_y_ratio": 1.0,
                "central_score": 0.0,
                "central_delta_x": 0.0,
            }

        kernel_size = max(1, int(self.segmentation_connection_dilation_pixels))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        road_u8 = road_mask.astype(np.uint8)
        bridge_u8 = bridge_mask.astype(np.uint8)
        road_dilated = cv2.dilate(road_u8, kernel, iterations=1).astype(bool)
        bridge_dilated = cv2.dilate(bridge_u8, kernel, iterations=1).astype(bool)
        contact = (road_dilated & bridge_mask) | (bridge_dilated & road_mask)
        contact_count = int(np.count_nonzero(contact))

        road_ys, _ = np.nonzero(road_mask)
        bridge_ys, _ = np.nonzero(bridge_mask)
        vertical_gap = max(
            0,
            max(int(np.min(road_ys)), int(np.min(bridge_ys)))
            - min(int(np.max(road_ys)), int(np.max(bridge_ys))),
        )
        gap_y_ratio = float(vertical_gap / max(1, height))

        if contact_count <= 0:
            return {
                "connected": 0.0,
                "score": 0.0,
                "delta_x": 0.0,
                "gap_y_ratio": gap_y_ratio,
                "central_score": 0.0,
                "central_delta_x": 0.0,
            }

        _, contact_xs = np.nonzero(contact)
        center_margin = int(width * 0.30)
        central_contact = contact[:, center_margin : width - center_margin]
        _, central_xs = np.nonzero(central_contact)
        score = contact_count / max(1, min(road_area, bridge_area))
        return {
            "connected": 1.0,
            "score": float(score),
            "delta_x": float(np.mean(contact_xs) - (width / 2.0)),
            "gap_y_ratio": gap_y_ratio,
            "central_score": float(np.count_nonzero(central_contact) / max(1, contact_count)),
            "central_delta_x": (
                float((np.mean(central_xs) + center_margin) - (width / 2.0))
                if len(central_xs) > 0
                else 0.0
            ),
        }

    def publish_target_info(self, found, distance, delta_x):
        """發佈目標資訊 (找到目標, 距離)"""
        msg = Float32MultiArray()
        msg.data = [float(found), float(distance), float(delta_x)]
        self.target_pub.publish(msg)

    def publish_target_bbox(self, found, target):
        """
        發佈目前鎖定目標的 bounding box。
        data layout:
        [found, center_x, center_y, width, height, x1, y1, x2, y2,
         image_width, image_height, confidence, distance]
        """
        msg = Float32MultiArray()
        if found and target is not None:
            msg.data = [
                1.0,
                float(target["center_x"]),
                float(target["center_y"]),
                float(target["width"]),
                float(target["height"]),
                float(target["x1"]),
                float(target["y1"]),
                float(target["x2"]),
                float(target["y2"]),
                float(target["image_width"]),
                float(target["image_height"]),
                float(target["confidence"]),
                float(target["depth"]),
            ]
        else:
            msg.data = [0.0] * 13
        self.target_bbox_pub.publish(msg)

    def publish_target_surface_info(self, target, bridge_mask):
        """
        Publish selected target/bridge association from the same image callback.
        data layout:
        [target_found, bridge_found, bbox_bridge_overlap_ratio,
         bbox_lower_half_bridge_overlap_ratio, target_center_on_bridge,
         target_bottom_center_on_bridge, image_width, image_height,
         target_left_side_bridge_contact, target_right_side_bridge_contact,
         target_side_bridge_contact, target_side_bridge_contact_ratio,
         target_side_bridge_contact_pixels]
        """
        msg = Float32MultiArray()
        height, width = bridge_mask.shape
        bridge_found = float(np.count_nonzero(bridge_mask) > 0)
        if target is None:
            msg.data = [
                0.0,
                bridge_found,
                0.0,
                0.0,
                0.0,
                0.0,
                float(width),
                float(height),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            self.target_surface_pub.publish(msg)
            return

        x1 = max(0, min(width, int(target.get("x1", 0))))
        y1 = max(0, min(height, int(target.get("y1", 0))))
        x2 = max(x1, min(width, int(target.get("x2", 0))))
        y2 = max(y1, min(height, int(target.get("y2", 0))))
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        bbox_bridge_overlap = 0.0
        lower_bridge_overlap = 0.0
        left_side_contact = 0.0
        right_side_contact = 0.0
        side_contact = 0.0
        side_contact_ratio = 0.0
        side_contact_pixels = 0.0

        if x2 > x1 and y2 > y1:
            bbox_bridge_overlap = float(np.count_nonzero(bridge_mask[y1:y2, x1:x2]) / bbox_area)
            lower_y1 = y1 + max(0, (y2 - y1) // 2)
            lower_area = max(1, (x2 - x1) * (y2 - lower_y1))
            lower_bridge_overlap = float(
                np.count_nonzero(bridge_mask[lower_y1:y2, x1:x2]) / lower_area
            )
            side_margin = max(2, int(width * 0.01))
            left_x0 = max(0, x1 - side_margin)
            left_x1 = min(width, x1 + side_margin + 1)
            right_x0 = max(0, x2 - side_margin - 1)
            right_x1 = min(width, x2 + side_margin)
            side_y0 = y1
            side_y1 = y2
            left_area = max(1, (left_x1 - left_x0) * (side_y1 - side_y0))
            right_area = max(1, (right_x1 - right_x0) * (side_y1 - side_y0))
            left_pixels = int(np.count_nonzero(bridge_mask[side_y0:side_y1, left_x0:left_x1]))
            right_pixels = int(np.count_nonzero(bridge_mask[side_y0:side_y1, right_x0:right_x1]))
            side_contact_pixels = float(left_pixels + right_pixels)
            side_contact_ratio = float(side_contact_pixels / max(1, left_area + right_area))
            min_side_contact_pixels = max(3, int((side_y1 - side_y0) * 0.05))
            left_side_contact = float(left_pixels >= min_side_contact_pixels)
            right_side_contact = float(right_pixels >= min_side_contact_pixels)
            side_contact = float(bool(left_side_contact or right_side_contact))

        center_x = max(0, min(width - 1, int(target.get("center_x", 0))))
        center_y = max(0, min(height - 1, int(target.get("center_y", 0))))
        bottom_x = max(0, min(width - 1, int(target.get("center_x", 0))))
        bottom_y = max(0, min(height - 1, y2 - 1))
        center_on_bridge = float(bool(bridge_mask[center_y, center_x])) if height and width else 0.0
        bottom_on_bridge = float(bool(bridge_mask[bottom_y, bottom_x])) if height and width else 0.0

        msg.data = [
            1.0,
            bridge_found,
            bbox_bridge_overlap,
            lower_bridge_overlap,
            center_on_bridge,
            bottom_on_bridge,
            float(width),
            float(height),
            left_side_contact,
            right_side_contact,
            side_contact,
            side_contact_ratio,
            side_contact_pixels,
        ]
        self.target_surface_pub.publish(msg)

    def publish_bridge_geometry(self, road_mask, bridge_mask, header=None):
        if header is None:
            self.last_bridge_geometry_reason = "missing image header"
            self._publish_bridge_geometry_debug(None, None, 0, header=None)
            return
        bridge = self._segmentation_stats(bridge_mask)
        entry = self._bridge_entry_stats(road_mask, bridge_mask, bridge, header)
        entry.update(self._compute_bridge_ramp_quality(bridge_mask, road_mask))
        frame_id = (
            self.bridge_geometry_camera_frame
            or header.frame_id
            or self.latest_depth_frame_raw
            or self.latest_depth_frame_compressed
        )
        if not frame_id:
            self.last_bridge_geometry_reason = "missing source frame"
            self.bridge_edge_points_last_published_count = 0
            self.bridge_boundary_points_last_published_count = 0
            self.bridge_entry_point_published_last = False
            self._publish_bridge_geometry_debug(entry, None, 0, header=header)
            return

        entry_confirmed = bool(entry.get("entry_confirmed", 0.0) >= 0.5)
        entry_point = (
            self._project_pixel_to_camera(entry["u"], entry["v"], header)
            if entry_confirmed
            else None
        )
        self.bridge_entry_point_published_last = False
        if entry_point is not None:
            msg = PointStamped()
            msg.header.stamp = header.stamp
            msg.header.frame_id = frame_id
            msg.point.x = entry_point[0]
            msg.point.y = entry_point[1]
            msg.point.z = entry_point[2]
            self.bridge_entry_point_pub.publish(msg)
            self.bridge_entry_points_published += 1
            self.bridge_entry_point_published_last = True

        pre_entry_point = None
        if (
            entry_confirmed
            and float(entry.get("pre_entry_confidence", 0.0)) > 0.0
            and float(entry.get("pre_entry_depth", 0.0)) > 0.0
        ):
            pre_entry_point = self._project_pixel_to_camera(
                entry["pre_entry_u"], entry["pre_entry_v"], header
            )
        if pre_entry_point is not None:
            msg = PointStamped()
            msg.header.stamp = header.stamp
            msg.header.frame_id = frame_id
            msg.point.x = pre_entry_point[0]
            msg.point.y = pre_entry_point[1]
            msg.point.z = pre_entry_point[2]
            self.bridge_pre_entry_point_pub.publish(msg)

        edge_points = self._bridge_edge_camera_points(bridge_mask, header)
        if point_cloud2 is None or not edge_points:
            if point_cloud2 is None:
                self.last_bridge_geometry_reason = "sensor_msgs_py.point_cloud2 unavailable"
            elif not edge_points:
                self.last_bridge_geometry_reason = "no projected bridge boundary samples"
            self.bridge_edge_points_last_published_count = 0
            self.bridge_boundary_points_last_published_count = 0
            self._publish_bridge_geometry_debug(entry, frame_id, 0, header=header)
            return
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="side", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud = point_cloud2.create_cloud(header, fields, edge_points)
        cloud.header.frame_id = frame_id
        self.bridge_edge_points_pub.publish(cloud)
        self.bridge_edge_points_published += 1
        self.bridge_edge_points_last_published_count = len(edge_points)

        boundary_points = self._bridge_boundary_role_points(edge_points, entry_point)
        boundary_fields = fields + [
            PointField(name="role", offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        boundary_cloud = point_cloud2.create_cloud(header, boundary_fields, boundary_points)
        boundary_cloud.header.frame_id = frame_id
        self.bridge_boundary_points_pub.publish(boundary_cloud)
        self.bridge_boundary_points_published += 1
        self.bridge_boundary_points_last_published_count = len(boundary_points)
        self.last_bridge_geometry_reason = "published bridge boundary geometry"
        self._publish_bridge_geometry_debug(entry, frame_id, len(edge_points), header=header)

    def _bridge_boundary_role_points(self, edge_points, entry_point):
        boundary_points = []
        for x, y, z, side in edge_points:
            side_id = int(round(float(side)))
            if side_id == 0:
                role = 0.0
            elif side_id == 1:
                role = 1.0
            else:
                role = 3.0
            boundary_points.append((x, y, z, float(side_id), role))

        if entry_point is not None:
            boundary_points.append(
                (
                    entry_point[0],
                    entry_point[1],
                    entry_point[2],
                    2.0,
                    2.0,
                )
            )
        return boundary_points

    def _depth_age_seconds(self, header=None):
        _, depth_stamp, _ = self._depth_source()
        if header is None or depth_stamp is None:
            return -1.0
        rgb_time = self._stamp_to_seconds(header.stamp)
        depth_time = self._stamp_to_seconds(depth_stamp)
        if rgb_time is None or depth_time is None:
            return -1.0
        return abs(rgb_time - depth_time)

    def _publish_bridge_geometry_debug(self, entry, frame_id, edge_count, header=None):
        msg = String()
        entry_u = 0.0 if entry is None else float(entry.get("u", 0.0))
        entry_v = 0.0 if entry is None else float(entry.get("v", 0.0))
        entry_depth = 0.0 if entry is None else float(entry.get("depth", 0.0))
        ramp_valid = False if entry is None else bool(entry.get("ramp_valid", False))
        ramp_confidence = 0.0 if entry is None else float(entry.get("ramp_confidence", 0.0))
        side_view_score = 0.0 if entry is None else float(entry.get("side_view_score", 0.0))
        ramp_continuous = False if entry is None else bool(entry.get("continuous_lower_to_upper", False))
        ramp_reason = "" if entry is None else str(entry.get("reason", ""))
        msg.data = (
            "node_alive=True "
            f"segmentation_enabled={self.enable_segmentation} "
            f"camera_info_ok={self.camera_info_received or self._camera_intrinsics_valid()} "
            f"depth_available={self._depth_source()[0] is not None} "
            f"depth_age={self._depth_age_seconds(header):.3f} "
            f"frame_id={frame_id or ''} "
            f"bridge_mask_found={self.bridge_mask_found_last} "
            f"bridge_entry_confirmed={self.bridge_entry_confirmed_last} "
            f"entry_u={entry_u:.1f} entry_v={entry_v:.1f} "
            f"entry_depth={entry_depth:.2f} "
            f"ramp_valid={ramp_valid} "
            f"ramp_confidence={ramp_confidence:.2f} "
            f"side_view_score={side_view_score:.2f} "
            f"ramp_continuous={ramp_continuous} "
            f"ramp_reason={ramp_reason} "
            f"edge_points={edge_count} "
            f"edge_points_last_published_count={self.bridge_edge_points_last_published_count} "
            f"boundary_points_last_published_count={self.bridge_boundary_points_last_published_count} "
            f"entry_point_published={self.bridge_entry_point_published_last} "
            f"valid_depth_samples={self.valid_depth_samples} "
            f"rejected_depth_samples={self.rejected_depth_samples} "
            f"entry_points_published={self.bridge_entry_points_published} "
            f"edge_clouds_published={self.bridge_edge_points_published} "
            f"reason_if_not_publishing={self.last_bridge_geometry_reason}"
        )
        self.bridge_geometry_debug_pub.publish(msg)

    def _publish_bridge_geometry_heartbeat(self):
        self._publish_bridge_geometry_debug(None, self.bridge_geometry_camera_frame, 0)

    def _bridge_edge_camera_points(self, bridge_mask, header=None):
        height, width = bridge_mask.shape
        if not self._camera_intrinsics_valid() or np.count_nonzero(bridge_mask) <= 0:
            return []

        y_values = np.linspace(
            int(height * 0.55),
            int(height * 0.95),
            max(2, int(self.bridge_geometry_edge_sample_count)),
        )
        points = []
        min_pixels_per_row = max(4, int(width * 0.01))
        inset = max(2, int(self.bridge_side_inward_sample_pixels))
        valid_pairs = 0
        for y in y_values:
            row_y = max(0, min(height - 1, int(round(y))))
            xs = np.flatnonzero(bridge_mask[row_y, :])
            if len(xs) < min_pixels_per_row:
                continue
            left_u = min(width - 1, int(xs[0]) + inset)
            right_u = max(0, int(xs[-1]) - inset)
            if right_u <= left_u:
                continue
            center_u = int(round((float(xs[0]) + float(xs[-1])) * 0.5))
            radius = max(0, int(self.bridge_side_depth_patch_radius))
            left_depth = self._depth_patch_meters(left_u, row_y, header, radius)
            right_depth = self._depth_patch_meters(right_u, row_y, header, radius)
            center_depth = self._depth_patch_meters(center_u, row_y, header, radius)
            if min(left_depth, right_depth, center_depth) <= 0.0:
                continue
            if (
                abs(left_depth - center_depth) > self.bridge_side_depth_max_row_difference
                or abs(right_depth - center_depth) > self.bridge_side_depth_max_row_difference
            ):
                continue

            left = self._project_pixel_to_camera_with_depth(left_u, row_y, left_depth)
            right = self._project_pixel_to_camera_with_depth(right_u, row_y, right_depth)
            center = self._project_pixel_to_camera_with_depth(center_u, row_y, center_depth)
            if left is None or right is None or center is None:
                continue
            pair_width = float(np.linalg.norm(np.array(left[:2]) - np.array(right[:2])))
            if not (
                self.bridge_side_min_pair_width
                <= pair_width
                <= self.bridge_side_max_pair_width
            ):
                continue
            if pair_width < 0.05:
                continue
            points.extend(
                [
                    (left[0], left[1], left[2], 0.0),
                    (right[0], right[1], right[2], 1.0),
                    (center[0], center[1], center[2], 2.0),
                ]
            )
            valid_pairs += 1
        if valid_pairs < self.bridge_side_projection_min_valid_pairs:
            self.last_bridge_geometry_reason = (
                f"low bridge side projection confidence: valid_pairs={valid_pairs}"
            )
            return []
        return points

    def publish_x_multi_depths(self, image):
        """
        取得畫面 n 個等分點的深度並發布
        """
        height, width = image.shape[:2]
        cy_center = height // 2  # 固定 Y 座標在畫面中心
        segment_length = width // self.x_num_splits

        # 計算 10 個等分點的 X 座標
        points = [(i * segment_length, cy_center) for i in range(self.x_num_splits)]

        # 取得每個等分點的深度值
        depth_values = [self.get_depth_at(x, cy_center) for x, _ in points]

        # 以 Float32MultiArray 發布
        depth_msg = Float32MultiArray()
        depth_msg.data = depth_values
        self.x_multi_depth_pub.publish(depth_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

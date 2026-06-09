import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory
import torch

using_yolo_det_model = True
using_yolo_seg_model = True

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__("yolo_detection_node")

        self.declare_parameter("target_labels", ["bear"])
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

        # 初始化 cv_bridge
        self.bridge = CvBridge()

        self.latest_depth_image_raw = None
        self.latest_depth_image_compressed = None

        self.allowed_labels = {
            label.strip()
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
        self.locked_target = None
        self.lock_misses = 0

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

        # 發布 目標檢測數據 (是否找到目標 + 距離)
        self.target_pub = self.create_publisher(
            Float32MultiArray, "/yolo/target_info", 10
        )
        self.target_bbox_pub = self.create_publisher(
            Float32MultiArray, "/yolo/target_bbox", 10
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

    def _string_set_param(self, name):
        return {
            label.strip().lower()
            for label in self.get_parameter(name)
            .get_parameter_value()
            .string_array_value
            if label.strip()
        }

    def depth_callback_raw(self, msg):
        """接收 **無壓縮** 深度圖"""
        try:
            self.latest_depth_image_raw = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
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
        except Exception as e:
            self.get_logger().error(f"Could not convert compressed depth image: {e}")

    def image_callback(self, msg):
        """接收影像並進行物體檢測"""
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
            seg_image, has_mask = self.draw_masks(cv_image, seg_results)
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
                class_name = self.det_model.names[class_id]

                # 只保留設定內的標籤；target_labels 空陣列時代表不過濾
                if self.allowed_labels and class_name not in self.allowed_labels:
                    continue

                # 計算 Bounding Box 正中心點
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 優先使用無壓縮的深度圖
                depth_value = self.get_depth_at(cx, cy)
                depth_text = f"{depth_value:.2f}m" if depth_value > 0.0 else "N/A"
                delta_x = cx - image_center_x
                target_candidate = {
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

        best_target = self._select_target(target_candidates)
        if best_target is not None:
            found_target = 1
            target_distance = best_target["depth"]
            delta_x = best_target["delta_x"]

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

    def draw_masks(self, image, results):
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

        self.publish_segmentation_info(road_mask, bridge_mask)
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

    def publish_segmentation_info(self, road_mask, bridge_mask):
        """
        data layout:
        [road_found, road_delta_x, road_area_ratio, road_bottom_coverage,
         bridge_found, bridge_delta_x, bridge_area_ratio, bridge_bottom_coverage,
         image_width, image_height,
         road_center_x, road_center_y_ratio, road_top_y_ratio, road_bottom_y_ratio,
         bridge_center_x, bridge_center_y_ratio, bridge_top_y_ratio, bridge_bottom_y_ratio,
         connected, connection_score, connection_delta_x, connection_gap_y_ratio,
         road_bottom_center_x, road_top_center_x, bridge_bottom_center_x, bridge_mid_center_x,
         central_connection_score, central_connection_delta_x]
        """
        msg = Float32MultiArray()
        height, width = road_mask.shape
        road = self._segmentation_stats(road_mask)
        bridge = self._segmentation_stats(bridge_mask)
        connection = self._segmentation_connection_stats(road_mask, bridge_mask)
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
            }

        ys, xs = np.nonzero(mask)
        bottom_start = int(height * 0.65)
        bottom_roi = mask[bottom_start:, :]
        bottom_area = max(1, bottom_roi.size)
        top_end = int(height * 0.45)
        mid_start = int(height * 0.35)
        mid_end = int(height * 0.75)
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

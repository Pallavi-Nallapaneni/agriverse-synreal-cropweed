#!/usr/bin/env python3

import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)
from cv_bridge import CvBridge
from ultralytics import YOLO


class CropWeedDetectorNode(Node):
    def __init__(self):
        super().__init__("crop_weed_detector_node")

        self.declare_parameter(
            "weights_path",
            "best.pt",
        )

        self.declare_parameter("image_topic", "image_raw")
        self.declare_parameter("conf_threshold", 0.35)
        self.declare_parameter("publish_annotated", True)

        weights_path = self.get_parameter(
            "weights_path"
        ).value
        image_topic = self.get_parameter(
            "image_topic"
        ).value
        self.conf_threshold = self.get_parameter(
            "conf_threshold"
        ).value
        self.publish_annotated = self.get_parameter(
            "publish_annotated"
        ).value

        self.get_logger().info(
            f"Loading model weights from: {weights_path}"
        )

        self.model = YOLO(weights_path)
        self.class_names = self.model.names

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10,
        )

        self.detections_pub = self.create_publisher(
            Detection2DArray,
            "detections",
            10,
        )

        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(
                Image,
                "annotated_image",
                10,
            )

        self._frame_count = 0
        self._t_last_log = time.time()

        self.get_logger().info(
            f'Crop/weed detector ready. '
            f'Subscribed to "{image_topic}", '
            f"conf_threshold={self.conf_threshold}"
        )

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8",
        )

        t0 = time.time()

        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            verbose=False,
        )[0]

        inference_ms = (time.time() - t0) * 1000.0

        detection_array = Detection2DArray()
        detection_array.header = msg.header

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            det = Detection2D()
            det.header = msg.header

            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1

            hypothesis = ObjectHypothesisWithPose()

            hypothesis.hypothesis.class_id = self.class_names.get(
                cls_id,
                str(cls_id),
            )
            hypothesis.hypothesis.score = conf

            det.results.append(hypothesis)
            detection_array.detections.append(det)

        self.detections_pub.publish(detection_array)

        if self.publish_annotated:
            annotated = results.plot()

            annotated_msg = self.bridge.cv2_to_imgmsg(
                annotated,
                encoding="bgr8",
            )

            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

        self._frame_count += 1

        now = time.time()

        if now - self._t_last_log > 5.0:
            elapsed = now - self._t_last_log
            fps = self._frame_count / elapsed

            self.get_logger().info(
                f"~{fps:.1f} FPS | "
                f"last inference {inference_ms:.1f} ms | "
                f"{len(detection_array.detections)} detections"
            )

            self._frame_count = 0
            self._t_last_log = now


def main(args=None):
    rclpy.init(args=args)

    node = CropWeedDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


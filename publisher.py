# PUBLISHER NODE

import paho.mqtt.client as mqtt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from datetime import datetime
import numpy as np
import cv2

BROKER_ADDRESS = "localhost"
MQTT_TOPIC = "camera/images"
ROS_TOPIC = "image_topic"

class MqttToRosPublisher(Node):
    def __init__(self):
        super().__init__('mqtt_to_ros_publisher')
        self.publisher_ = self.create_publisher(Image, ROS_TOPIC, 10)
        self.bridge = CvBridge()

        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect(BROKER_ADDRESS, 1883, 60)
        self.mqtt_client.subscribe(MQTT_TOPIC)
        self.mqtt_client.loop_start()
        self.get_logger().info(f"Subscribed to MQTT topic: {MQTT_TOPIC}")


    def on_message(self, client, userdata, msg):
        self.get_logger().info(f"Image received from MQTT topic: {MQTT_TOPIC}")

        image_data = msg.payload
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            np_arr = np.frombuffer(image_data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            ros_image.header.frame_id = f"mqtt_image_{timestamp}"

            self.publisher_.publish(ros_image)
            self.get_logger().info(f"Published image to ROS topic: {ROS_TOPIC}")

        except Exception as e:
            self.get_logger().error(f"Failed to process or publish image: {e}")

def main(args=None):
    rclpy.init(args=args)
    mqtt_to_ros_publisher = MqttToRosPublisher()
    rclpy.spin(mqtt_to_ros_publisher)
    mqtt_to_ros_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

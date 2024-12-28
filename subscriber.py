# SUBSCRIBER NODE

import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import shutil
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from datetime import datetime

class ImageSubscriber(Node):
    @tf.keras.utils.register_keras_serializable()
    def dice_loss(self, y_true, y_pred):
        smooth = 1.
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    @tf.keras.utils.register_keras_serializable()
    def dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    @tf.keras.utils.register_keras_serializable()
    def iou(self, y_true, y_pred):
        smooth = 1.
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)
    
    def __init__(self):
        super().__init__('subscriber')
        self.subscription = self.create_subscription(
            Image,
            'image_topic',
            self.listener_callback,
            10)
        self.subscription  
        self.bridge = CvBridge()
        model_path = '/home/tamuq/Downloads/SegmentationLabwithcropandbad.keras' #change this to the path to your model file
        custom_objects = {
            'dice_loss': self.dice_loss,
            'dice_coefficient': self.dice_coefficient,
            'iou': self.iou
        }
        self.model = load_model(model_path, custom_objects=custom_objects)
        
        current_date = datetime.now()
        month_name = current_date.strftime("%b")
        day = current_date.strftime("%d")
        timestamp = current_date.strftime("%Y%m%d_%H%M%S")
        
        self.output_dir = os.path.join(
            '/home/tamuq/Downloads/New Pictures',
            f"{month_name} {day}",
            f"Data_{timestamp}"
        )


        self.layer_height_publisher = self.create_publisher(Float32MultiArray, 'layer_height', 10)

    def preprocess_image(self, cv_image):
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray_image)
        blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)

        normalized_image = cv_image / 255.0
        normalized_image = np.expand_dims(normalized_image, axis=0)

        return normalized_image

    def save_images(self, base_filename, images, measurement_points):
        subfolder = os.path.join(self.output_dir, f"{base_filename}_results")
        
        if os.path.exists(subfolder):
            shutil.rmtree(subfolder)
        
        os.makedirs(subfolder, exist_ok=True)

        for idx, image in enumerate(images):
            save_path = os.path.join(subfolder, f"{base_filename}_output_{idx + 1}.png")
            cv2.imwrite(save_path, image)
            self.get_logger().info(f"Saved {save_path}")

        csv_path = os.path.join(subfolder, f"{base_filename}_measurements.csv")
        with open(csv_path, 'w') as csv_file:
            csv_file.write("Layer Index,Layer Height,X Coordinate,Top Y Coordinate,Bottom Y Coordinate,Midpoint Y Coordinate,Image Name\n")
            for layer_index, height, x, top_y, bottom_y, midpoint_y in measurement_points:
                csv_file.write(f"{layer_index},{height},{x},{top_y},{bottom_y},{midpoint_y},{base_filename}.png\n")
        
        self.get_logger().info(f"Saved CSV file {csv_path}")

    def listener_callback(self, msg):
        self.get_logger().info('Receiving image')

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        preprocessed_image = self.preprocess_image(cv_image)

        start_time = time.perf_counter()
        predicted_mask = self.model.predict(preprocessed_image)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        self.get_logger().info(f"Model prediction took {elapsed_time:.8f} seconds.")

        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

        base_filename = msg.header.frame_id.split('/')[-1].split('.')[0]

        measurement_points = self.calculate_layer_heights(predicted_mask)

        self.save_images(base_filename, [
            cv_image,  
            predicted_mask.squeeze(),  
            self.get_colored_overlay(cv_image, predicted_mask),  
            self.get_mask_with_measurements(predicted_mask, measurement_points)  
        ], measurement_points)


        if measurement_points:
            highest_layer_index = max(mp[0] for mp in measurement_points)
            top_layer_measurements = [
                mp[1] for mp in sorted(measurement_points, key=lambda mp: mp[2], reverse=True)
                if mp[0] == highest_layer_index
            ]


            layer_height_msg = Float32MultiArray()


            dim = MultiArrayDimension()
            dim.label = base_filename  
            dim.size = len(top_layer_measurements)
            dim.stride = len(top_layer_measurements)
            layer_height_msg.layout.dim.append(dim)

            layer_height_msg.data = top_layer_measurements  


            self.layer_height_publisher.publish(layer_height_msg)

            self.get_logger().info(f"Published {len(top_layer_measurements)} measurements for the top layer of {base_filename}")

    def calculate_layer_heights(self, predicted_mask, region_width=40, scaling_factor=2, y_threshold=20):
        measurement_points = []
        scaled_mask = cv2.resize(predicted_mask.squeeze(), (predicted_mask.shape[2] * scaling_factor, predicted_mask.shape[1] * scaling_factor), interpolation=cv2.INTER_LINEAR)
        scaled_mask = cv2.GaussianBlur(scaled_mask, (7, 7), 0)
        contours, _ = cv2.findContours(scaled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1], reverse=True)

        num_layers = len(contours) - 1
        self.get_logger().info(f"Identified {num_layers} layers.")

        current_layer_index = 1
        previous_y = None

        for i in range(num_layers):
            bottom_contour = contours[i]  
            top_contour = contours[i + 1]  

            bottom_points = [pt[0] for pt in bottom_contour]
            top_points = [pt[0] for pt in top_contour]

            for x in range(scaled_mask.shape[1] - region_width * scaling_factor, -1, -region_width * scaling_factor):
                bottom_y = max([pt[1] for pt in bottom_points if x - 15 <= pt[0] < x + region_width * scaling_factor + 15], default=None)
                top_y = min([pt[1] for pt in top_points if x - 15 <= pt[0] < x + region_width * scaling_factor + 15], default=None)

                if top_y is not None and bottom_y is not None:
                    height = abs(bottom_y - top_y) / scaling_factor
                    midpoint_y = (top_y + bottom_y) // 2

                    if previous_y is not None and abs(midpoint_y - previous_y) > y_threshold:
                        current_layer_index += 1 

                    previous_y = midpoint_y 

                    measurement_points.append((current_layer_index, height, x // scaling_factor + region_width // 2, top_y // scaling_factor, bottom_y // scaling_factor, midpoint_y // scaling_factor))
                else:
                    if measurement_points:
                        prev_layer_index, prev_height, prev_x, prev_top_y, prev_bottom_y, prev_midpoint_y = measurement_points[-1]
                        measurement_points.append((prev_layer_index, prev_height, prev_x, prev_top_y, prev_bottom_y, prev_midpoint_y))

        return measurement_points

    def get_colored_overlay(self, original_image, predicted_mask):
        colored_mask = cv2.cvtColor(predicted_mask.squeeze() * 255, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(original_image, 0.5, colored_mask, 1, 0)
        return overlay

    def get_mask_with_measurements(self, predicted_mask, measurement_points):
        mask_with_points = cv2.cvtColor(predicted_mask.squeeze() * 255, cv2.COLOR_GRAY2RGB)

        for (layer_index, height, x, top_y, bottom_y, midpoint_y) in measurement_points:
            cv2.circle(mask_with_points, (x, top_y), 2, (255, 0, 0), -1)
            cv2.circle(mask_with_points, (x, bottom_y), 2, (255, 0, 0), -1)
            cv2.line(mask_with_points, (x, top_y), (x, bottom_y), (255, 255, 255), 1)
            cv2.putText(mask_with_points, f"{int(height)}", (x + 5, midpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return mask_with_points

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

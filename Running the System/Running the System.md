# **Running the System**

This page provides a step-by-step guide to activating the entire data collection and processing system after both the Raspberry Pi and laptop are powered on. Follow these instructions each time you want to run the system.

---

## **1. Start the MQTT Broker on the Laptop**

1. Open a terminal on the laptop.
2. Start the Mosquitto MQTT broker:
   ```bash
   sudo systemctl start mosquitto
   ```
3. Verify that the broker is running:
   ```bash
   sudo systemctl status mosquitto
   ```
   - You should see `active (running)` in the output.

---

## **2. Run the Image Capture Script on the Raspberry Pi**

1. Open a terminal on the Raspberry Pi.
2. Navigate to the directory containing the `image_sender.py` script:
   ```bash
   cd ~
   ```
3. Source the environment:
   ```bash
   source mqtt_env/bin/activate
   ```
4. Run the script to start capturing and sending images:
   ```bash
   python3 image_sender.py
   ```
5. Confirm that the script is running by checking for logs like:
   ```
   Image sent at 2023-12-28 15:30:45
   ```

---

## **3. Run the ROS 2 Publisher Node on the Laptop**

1. Open another terminal on the laptop.
2. Source the ROS 2 environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```
3. Navigate to your ROS workspace:
   ```bash
   cd ~/ros2_ws
   source install/setup.bash
   ```
4. Start the MQTT to ROS publisher node:
   ```bash
   ros2 run system publisher
   ```
5. Confirm the node is running by checking for logs like:
   ```
   [INFO] [mqtt_to_ros_publisher]: Subscribed to MQTT topic: camera/images
   ```

---

## **4. Run the ROS 2 Subscriber Node on the Laptop**

1. Open another terminal on the laptop.
2. Source the ROS 2 environment (if not already done):
   ```bash
   source /opt/ros/humble/setup.bash
   cd ~/ros2_ws
   source install/setup.bash
   ```
3. Start the ROS subscriber node to process the images:
   ```bash
   ros2 run system subscriber
   ```
4. Confirm the node is running by checking for logs like:
   ```
   [INFO] [image_subscriber]: Receiving image
   [INFO] [image_subscriber]: Model prediction took X.XX seconds.
   ```

---

## **5. Verifying the Outputs**

1. **Check Processed Images and Overlays**:
   - Navigate to the output directory (automatically created based on the date and timestamp) to verify saved images, overlays, and measurements:
     ```bash
     cd /home/tamuq/Downloads/New\ Pictures/<date_folder>
     ```

2. **Real-Time Monitoring**:
   - Use the following commands to monitor real-time outputs:
     - Published images:
       ```bash
       ros2 topic echo image_topic
       ```
     - Layer height measurements:
       ```bash
       ros2 topic echo layer_height
       ```

---

## **6. Stopping the System**

To gracefully stop the system:
1. **Stop the Image Capture Script**:
   - On the Raspberry Pi, press `Ctrl + C` in the terminal running the `image_sender.py` script.

2. **Stop ROS 2 Nodes**:
   - On the laptop, press `Ctrl + C` in the terminals running the ROS publisher and subscriber nodes.

3. **Stop the MQTT Broker** (optional):
   - If you no longer need the broker running, stop it with:
     ```bash
     sudo systemctl stop mosquitto
     ```



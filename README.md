# Data Collection and Processing System for 3D Printing Analysis

This repository documents the setup and implementation of a comprehensive system for capturing, transmitting, and analyzing 3D printed structures. The system enables real-time data collection, layer height measurement, and anomaly detection for 3D printing processes, using a Raspberry Pi HQ Camera, an MQTT-based transmission system, and a ROS 2-powered image processing pipeline.

---

## Hardware Used

The following table lists all the hardware components used in this project:

| **Hardware**            | **Details**                                                                                                                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Laptop**              | Lenovo Laptop running Ubuntu 22.04                                                                                                                                                                         |
| **Raspberry Pi**        | Raspberry Pi 5 with 8GB RAM                                                                                                                                                                                |
| **Camera**              | Raspberry Pi High Quality Camera (CS Mount)                                                                                                                                                                |
| **MicroSD Card**        | MakerDisk 32GB Class 10                                                                                                                                                                                     |
| **Power Supply**        | Official Raspberry Pi 27W USB-C Power Supply                                                                                                                                                               |
| **Camera Lens**         | CS Mount Lens compatible with the HQ Camera                                                                                                                                                                |
| **Networking Setup**    | Shared Wi-Fi network between the Raspberry Pi and the laptop                                                                                                                                               |
| **Software Dependencies** | ROS 2 Humble, Python 3.10, TensorFlow, OpenCV, paho-mqtt                                                                                                                                                    |

---

## System Overview

This project implements a data collection and analysis system for 3D printing quality control. It integrates hardware and software components into a seamless pipeline that:
1. Captures high-resolution images of the 3D printed structure.
2. Transmits images from the Raspberry Pi to the laptop over an MQTT network.
3. Processes images using a segmentation model to measure layer heights and identify anomalies.
4. Displays processed results and stores them for further analysis.

---

## How It Works

The system is divided into four main stages:

### **1. Image Capture**
- **Hardware**: The Raspberry Pi HQ Camera captures high-resolution images of the 3D printed structure.
- **Software**: A Python script on the Raspberry Pi (`mqtt_image_sender.py`) starts the camera stream and captures snapshots every 2 seconds.
- **Output**: Each captured image is resized to fit the input requirements of the ML model and sent via MQTT.

### **2. Image Transmission**
- **Protocol**: MQTT is used for lightweight and reliable message transmission.
- **Implementation**:
  - The Raspberry Pi publishes images to the MQTT broker running on the laptop.
  - The broker relays these images to a custom ROS 2 publisher node (`mqtt_image_publisher.py`).

### **3. Image Processing**
- **Subscriber Node**: A ROS 2 subscriber node (`image_subscriber.py`) receives images from the ROS topic.
- **Segmentation**:
  - A TensorFlow model segments the images, identifying interlayer boundaries.
  - The system calculates layer height measurements and overlays annotations on the images.
- **Output**:
  - Processed images, overlays, and CSV files with measurement data are saved in dynamically named subdirectories based on the date and time.

### **4. Visualization and Analysis**
- Processed results are:
  - Saved in structured directories for offline analysis.
  - Published to a ROS topic for real-time monitoring.

---

## Getting Started

### **1. Setting Up the Laptop**
1. Install **Ubuntu 22.04** and ROS 2 Humble.
   - Follow [this guide](https://docs.ros.org/en/humble/Installation.html) for ROS 2 installation.
2. Install additional dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip
   pip install tensorflow opencv-python-headless paho-mqtt
   ```

### **2. Setting Up the Raspberry Pi**
1. Install Raspberry Pi OS and enable the camera:
   ```bash
   sudo raspi-config
   ```
2. Install required Python libraries:
   ```bash
   pip install opencv-python-headless paho-mqtt
   ```

### **3. Running the System**
1. Start the MQTT broker on the laptop:
   ```bash
   sudo systemctl start mosquitto
   ```
2. Run the image capture script on the Raspberry Pi:
   ```bash
   python3 mqtt_image_sender.py
   ```
3. Start the ROS 2 publisher node on the laptop:
   ```bash
   ros2 run image_processor_test mqtt_image_publisher
   ```
4. Run the subscriber node to process images:
   ```bash
   ros2 run image_processor_test image_subscriber
   ```

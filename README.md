# Data Collection and Processing System for 3D Printing Analysis

This repository documents the setup and implementation of a comprehensive system for capturing, transmitting, and analyzing 3D printed concrete structures. The system enables real-time data collection, layer height measurement, and anomaly detection for 3DCP processes, using a Raspberry Pi HQ Camera, an MQTT-based transmission system, and a ROS 2-powered image processing pipeline.

---

## Hardware and Software Used

The following table lists all the hardware components used in this project:

| **Category**            | **Details**                                                                                                                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Laptop**              | Lenovo Laptop running Ubuntu 22.04                                                                                                                                                                         |
| **Raspberry Pi**        | Raspberry Pi 5 with 8GB RAM                                                                                                                                                                                |
| **Camera**              | Raspberry Pi High Quality Camera (CS Mount)                                                                                                                                                                |
| **MicroSD Card**        | MakerDisk 64GB Class 10                                                                                                                                                                                     |
| **Power Supply**        | Official Raspberry Pi 27W USB-C Power Supply                                                                                                                                                               |
| **Camera Lens**         | CS Mount Lens compatible with the HQ Camera                                                                                                                                                                |
| **Networking Setup**    | Shared Wi-Fi network between the Raspberry Pi and the laptop                                                                                                                                               |
| **ROS 2 Version**       | ROS 2 Humble                                                                                                                                                                                               |
| **Python Version**      | Python 3.10 (ROS 2 Environment), Python 3.x (Raspberry Pi Environment)                                                                                                                                     |
| **Libraries**           | TensorFlow, OpenCV (Headless), paho-mqtt, rclpy, cv_bridge                                                                                                                                                 |
| **MQTT Broker**         | Mosquitto                                                                                                                                                                                                  |

---

## System Overview

This project implements a data collection and analysis system for 3D printing quality control. It integrates hardware and software components into a pipeline that:
1. Captures images of the 3D printed structure.
2. Transmits images from the Raspberry Pi to the laptop over an MQTT network.
3. Processes images using a segmentation model to measure layer heights and identify anomalies.
4. Displays processed results and stores them for further analysis.

---

## How It Works

The system is divided into four main stages:

### **1. Image Capture**
- **Hardware**: The Raspberry Pi HQ Camera captures high-resolution images of the 3D printed structure.
- **Software**: A Python script on the Raspberry Pi (`mqtt_image_sender.py`) starts the camera stream and captures snapshots at specified time intervals.
- **Output**: Each captured image is resized to fit the input requirements of the ML model and sent via MQTT.

### **2. Image Transmission**
- **Protocol**: MQTT is used for lightweight and reliable message transmission.
- **Implementation**:
  - The Raspberry Pi publishes images to the MQTT broker running on the laptop.
  - The broker relays these images to a custom ROS 2 publisher node (`publisher.py`).

### **3. Image Processing**
- **Subscriber Node**: A ROS 2 subscriber node (`subscriber.py`) receives images from the ROS topic.
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

The following pages walk you through a systematic setup of this system starting from the software setup all the way to writing and running the scripts used to perform the individual actions: 

1. [Setting Up the Laptop]()
2. [Setting Up the Raspberry Pi]()
3. [Running the System]()

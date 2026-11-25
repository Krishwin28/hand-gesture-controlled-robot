# Hand Gesture Controlled Robot Using Computer Vision

A real-time hand gesture-controlled robotic vehicle developed using Python, OpenCV, and an ESP32 microcontroller. The system uses webcam-based image processing to recognize predefined hand gestures and translate them into movement commands for a two-wheel robotic vehicle.

## Project Overview

This project explores the integration of computer vision and embedded systems to develop a human-machine interaction system. Instead of using conventional remote controls, the robot is operated through hand gestures detected by a webcam.

The system processes live video frames, identifies the hand region, analyzes finger positions and orientation, and classifies the detected gesture into a corresponding movement command.

The project primarily uses classical image processing techniques rather than machine learning-based gesture recognition.

## Key Features

- Real-time hand detection using a webcam.
- Background subtraction-based foreground segmentation.
- Hand contour extraction and analysis.
- Finger detection and gesture classification.
- Direction recognition based on finger orientation.
- Wireless communication between the computer and ESP32 (when enabled).
- Motor control for a two-wheel robotic vehicle.

## Gesture Control System

The robot responds to the following predefined gestures:

| Hand Gesture | Robot Action |
|---|---|
| One finger pointing upward | Move Forward |
| One finger tilted towards the left | Turn Left (90°) |
| One finger tilted towards the right | Turn Right (90°) |
| Open palm | Move Backward |

The gesture classification system uses finger count and angular orientation to distinguish between movement commands.

## System Architecture

The system consists of two main components:

**1. Computer Vision and Gesture Recognition**

The webcam captures live video frames, which are processed using Python and OpenCV. The hand is segmented from the background, and its contour and geometric features are analyzed to determine the intended gesture.

**2. Embedded Control and Motor Actuation**

The classified gesture is translated into a movement command and transmitted to the ESP32. The microcontroller processes the received command and controls the motor driver to operate the robotic vehicle.

### Processing Pipeline

```text
Webcam Input
     |
     v
Frame Acquisition
     |
     v
Image Preprocessing
     |
     v
Background Subtraction
     |
     v
Hand Segmentation
     |
     v
Contour Extraction
     |
     v
Finger Detection
     |
     v
Gesture Classification
     |
     v
Movement Command
     |
     v
ESP32 Microcontroller
     |
     v
Motor Driver
     |
     v
Robot Movement
```

## Technologies and Tools

| Category | Technologies |
|---|---|
| Programming Language | Python, C++ |
| Computer Vision | OpenCV |
| Numerical Processing | NumPy |
| Microcontroller | ESP32 |
| Embedded Development | Arduino IDE |
| Hardware | DC Motors, Motor Driver, Webcam |
| Communication | Serial / Wireless Communication (depending on configuration) |

## Hardware Components

- ESP32 development board
- Two-wheel robotic car chassis
- Two DC geared motors
- Dual-channel motor driver
- Rechargeable battery pack
- Buck converter
- Webcam or laptop camera
- Connecting wires and supporting components

## Software Requirements

- Python 3.x
- OpenCV
- NumPy
- Arduino IDE
- ESP32 board support package

### Python Dependencies

Install the required Python libraries:

```bash
pip install opencv-python numpy
```

## Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/hand-gesture-controlled-robot.git
```

### 2. Navigate to the Project Directory

```bash
cd hand-gesture-controlled-robot
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the ESP32

- Open the embedded controller sketch in Arduino IDE.
- Select the appropriate ESP32 board.
- Configure the required communication settings.
- Upload the firmware to the ESP32.

### 5. Run the Application

Connect the webcam and execute the main Python application:

```bash
python gesture_recognition.py
```

## Challenges and Learning Outcomes

Throughout the development process, several practical challenges were addressed:

- Improving hand segmentation under varying lighting conditions.
- Reducing background noise and unwanted contour detection.
- Distinguishing actual fingertips from contour irregularities.
- Improving gesture classification using geometric features.
- Integrating computer vision with embedded motor control.
- Troubleshooting hardware connections, power distribution, and motor driver operation.

This project provided practical experience in image processing, computer vision, embedded programming, hardware interfacing, and system integration.

## Future Improvements

- Improve hand segmentation and recognition robustness.
- Introduce more complex gesture-based movement commands.
- Implement improved noise filtering and gesture stabilization.
- Optimize communication between the computer and robot.
- Explore onboard vision processing for a standalone robotic system.

## Author

**Krishwinthan Gobinath**

Electrical and Electronic Engineering Undergraduate

University of Sri Jayewardenepura

[LinkedIn](linkedin.com/in/kriishwinthan-gobinath-673847266)

---

*Developed as a practical exploration of computer vision, human-machine interaction, and embedded robotics.*

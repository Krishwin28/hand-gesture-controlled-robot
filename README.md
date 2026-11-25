# Hand Gesture Controlled Robot

## Overview
A hand gesture-controlled robotic vehicle using computer vision and embedded systems. The system detects hand gestures through a webcam and translates them into movement commands.

## Features
- Real-time hand detection using OpenCV
- Finger-count-based gesture recognition
- Direction classification
- ESP32-based motor control

## Technologies Used
- Python
- OpenCV
- NumPy
- ESP32
- Arduino IDE

## Gesture Controls
| Gesture | Robot Action |
|---|---|
| One finger | Forward |
| Open palm | Backward |
| Finger tilted left | Turn left |
| Finger tilted right | Turn right |

## Project Structure
```text
hand-gesture-controlled-robot/
├── background_detection.py
├── gesture_recognition.py
├── robot_control.ino
├── requirements.txt
├── README.md
└── demo/
    └── demonstration.mp4
```

## Installation
```bash
pip install opencv-python numpy
```

## How to Run
1. Connect a webcam.
2. Upload the Arduino sketch to the ESP32.
3. Run the Python application.
4. Perform the supported hand gestures.

## Future Improvements
- Improve gesture detection under different lighting conditions.
- Wireless communication between the computer and robot.
- Improve motor control and movement accuracy.


"""
gesture_recognition.py
Hand gesture recognition + ESP32 robot control
Classical computer vision only (OpenCV, no ML).

Pipeline:
    webcam -> mirror -> ROI -> blur -> MOG2 background subtraction
           -> morphology -> largest contour -> filled hand mask
           -> palm centre + radius -> convexity defects
           -> fingertip detection -> gesture classification
           -> serial command -> ESP32 -> L298N -> motors

Gestures:
    One finger UP       -> FORWARD
    One finger DOWN     -> BACKWARD
    One finger LEFT     -> TURN LEFT
    One finger RIGHT    -> TURN RIGHT
    Open palm (5 tips)  -> BACKWARD
    Fist (0 tips)       -> STOP
    Unknown / no hand   -> STOP

Keys:
    q = quit
    r = relearn background
"""

import cv2
import numpy as np
import serial
import time


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

CAMERA_INDEX = 0

# ESP32 serial configuration
SERIAL_PORT = "COM16"
BAUD_RATE = 115200
COMMAND_INTERVAL = 0.2       # Send heartbeat every 200 ms

# Region of interest
X1, Y1, X2, Y2 = 160, 160, 480, 440

# Background subtraction (MOG2)
HISTORY = 300
VAR_THRESHOLD = 50
LEARNING_FRAMES = 90
LEARNING_RATE = 0.03

# Image filtering
BLUR_KSIZE = 5
MEDIAN_KSIZE = 5

# Hand validity checks
MIN_HAND_AREA = 4000
MAX_FG_FRACTION = 0.65

# Fingertip detection
K_FACTOR = 0.6
TIP_MAX_ANGLE = 90
TIP_MIN_DIST = 1.6
TIP_MAX_BELOW = 0.2
TIP_MERGE_DIST = 0.4
EDGE_MARGIN = 6

# Direction classification
DIRECTION_TOLERANCE = 45

# Finger gaps
GAP_MIN_DEPTH = 0.4
GAP_MAX_ANGLE = 85
GAP_MAX_BELOW = 0.3

# Colours (BGR)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
GREY = (170, 170, 170)


# ----------------------------------------------------------------------
# ESP32 SERIAL COMMUNICATION
# ----------------------------------------------------------------------

esp32 = None
last_command = None
last_command_time = 0


def connect_esp32():
    """Connect to the ESP32 and allow it to reset after serial opens."""
    global esp32

    try:
        print(f"Connecting to ESP32 on {SERIAL_PORT}...")

        esp32 = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=1,
            write_timeout=1
        )

        # ESP32 may reset when serial connection opens
        time.sleep(2)

        esp32.reset_input_buffer()

        print("ESP32 connected successfully.")
        return True

    except serial.SerialException as e:
        print(f"ESP32 connection failed: {e}")
        print("Check the COM port and close Arduino Serial Monitor.")
        return False


def send_motor_command(command):
    """
    Send a command immediately when it changes.
    Repeat the command periodically as a heartbeat.
    """
    global last_command, last_command_time

    if esp32 is None or not esp32.is_open:
        return

    current_time = time.monotonic()

    command_changed = command != last_command
    heartbeat_due = (
        current_time - last_command_time >= COMMAND_INTERVAL
    )

    if command_changed or heartbeat_due:
        try:
            esp32.write(command.encode("ascii"))

            last_command = command
            last_command_time = current_time

            print(f"Motor command: {command}")

        except serial.SerialException as e:
            print(f"Serial communication error: {e}")


def close_esp32():
    """Send STOP before closing the serial connection."""
    global esp32

    if esp32 is not None and esp32.is_open:
        try:
            esp32.write(b"S")
            time.sleep(0.1)
            esp32.close()
            print("ESP32 serial connection closed.")

        except serial.SerialException as e:
            print(f"Error closing serial connection: {e}")


# ----------------------------------------------------------------------
# SEGMENTATION
# ----------------------------------------------------------------------

def make_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=HISTORY,
        varThreshold=VAR_THRESHOLD,
        detectShadows=False
    )


def segment_hand(roi, subtractor, learning_rate,
                 open_kernel, close_kernel):

    """Return largest contour or None, and filled hand mask."""

    if BLUR_KSIZE > 0:
        src = cv2.GaussianBlur(
            roi, (BLUR_KSIZE, BLUR_KSIZE), 0
        )
    else:
        src = roi

    mask = subtractor.apply(
        src,
        learningRate=learning_rate
    )

    _, mask = cv2.threshold(
        mask, 127, 255, cv2.THRESH_BINARY
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, open_kernel, iterations=1
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, close_kernel, iterations=1
    )

    if MEDIAN_KSIZE > 0:
        mask = cv2.medianBlur(mask, MEDIAN_KSIZE)

    hand_mask = np.zeros_like(mask)

    # Ignore major lighting changes
    fg_fraction = cv2.countNonZero(mask) / float(mask.size)

    if fg_fraction > MAX_FG_FRACTION:
        return None, hand_mask

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None, hand_mask

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < MIN_HAND_AREA:
        return None, hand_mask

    cv2.drawContours(
        hand_mask,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    return contour, hand_mask


# ----------------------------------------------------------------------
# HAND ANALYSIS
# ----------------------------------------------------------------------

def find_palm(hand_mask):

    """Find palm centre and approximate palm radius."""

    dist = cv2.distanceTransform(
        hand_mask,
        cv2.DIST_L2,
        5
    )

    _, radius, _, centre = cv2.minMaxLoc(dist)

    return centre, float(radius)


def find_gaps(contour, palm_c, palm_r):

    """Find valid convexity defects between fingers."""

    hull_idx = cv2.convexHull(
        contour,
        returnPoints=False
    )

    if hull_idx is None or len(hull_idx) < 4:
        return []

    defects = cv2.convexityDefects(contour, hull_idx)

    if defects is None:
        return []

    defects = np.asarray(defects).reshape(-1, 4)

    cy = palm_c[1]
    min_depth = GAP_MIN_DEPTH * palm_r

    gaps = []

    for start_idx, end_idx, far_idx, depth in defects:

        start = contour[start_idx][0].astype(np.float64)
        end = contour[end_idx][0].astype(np.float64)
        far = contour[far_idx][0].astype(np.float64)

        a = np.linalg.norm(start - end)
        b = np.linalg.norm(start - far)
        c = np.linalg.norm(end - far)

        if b == 0 or c == 0:
            continue

        cosine = np.clip(
            (b * b + c * c - a * a) / (2 * b * c),
            -1.0,
            1.0
        )

        angle = np.degrees(np.arccos(cosine))
        depth_px = depth / 256.0

        if depth_px < min_depth:
            continue

        if angle > GAP_MAX_ANGLE:
            continue

        if far[1] > cy + GAP_MAX_BELOW * palm_r:
            continue

        gaps.append((int(far[0]), int(far[1])))

    return gaps


def find_fingertips(contour, palm_c, palm_r, roi_w):

    """Detect fingertips using convex hull and k-curvature."""

    cx, cy = palm_c
    n = len(contour)

    k = int(np.clip(K_FACTOR * palm_r, 8, 45))

    if n < 2 * k + 1:
        return []

    pts = contour[:, 0, :].astype(np.float64)

    hull_idx = cv2.convexHull(
        contour,
        returnPoints=False
    )

    if hull_idx is None:
        return []

    hull_idx = hull_idx[:, 0]

    candidates = []

    for i in hull_idx:

        p = pts[i]
        px, py = p[0], p[1]

        # Ignore points near ROI edges
        if (
            px < EDGE_MARGIN
            or px > roi_w - 1 - EDGE_MARGIN
            or py < EDGE_MARGIN
        ):
            continue

        d = np.hypot(px - cx, py - cy)

        if d < TIP_MIN_DIST * palm_r:
            continue

        if py > cy + TIP_MAX_BELOW * palm_r:
            continue

        # K-curvature calculation
        v1 = pts[(i - k) % n] - p
        v2 = pts[(i + k) % n] - p

        denom = np.linalg.norm(v1) * np.linalg.norm(v2)

        if denom == 0:
            continue

        cosine = np.clip(
            np.dot(v1, v2) / denom,
            -1.0,
            1.0
        )

        angle = np.degrees(np.arccos(cosine))

        if angle > TIP_MAX_ANGLE:
            continue

        candidates.append((d, int(px), int(py)))

    # Merge duplicate detections
    candidates.sort(reverse=True)

    merge_px = TIP_MERGE_DIST * palm_r
    tips = []

    for d, x, y in candidates:

        if all(
            np.hypot(x - tx, y - ty) > merge_px
            for tx, ty in tips
        ):
            tips.append((x, y))

    return tips


def touches_roi_edge(contour, roi_w):

    """Check whether the hand is cut off by the ROI edge."""

    x, y, w, h = cv2.boundingRect(contour)

    return (
        x <= EDGE_MARGIN
        or x + w >= roi_w - EDGE_MARGIN
        or y <= EDGE_MARGIN
    )


# ----------------------------------------------------------------------
# GESTURE CLASSIFICATION
# ----------------------------------------------------------------------

def classify_direction(palm_c, fingertip):

    palm_x, palm_y = palm_c
    tip_x, tip_y = fingertip

    dx = tip_x - palm_x
    dy = tip_y - palm_y

    # Angle measured counter-clockwise from positive x-axis
    angle = np.degrees(np.arctan2(-dy, dx)) % 360

    if 70 <= angle <= 110:
        direction = "UP"

    elif 250 <= angle <= 290:
        direction = "DOWN"

    elif 110 < angle < 250:
        direction = "LEFT"

    else:
        direction = "RIGHT"

    return angle, direction


def recognise_gesture(tips, edge, palm_c):

    """Convert fingertip count and direction into a robot command."""

    if edge:
        return "UNKNOWN"

    if len(tips) == 0:
        return "STOP"

    elif len(tips) == 1:

        angle, direction = classify_direction(
            palm_c,
            tips[0]
        )

        if direction == "UP":
            return "FORWARD"

        elif direction == "DOWN":
            return "BACKWARD"

        elif direction == "LEFT":
            return "TURN LEFT"

        elif direction == "RIGHT":
            return "TURN RIGHT"

    elif len(tips) == 5:
        return "BACKWARD"

    return "UNKNOWN"


def gesture_to_motor_command(label):

    """Convert gesture label into ESP32 serial command."""

    commands = {
        "FORWARD": "F",
        "BACKWARD": "B",
        "TURN LEFT": "L",
        "TURN RIGHT": "R",
        "STOP": "S",
        "UNKNOWN": "S"
    }

    return commands.get(label, "S")


# ----------------------------------------------------------------------
# DRAWING
# ----------------------------------------------------------------------

def put_text(img, text, y, colour):

    cv2.putText(
        img,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        colour,
        2
    )


def draw_debug(roi, contour, palm_c, palm_r, gaps, tips):

    cv2.drawContours(
        roi,
        [contour],
        -1,
        GREEN,
        2
    )

    hull = cv2.convexHull(contour)

    cv2.polylines(
        roi,
        [hull],
        True,
        BLUE,
        2
    )

    cv2.circle(
        roi,
        palm_c,
        int(palm_r),
        MAGENTA,
        2
    )

    cv2.circle(
        roi,
        palm_c,
        4,
        MAGENTA,
        -1
    )

    cv2.circle(
        roi,
        palm_c,
        int(TIP_MIN_DIST * palm_r),
        GREY,
        1
    )

    for far in gaps:
        cv2.circle(roi, far, 5, RED, -1)

    for tip in tips:

        cv2.line(
            roi,
            palm_c,
            tip,
            YELLOW,
            1
        )

        cv2.circle(
            roi,
            tip,
            9,
            YELLOW,
            -1
        )


# ----------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------

def main():

    global esp32

    # Connect to ESP32 first
    if not connect_esp32():
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        close_esp32()
        return

    subtractor = make_subtractor()

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    frame_count = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Webcam frame not received.")
                send_motor_command("S")
                break

            frame = cv2.flip(frame, 1)

            roi = frame[Y1:Y2, X1:X2]
            roi_h, roi_w = roi.shape[:2]

            learning = frame_count < LEARNING_FRAMES

            rate = LEARNING_RATE if learning else 0

            # Process segmentation
            contour, hand_mask = segment_hand(
                roi,
                subtractor,
                rate,
                open_kernel,
                close_kernel
            )

            frame_count += 1

            # Default state: STOP
            label = "UNKNOWN"

            if learning:

                hand_mask[:] = 0

                put_text(
                    frame,
                    f"LEARNING BACKGROUND {frame_count}/{LEARNING_FRAMES}",
                    30,
                    YELLOW
                )

            elif contour is None:

                label = "STOP"

                put_text(
                    frame,
                    "NO HAND -> STOP",
                    30,
                    RED
                )

            else:

                palm_c, palm_r = find_palm(hand_mask)

                gaps = find_gaps(
                    contour,
                    palm_c,
                    palm_r
                )

                tips = find_fingertips(
                    contour,
                    palm_c,
                    palm_r,
                    roi_w
                )

                edge = touches_roi_edge(
                    contour,
                    roi_w
                )

                draw_debug(
                    roi,
                    contour,
                    palm_c,
                    palm_r,
                    gaps,
                    tips
                )

                # Classify gesture
                label = recognise_gesture(
                    tips,
                    edge,
                    palm_c
                )

                put_text(
                    frame,
                    f"COMMAND: {label}",
                    30,
                    YELLOW
                )

                put_text(
                    frame,
                    f"TIPS: {len(tips)}  GAPS: {len(gaps)}  PALM R: {palm_r:.0f}px",
                    60,
                    YELLOW
                )

                if edge:

                    put_text(
                        frame,
                        "HAND TOUCHES BOX EDGE",
                        90,
                        RED
                    )

            # Convert gesture into motor command
            if learning:
                motor_command = "S"
            else:
                motor_command = gesture_to_motor_command(label)

            # Send command to ESP32
            send_motor_command(motor_command)

            put_text(
                frame,
                f"ESP32 COMMAND: {motor_command}",
                120,
                GREEN
            )

            cv2.rectangle(
                frame,
                (X1, Y1),
                (X2, Y2),
                GREEN,
                2
            )

            cv2.imshow(
                "Gesture Recognition",
                frame
            )

            cv2.imshow(
                "Hand Mask",
                hand_mask
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quitting...")
                break

            elif key == ord("r"):

                # Stop before relearning
                send_motor_command("S")

                subtractor = make_subtractor()
                frame_count = 0

                print("Background reset.")

    except KeyboardInterrupt:
        print("Program interrupted.")

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:

        # Always stop motors before exiting
        try:
            if esp32 is not None and esp32.is_open:
                esp32.write(b"S")
                time.sleep(0.1)
        except serial.SerialException:
            pass

        cap.release()
        cv2.destroyAllWindows()

        close_esp32()

        print("Program safely stopped.")


if __name__ == "__main__":
    main()

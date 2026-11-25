
import cv2
import numpy as np

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

# Background subtraction settings
subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=35,
    detectShadows=False
)

# Hand detection region (ROI)
# Adjust these values if needed
x1, y1 = 160, 160
x2, y2 = 480, 440

frame_count = 0
learning_frames = 90

# Remove small isolated noise
open_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)

# Fill small gaps without excessively joining fingers
close_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)

print("Background detection started.")
print("Place your hand inside the rectangle.")
print("Press Q to quit.")
print("Press R to reset background learning.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    # Mirror the webcam image
    frame = cv2.flip(frame, 1)

    # Select the hand detection region
    roi = frame[y1:y2, x1:x2]

    # Learn the background during the first 90 frames
    if frame_count < learning_frames:
        learning_rate = 0.03
    else:
        learning_rate = 0

    # Separate moving foreground from the background
    mask = subtractor.apply(
        roi,
        learningRate=learning_rate
    )

    frame_count += 1

    
    # Remove small isolated foreground noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )

    # Close only tiny gaps in the hand mask.
    # A small kernel helps preserve spaces between fingers.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )

    # Keep only reasonably large foreground components.
    # This removes small stray patches around the hand.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cleaned_mask = np.zeros_like(mask)

    for contour in contours:
        if cv2.contourArea(contour) > 150:
            cv2.drawContours(
                cleaned_mask,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED
            )

    mask = cleaned_mask


    # Fill small gaps in the detected hand
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )

    # Find the external contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Fill enclosed black holes in the foreground
    filled_mask = np.zeros_like(mask)

    cv2.drawContours(
        filled_mask,
        contours,
        -1,
        255,
        thickness=cv2.FILLED
    )

    mask = filled_mask

    # Draw the detection region on the webcam image
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Find the largest contour, likely to be the hand
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(largest_contour)

        # Ignore small objects and noise
        if area > 1500:
            # Move ROI contour coordinates into full-frame coordinates
            contour_offset = largest_contour.copy()
            contour_offset[:, :, 0] += x1
            contour_offset[:, :, 1] += y1

            # Draw the detected hand outline
            cv2.drawContours(
                frame,
                [contour_offset],
                -1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Hand candidate: {int(area)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Display the webcam and foreground mask
    cv2.imshow("Background Detection", frame)
    cv2.imshow("Foreground Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        # Reset background learning
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=35,
            detectShadows=False
        )
        frame_count = 0
        print("Background reset. Keep your hand out of the rectangle briefly.")

cap.release()
cv2.destroyAllWindows()

import cv2
import os
import time
from datetime import datetime

# Directory to save images
save_dir = "asl_alphabet/asl_alphabet/Z"
os.makedirs(save_dir, exist_ok=True)

# Start the camera
cap = cv2.VideoCapture(0)

# Set the camera resolution (e.g., 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

print("Hold 'c' to capture images. Press 'q' to quit.")

# Frame capture rate
capture_rate = 30  # Frames per second
frame_interval = 1 / capture_rate  # Time between frames
last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Invert the camera (flip vertically and horizontally)
    frame = cv2.flip(frame, 1)

    # Display the frame
    cv2.imshow("Inverted Camera Feed", frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()

    if key == ord('q'):  # Quit if 'q' is pressed
        break
    elif key == ord('c') and current_time - last_capture_time >= frame_interval:
        # Capture and save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{save_dir}/image_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image saved: {filename}")
        last_capture_time = current_time

# Release resources
cap.release()
cv2.destroyAllWindows()
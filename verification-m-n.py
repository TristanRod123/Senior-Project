import csv
import copy
import cv2 as cv
import mediapipe as mp
from app import *
import os 
import time
import gc


# Disable OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_landmarks(Path: str, number: int, left_writer, right_writer, hands):
    try:
        image = cv.imread(Path)
        if image is None:
            print(f"Error reading image: {Path}")
            return
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Determine if it's a left or right hand
                label = handedness.classification[0].label
                is_left_hand = label == "Left"

                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                
                # Write to the appropriate dataset file
                writer = left_writer if is_left_hand else right_writer
                writer.writerow([number, *pre_processed_landmark_list])
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Free up memory by deleting the images explicitly
        del image, debug_image

    # Clean up any mess I've missed
    gc.collect()
    

def iter_archive(cur_path: str):
    # Load model a single time to prevent overflow
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    num = 0

    # File paths for left and right hand landmark CSVs
    left_csv_path = 'model/keypoint_classifier/left_hand_keypoint.csv'
    right_csv_path = 'model/keypoint_classifier/right_hand_keypoint.csv'

    with open(left_csv_path, "a", newline="") as left_file, \
         open(right_csv_path, "a", newline="") as right_file:
        left_recorder = csv.writer(left_file)
        right_recorder = csv.writer(right_file)

        for sub_dir in os.listdir(cur_path):
            csv_label_path = 'model/keypoint_classifier/M&N.csv'

            with open(csv_label_path, "a", newline="") as label_file:
                writer = csv.writer(label_file)
                writer.writerow([sub_dir])

            letter_dir = os.path.join(cur_path, sub_dir)

            for img in os.listdir(letter_dir):
                full_path = os.path.join(letter_dir, img)
                get_landmarks(full_path, num, left_recorder, right_recorder, hands)

            num += 1

        # Close the MediaPipe Hands Model
        hands.close()

# Call the main function
iter_archive("M&N")

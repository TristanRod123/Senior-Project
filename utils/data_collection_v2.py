import csv
import copy
import cv2 as cv
import mediapipe as mp
from app import *
import os 
import time
import gc
import shutil  # Import shutil for file copying

# Disable OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_landmarks(Path: str, number: int, writer, hands, success_folder: str, cur_path: str):
    try:
        image = cv.imread(Path)
        if image is None:
            print(f"Error reading image: {Path}")
            return False  # Return False if image could not be read
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, _ in zip(results.multi_hand_landmarks,
                                         results.multi_handedness):
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                
                # Write to the dataset file
                writer.writerow([number, *pre_processed_landmark_list])                


            # Copy the image to the success folder with the same directory structure
            relative_path = os.path.relpath(Path, cur_path)  # Calculate relative path from cur_path
            dest_path = os.path.join(success_folder, relative_path)  # Destination path in success_folder
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Create the directory structure if it doesn't exist
            shutil.copy(Path, dest_path)  # Copy the image to the success folder

            return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Free up memory by deleting the images explicitly
        del image, debug_image

    # Clean up any mess I've missed
    gc.collect()

def iter_archive(cur_path: str, success_folder: str):
    # Load model a single time to prevent overflow
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(        
        max_num_hands=1,
        min_detection_confidence=.7,
        min_tracking_confidence=.5,
    )

    num = 0
    no_hands_count = 0  # Counter for images with no hands detected

    csv_keypoint_path = 'model/keypoint_classifier/keypoint_original_data_v2.csv'

    with open(csv_keypoint_path, "a", newline="") as f:
        recorder = csv.writer(f)
        
        for sub_dir in os.listdir(cur_path):

            csv_label_path = 'model/keypoint_classifier/keypoint_classifier_label_original_data_v2.csv'

            with open(csv_label_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([sub_dir])
            
            letter_dir = os.path.join(cur_path, sub_dir)
            
            for img in os.listdir(letter_dir):
                full_path = os.path.join(letter_dir, img)
                if get_landmarks(full_path, num, recorder, hands, success_folder, cur_path):
                    pass
                else:
                    print(f"No hands detected: {full_path}")
                    no_hands_count += 1  # Increment counter for images with no hands
                
            num += 1

        # Close the mediapipe Hands Model
        hands.close()

    print(f"\nTotal images with no hands detected: {no_hands_count}")

# Create the success folder if it doesn't exist
success_folder_path = "flipped_asl_alphabet/processed_images"
os.makedirs(success_folder_path, exist_ok=True)

# Call the iter_archive function with the path and the success folder
iter_archive("flipped_asl_alphabet/asl_alphabet", success_folder_path)

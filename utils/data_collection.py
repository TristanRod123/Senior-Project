import csv
import copy
import cv2 as cv
import mediapipe as mp
from app_functions import *
import os 
import time
import gc

# Disable OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_landmarks(Path:str, number: int, writer, hands):
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
            for hand_landmarks, _ in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                
                # Write to the dataset file
                writer.writerow([number, *pre_processed_landmark_list])                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Free up memory by deleting the images explicitly
        del image, debug_image

    # Clean up any mess I've missed
    gc.collect()
    

def iter_archive(cur_path: str):
    #Load model a single time to prevent overflow
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(        
        max_num_hands=1,
        min_detection_confidence=.7,
        min_tracking_confidence=.5,
    )

    num = 0

    csv_keypoint_path = 'model/keypoint_classifier/keypoint.csv'

    with open(csv_keypoint_path, "a", newline="") as f:
        recorder = csv.writer(f)
        
        for sub_dir in os.listdir(cur_path):

            csv_label_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'

            with open(csv_label_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([sub_dir])
            
            letter_dir = os.path.join(cur_path, sub_dir)
            
            for img in os.listdir(letter_dir):
                full_path = os.path.join(letter_dir, img)
                get_landmarks(full_path, num, recorder, hands)
                
            num += 1
        # Close the mediapipe Hands Model
        hands.close()
            

iter_archive("flipped_archive/Train_Alphabet")

# Senior-Project
ASL Recognition and Translation
- Utilizing MediaPipe and MultilayerPerceptron to recognize ASL letters frame-by-frame

# Setup
INSTALL ALL DEPENDENCIES FROM requirements.txt

```pip install -r requirements.txt```

- NVIDIA GPU required to run appUI.py
- Need Python version 3.9-3.11
- Need a webcam (integrated or not)

DOWNLOAD AND INSTALL DATASET (not necessary for running demo, data points recorded on keypoint.csv)
-  https://utrgv-my.sharepoint.com/:u:/g/personal/elijah_khamo01_utrgv_edu/ERlo29EEuoRAmaxpRSBpiWkBEOSU-i46tNT5E2G2t4pXTw?e=v8WOOx (EXPIRES EVERY 90 DAYS)
-  Contact ```Salomonibarra01@outlook.com``` for a new link

# Demo
Connect Webcam

```python AppUI.py```

After running the demo press the live button to show an asl hand sign from A-Z simply 

You can make a word or sentence by showing the sign and concatenating, then clear the output by pressing the clear button.  

[Screencast from 2024-12-12 05-22-26.webm](https://github.com/user-attachments/assets/65c80fdf-be2b-4410-b7fe-6c4fbaf6a1ad)


# Directory
```
│  .gitattributes
|  .gitignore
|  AppUI.py
|  LICENSE
|  README.md
|  app_functions.py
|  cnn_keypoint_classification.ipynb
|  fnn_keypoint_classification.ipynb
│  fnn_keypoint_classification_v2.ipynb  (MAIN MODEL)
│  keypoint_classification_cross_val.ipynb
│  mlp_image_compare.ipynb               (MAIN COMPARISON)
|  mlp_mediapipe_compare.ipynb           {MAIN COMPARISON)
|  requirements.txt
|  testing-distance.ipynb
|  testing.ipynb
├─model
│  ├─keypoint_classifier
│     │  keypoint.csv
│     │  keypoint_classifier.hdf5
│     │  keypoint_classifier.py
│     │  keypoint_classifier.tflite
│     └─ keypoint_classifier_label.csv
│            
│-utils
│  | calc.py
│  │ cvfpscalc.py
│  │ data_collection.py
│  │ distance.py
│  │ flip_images.py
│  | image_collection.py
```
# Files

### fnn_keypoint_classification_v2.ipynb
- This is a model training script for ASL sign recognition.

### AppUI 
- opens the interface that allows you to try making hand signs and detecting them.

### fnn_keypoint_classification_v2.ipynb 
- is the actual model that we used for training.

### data_collection.py 
-takes the images and puts them through mediapipe which then converts it to data the model uses to train on.

### keypoint_classifier_label.csv 
- is where the labels are for each class

### keypoint.csv 
- is where the data is stored from data_collection.py that is used to train the model

### flip_images.py 
- is used to augment the data and flip the images for right-to-left-hand conversion and vice versa

### mlp_image_compare.ipynb 
- is another model we created that takes images instead of tabular data

### testing-distance.ipynb 
-another model we created that uses the  distances between neighbor vectors on the hand

### image_collection.py 
- is used to collect images for media pipe to process by holding 'c'

### cnn_keypoint_classification.ipynb 
- another model we made where instead of an mlp if uses a cnn

### keypoint_classification_cross_val.ipynb 
- is when we introduced K-fold cross-validation
# Training
## Step 1 - Image collection
Our model is not trained using images. We extract coordinates from landmarks on hands using mediapipe and use those coordinates to train our model.
To collect these images to train the model with new classes do the following 
- Before you proceed make sure you have a folder where you can store all the images and make sure to change the variable at the top named save_dir to your chosen folder path.
  -  Example how to make a proper folder "Archive\Hello" Keep in mind that the subfolder name will become the name of the new class you are training
- Make your way to \utils\image_collection.py when you run the file a camera should open.  As soon as that is done all that has to be done is 
press 'c' to capture the images and stop by pressing 'q' `when you are satisfied.
## step 2 - Data Collection
Once you have your images ready it is time to extract the coordinates needed. 
- Make your way to \utils\data collection.py at the bottom of the code there is a function named "iter_archive" that takes in an argument for the path of the folder that contains all the all the classes you will be training.
- Give the function the file path of your folder containing all your classes which contain all the images of the classes you will be training 
- When you run the function will iterate through your folder that contains multiple classes and will then iterate through each class folder and extract the coordinates from each image. 
  - It will then append the coordinates to the csv files needed to train the model. Keep in mind that the name of the folder that contains the images will become a new class. 
## step 3 - Training
Finally to train the model might be more difficult depending on the resources of your computer. To train the model just run every cell in the file "fnn_keypoint_classification_v2.ipynb"
This will train the model with your newly added classes. 

# Credits

[Reference of Kazuhito Takahashi's Work](https://github.com/kinivi/hand-gesture-recognition-mediapipe.git)

[MediaPipe Hand Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)

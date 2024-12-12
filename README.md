# Senior-Project
ASL recognition and translation

# Requirements
PLEASE INSTALL ALL DEPENDENCIES IN rquirements.txt

pip install -r requirements.txt

NVIDIA GPU required to run appUI.py

# Demo
How to run it using your webcam

cd Senior-Project

python AppUI.py

[Screencast from 2024-12-12 05-22-26.webm](https://github.com/user-attachments/assets/7cf5888c-51d8-4e97-a986-7d5c09afa714)

# Files

\AppUI opens the interface that allows you to try making hand signs and detecting them.

\fnn_keypoint_classification_v2.ipynb is the actual model that we used for training.

\utils\data_collection.py takes the images and puts them through mediapipe which then converts it to data the model uses to train on.

\model\keypoint_classifier\keypoint_classifier_label.csv is where the labels are for each class

\model\keypoint_classifier\keypoint.csv is where the data is stored from data_collection.py that is used to trin the model

\utils\flip_images.py is used to augment the data and flip the images for right-to-left-hand conversion and vice versa

\mlp_image_compare.ipynb is another model we created that takes images instead of tabular data

\testing-distance.ipynb another model we created that uses the  distances between neighbor vectors on the hand

\utils\image_collection.py is used to collect images for media pipe to process by holding 'c'

\cnn_keypoint_classification.ipynb another model we made where instead of an mlp if uses a cnn

\keypoint_classification_cross_val.ipynb is when we introduced K-fold cross-validation

# Credits

The original code and model our own project was based can be found here

https://github.com/kinivi/hand-gesture-recognition-mediapipe.git

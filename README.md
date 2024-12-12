# Senior-Project
ASL recognition and translation 

# Requirements
PLEASE INSTALL REQUIRMENTS FOLDER

pip install -r requirements. txt

# Demo
How to run it using yor webcam

cd Senior-Project
python AppUI.py

# Files

\AppUI is what open the interface tat allows you to try making handsigns and detecting it.
\fnn_keypoint_classification_v2.ipynb is the actual model that we used for training.
\utils\data_collection.py is what takes the images and puts it trough mediapipe which then converts it to data that the model uses to trin on.
\model\keypoint_classifier\keypoint_classifier_label.csv is where the labels are for each class
\model\keypoint_classifier\keypoint.csv is where the data is stored from data_collection.py that is used to trin the model
\utils\flip_images.py is used to augment the data and flip the images for right to left hand conversion and viceversa
\mlp_image_compare.ipynb is another model we created that takes images instead of tabluar data
\testing-distance.ipynb another model we created that uses the  distances betweeen neighbor vectors on the hand
\utils\image_collection.py is used to collect images for mediapipe to process by holding 'c'
\cnn_keypoint_classification.ipynb another model we made where intead of an mlp if uses a cnn
\keypoint_classification_cross_val.ipynb is when we introduced K-fold cross validation

import cv2 as cv
import os
import shutil

def flip_and_copy_images(input_dir, output_dir, flip_type=1):
    """
    Flip all images in a directory and its subdirectories, save the original and flipped images
    to another directory while maintaining the same structure.
    
    Parameters:
        input_dir (str): Directory containing the images to flip.
        output_dir (str): Directory where the original and flipped images will be saved.
        flip_type (int): Type of flip (1 for horizontal, 0 for vertical, -1 for both).
    """
    # Iterate through all files and subdirectories in the input directory
    for root, dirs, files in os.walk(input_dir):
        # Calculate the relative path from the input directory
        relative_path = os.path.relpath(root, input_dir)
        # Construct the corresponding path in the output directory
        output_subdir = os.path.join(output_dir, relative_path)

        # Ensure the output subdirectory exists
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for filename in files:
            # Construct the full paths for the input and output images
            image_path = os.path.join(root, filename)
            original_image_path = os.path.join(output_subdir, filename)
            flipped_image_path = os.path.join(output_subdir, f"flipped_{filename}")

            # Read the image
            image = cv.imread(image_path)
            if image is None:
                print(f"Error reading image: {image_path}")
                continue

            # Copy the original image to the output directory
            shutil.copy(image_path, original_image_path)

            # Flip the image
            flipped_image = cv.flip(image, flip_type)

            # Save the flipped image
            cv.imwrite(flipped_image_path, flipped_image)
            print(f"Saved flipped image: {flipped_image_path}")
            print(f"Copied original image: {original_image_path}")

# Example usage
input_directory = 'archive/Train_Alphabet'
output_directory = 'flipped_archive/Train_Alphabet'
flip_and_copy_images(input_directory, output_directory, flip_type=1)  # 1 for horizontal flip

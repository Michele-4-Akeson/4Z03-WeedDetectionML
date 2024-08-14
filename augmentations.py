'''
augmentations.py:
    This script performs two data augmentation techniques (flipping and color transformation). 
    by iterating through all images in the weed and crop subdirectories:
        - flipping:
            weed and crop images are copied to new data/flippedData/ directory
            new weed and crop images are created from original images with flipping geometric transformation

        - color transformation:
            weed and crop images are converted to gray scale and copied to new data/colorlessData/

'''

import os, shutil
import re
from PIL import Image, ImageOps


sortedWeedImageDirectory = "data/sortedData/weed"
sortedCropImageDirectory = "data/sortedData/crop"

flippedWeedImageDirectory = "data/flippedData/weed"
flippedCropImageDirectory = "data/flippedData/crop"

colorlessWeedImageDirectory = "data/colorlessData/weed"
colorlessCropImageDirectory = "data/colorlessData/crop"

# Create directories if they don't exist
os.makedirs(flippedWeedImageDirectory, exist_ok=True)
os.makedirs(flippedCropImageDirectory, exist_ok=True)
os.makedirs(colorlessWeedImageDirectory, exist_ok=True)
os.makedirs(colorlessCropImageDirectory, exist_ok=True)

# Create an array of full paths to image files from weed and crop directories
weedImageFiles = [os.path.join(sortedWeedImageDirectory, filename) for filename in os.listdir(sortedWeedImageDirectory)]
cropImageFiles = [os.path.join(sortedCropImageDirectory, filename) for filename in os.listdir(sortedCropImageDirectory)]

# Function to process and save images
def augment_images(imageFiles, flippedDirectory, colorlessDirectory):
    for imagePath in imageFiles:
        shutil.copy(imagePath, flippedDirectory)
        imageFileNumber = re.findall('\d+\.\d+|\d+', imagePath)[1]
        with Image.open(imagePath) as img:
            # Create a flipped image
            flippedImage = ImageOps.mirror(img)
            flippedImage.save(os.path.join(flippedDirectory, f"{imageFileNumber}_flipped.jpeg"))

            # Create a grayscale (colorless) image
            colorlessImage = img.convert("L")
            colorlessImage.save(os.path.join(colorlessDirectory, f"{imageFileNumber}_colorless.jpeg"))

# Augment weed images
augment_images(weedImageFiles, flippedWeedImageDirectory, colorlessWeedImageDirectory)

# Augment crop images
augment_images(cropImageFiles, flippedCropImageDirectory, colorlessCropImageDirectory)

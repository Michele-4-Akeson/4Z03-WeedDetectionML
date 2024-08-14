'''
imageSize.py:
    This script is used to determine the max and minimum size images contained in the crop/weed dataset.
    It was determined that all images are of size 512 x 512 px
'''



import os
import sys
from PIL import Image


sortedWeedImageDirectory = "data/formattedData/weed"
sortedCropImageDirectory = "data/formattedData/crop"

weedImageFiles = os.listdir('data/formattedData/weed')
cropImageFiles = os.listdir('data/formattedData/crop')

# create an array of full path to image file
weedFiles = [os.path.join(sortedWeedImageDirectory, filename) for filename in weedImageFiles]
# create an array of full path to json file
cropFiles = [os.path.join(sortedCropImageDirectory, filename) for filename in cropImageFiles]

# Path to the image file
maxWidth, maxHeight, minWidth, minHeight = 0, 0, sys.maxsize, sys.maxsize



for image in weedFiles:
    with Image.open(image) as img:
        width, height = img.size
        maxHeight = max(maxHeight, height)
        maxWidth = max(maxWidth, width)
        minHeight = min(minHeight, height)
        minWidth = min(minWidth, width)


for image in cropFiles:
    with Image.open(image) as img:
        width, height = img.size
        maxHeight = max(maxHeight, height)
        maxWidth = max(maxWidth, width)
        minHeight = min(minHeight, height)
        minWidth = min(minWidth, width)



print(f"max: {maxWidth}x{maxHeight} min: {minWidth}x{minHeight}")
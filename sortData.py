import os, shutil, re, json

imageDirectory = 'data/data/img'
annotationsDirectory = 'data/data/ann'

sortedWeedImageDirectory = "data/sortedData/weed"
sortedCropImageDirectory = "data/sortedData/crop"

imageFiles = os.listdir('data/data/img')
annotationFiles = os.listdir('data/data/ann')

# create an array of full path to image file
images = [os.path.join(imageDirectory, filename) for filename in imageFiles]
# create an array of full path to json file
annotations = [os.path.join(annotationsDirectory, filename) for filename in annotationFiles]


images.sort()
annotations.sort()


# validate correct sorting of images and annotations files
for i in range(0, len(images)):
    image = images[i]
    annotation = annotations[i]
    imageFileNumber = re.findall('\d+\.\d+|\d+', image)
    annotationFileNumber = re.findall('\d+\.\d+|\d+', annotation)
    isSame = imageFileNumber == annotationFileNumber
    if (not isSame):
        print("ERROR")
        break

    print(f"{image} == {annotation} \n{isSame}: {imageFileNumber} == {annotationFileNumber}")


# separate image files into two different sub-directories based on the "classTitle" value in the associated .json file
os.makedirs(sortedWeedImageDirectory, exist_ok=False)
os.makedirs(sortedCropImageDirectory, exist_ok=False)

weedCount, cropCount = 0, 0 
for i in range(0, len(images)):
    image = images[i]
    annotation = annotations[i]

    with open(annotation, 'r') as jsonFile:
        data = json.load(jsonFile)
        className = data["objects"][0]["classTitle"] 
        print(className)
        if (className == "weed"):
            shutil.copy(image, sortedWeedImageDirectory)
            weedCount += 1
        else:
            shutil.copy(image, sortedCropImageDirectory)
            cropCount += 1


# 666 weeds, 634 crop
print(f"Weed Count:{weedCount}, Crop Count: {cropCount}")

    

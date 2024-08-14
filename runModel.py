'''
runModel.py:
    this script creates and trains three models on weed vs crop data:
    1. Control: The first model is trained on data in 'data/sortedData' which has no augmentations applied to it
    2. Flipped Augmentation: The second model is trained on data in 'data/flippedData' which contains all images in the original dataset with an additional copy of
       every image that is horizontally flipped across the y-axis
    3. Colorless Augmentation: The third model is trained on data in 'data/colorlessData' which contains all images in the original dataset, but converted to grayscale



    After creating and training each model using the createModel() function from cnn.py, this script calls graphAndSaveResult() from graphResults.py to graph and save the 
    recorded metrics of each model

'''


from cnn import createModel
from graphResults import graphAndSaveResults

results = createModel("data/sortedData")
graphAndSaveResults(results, "results/control")



results = createModel("data/flippedData")
graphAndSaveResults(results, "results/flipped")



results = createModel("data/colorlessData")
graphAndSaveResults(results, "results/colorless")

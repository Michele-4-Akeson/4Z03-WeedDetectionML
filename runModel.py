
from cnn import createModel
from graphResults import graphAndSaveResults
# the value a logistic regression model must exceed to be considered a positive prediction

results = createModel("data/sortedData")
graphAndSaveResults(results, "results/control")



results = createModel("data/flippedData")
graphAndSaveResults(results, "results/flipped")



results = createModel("data/colorlessData")
graphAndSaveResults(results, "results/colorless")

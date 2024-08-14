'''
graphResults.py
    contains two functions, savePlot(), and graphAndSaveResults(), which are used to plot the captured
    metrics of training a cnn model.

    These functions work to graph training and validation vs. Epochs for loss, accuracy, precision, and recall,
    and then save those graphs to the results folder
'''

import matplotlib.pyplot as plt



# Define a function to save individual plots
def save_plot(data, val_data, title, ylabel, filename):
    '''
    plots a graph with title, title, displaying data vs. epochs and val_data vs. epochs
    and then saves said plot with filename
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(data, label='Train')
    plt.plot(val_data, label='Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def graphAndSaveResults(history, graphTitle):
    '''
    Extracts the required values from the model's training history, history,
    and saves 4 graphs to the results folder with a title, graphTitle
    '''

    loss = history.history['loss']
    accuracy = history.history['accuracy']
    precision = history.history['precision']
    recall = history.history['recall']

    val_loss = history.history["val_loss"]
    val_accuracy = history.history['val_accuracy']
    val_precision = history.history['val_precision']
    val_recall = history.history['val_recall']


        
    # Save accuracy plot
    save_plot(
        accuracy, 
        val_accuracy,
        f'{graphTitle} Model Accuracy',
        'Accuracy',
        f'{graphTitle}_accuracy_plot.png'
    )

    # Save loss plot
    save_plot(
        loss, 
        val_loss,
         f'{graphTitle} Model Loss',
        'Loss',
        f'{graphTitle}_loss_plot.png'
    )

    # Save precision plot
    save_plot(
    precision, 
    val_precision,
         f'{graphTitle} Model Precision',
        'Precision',
        f'{graphTitle}_precision_plot.png'
    )

    # Save recall plot
    save_plot(
        recall, 
        val_recall,
        f'{graphTitle} Model Recall',
        'Recall',
        f'{graphTitle}_recall_plot.png'
    )







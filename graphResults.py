import matplotlib.pyplot as plt



# Define a function to save individual plots
def save_plot(data, val_data, title, ylabel, filename):
    '''
    plots a graph with title, title, displaying data vs. epochs and val_data vs. epochs
    and then saves said plot to at filename
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


def graphAndSaveResults(history, experimentName):

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
        f'{experimentName} Model Accuracy',
        'Accuracy',
        f'{experimentName}_accuracy_plot.png'
    )

    # Save loss plot
    save_plot(
        loss, 
        val_loss,
         f'{experimentName} Model Loss',
        'Loss',
        f'{experimentName}_loss_plot.png'
    )

    # Save precision plot
    save_plot(
    precision, 
    val_precision,
         f'{experimentName} Model Precision',
        'Precision',
        f'{experimentName}_precision_plot.png'
    )

    # Save recall plot
    save_plot(
        recall, 
        val_recall,
        f'{experimentName} Model Recall',
        'Recall',
        f'{experimentName}_recall_plot.png'
    )







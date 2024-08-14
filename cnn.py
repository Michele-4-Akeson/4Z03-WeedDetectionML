
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras import layers, Model
from keras.optimizers import RMSprop
from keras.utils import image_dataset_from_directory


def createModel(directory):
    """
    This function creates the layers of convolutional neural network, specifying that the
    input is an image of size 'imageSize' and returns an output of a positive prediction if
    the probability value is higher greater than or equal to the classification threshold

    This Model records the following metrics:
        BinaryAccuracy: proportion of correctly classified images
        Precision: proportion of true positives among the predicted positives
        Recall: Proportion of true positives among the actual positives
    
    Args:
        directory (str): relative path to directory containing two sorted sub directories containing images of each class for which the model most distinguish  
        imageSize (int): The width x height of the image sent to the input layer
        classificationThreshold (float): a value between 0-1 inclusively representing the required probability to produce a positive prediction

    Returns:
        The classification model history after training (recorded metrics)
    """

    classificationThreshold = 0.7
    imageSize = 150

    # Creates input layer: expects images that are 150x150 px containing 3 color channels (R, G, B)
    # Resizing is achieved by the Rescaling layer
    img_input = layers.Input(shape=(imageSize, imageSize, 3))

    # Add the Rescaling layer to normalize the pixel values from 0-255 to 0-1
    x = layers.Rescaling(1./255)(img_input)

    # First convolution extracts 16 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(16, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Second convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Third convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Flatten feature map to a 1-dim tensor so we can add fully connected layers
    x = layers.Flatten()(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = layers.Dense(512, activation='relu')(x)

    # Create output layer with a single node and sigmoid activation 
    # Sigmoid used as this is a binary classification problem (weed - not weed)
    output = layers.Dense(1, activation='sigmoid')(x)

    # Create model:
    # input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully 
    # connected layer + sigmoid output layer
    model = Model(img_input, output)

    # Outputs a table summarizing how each layer transforms the data
    model.summary()

    METRICS = [
        BinaryAccuracy(name='accuracy', threshold=classificationThreshold),
        Precision(name='precision',thresholds=classificationThreshold),
        Recall(name="recall", thresholds=classificationThreshold)
    ]

    # Binary crossentropy loss, because it's a binary classification problem and our final activation is a sigmoid
    model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(learning_rate=0.001),
                metrics=METRICS)

                    # Create datasets
    trainingDataSet = image_dataset_from_directory(
        directory,  # This is the source directory for training images
        image_size=(imageSize, imageSize), # size that images in subdirectories will be scaled to; set to 512 as we don't want any scaling 
        batch_size=20,
        label_mode='binary',  # Since we use binary_crossentropy loss, we need binary labels
        validation_split=0.2,  # Split 20% for validation if needed
        subset="training",
        seed=100 
    )

    validationDataSet = image_dataset_from_directory(
        directory,  # This is the source directory for validation images
        image_size=(imageSize, imageSize), # size that images in subdirectories will be scaled to; set to 512 as we don't want any scaling 
        batch_size=20,
        label_mode='binary',  # Since we use binary_crossentropy loss, we need binary labels
        validation_split=0.2,  # Split 20% for validation
        subset="validation",
        seed=100
    )

    # Train the model
    history = model.fit(
        trainingDataSet,
        epochs=10, 
        validation_data=validationDataSet
    )
    
    return history

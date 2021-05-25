import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    # init directory list from os.listdir
    directories = os.listdir(data_dir)

    # init image_array
    images_array = []
    labels_array = []

    # looping through dataset in directories
    for directory in directories:
        image_flies_path = data_dir + os.sep + directory  # path to individual image category directory
        image_flies = os.listdir(image_flies_path)  # list of paths to images of a category

        # logging through images individual category
        for image_file in image_flies:
            image_path = data_dir + os.sep + directory + os.sep + image_file  # path to image

            # load image
            image = cv2.imread(image_path)

            # resize image
            resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # image label
            image_label = int(directory)

            # add image and label to array
            images_array.append(resized_image)
            labels_array.append(image_label)

        # print("loaded category:", directory)

    # print("Dataset loaded successfully")
    return images_array, labels_array


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # init filter sizes
    filter_size = {8: 8, 16: 16, 32: 32, 64: 64, 128: 128}

    # init activation functions
    activation_function = {"relu": "relu", "softmax": "softmax"}

    # init pooling sizes
    pooling_sizes = {"2x2": (2, 2), "3x3": (3, 3), "5x5": (5, 5), "7x7": (7, 7)}

    # init kernel sizes
    kernel_sizes = {"2x2": (2, 2), "3x3": (3, 3), "5x5": (5, 5), "7x7": (7, 7)}

    # init hidden units
    hidden_units = {64: 64, 128: 128, 256: 256, 512: 512}

    # init dropout values
    dropout = {"30%": 0.3, "50%": 0.5, "70%": 0.7}

    # Create CNN model
    model = tf.keras.models.Sequential([

        # Convolution layer 1.
        tf.keras.layers.Conv2D(
            filter_size[32],
            kernel_sizes["3x3"],
            activation=activation_function["relu"],
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Max-Pooling layer 1.
        tf.keras.layers.MaxPool2D(pool_size=pooling_sizes["3x3"]),

        # Convolution layer 2.
        tf.keras.layers.Conv2D(
            filter_size[32],
            kernel_sizes["3x3"],
            activation=activation_function["relu"],
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Max-Pooling layer 2.
        tf.keras.layers.MaxPool2D(pool_size=pooling_sizes["2x2"]),
        #
        # # Convolution layer 3.
        # tf.keras.layers.Conv2D(
        #     filter_size[32],
        #     kernel_sizes["3x3"],
        #     activation=activation_function["relu"],
        #     input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        # ),
        #
        # # Max-Pooling layer 3.
        # tf.keras.layers.MaxPool2D(pool_size=pooling_sizes["2x2"]),

        # Flatten layer
        tf.keras.layers.Flatten(),

        # Hidden layers 1 with dropout
        tf.keras.layers.Dense(hidden_units[512], activation=activation_function["relu"]),
        tf.keras.layers.Dropout(dropout["50%"]),

        # # Hidden layers 2 with dropout
        # tf.keras.layers.Dense(hidden_units[128], activation=activation_function["relu"]),
        # tf.keras.layers.Dropout(dropout["50%"]),
        #
        # # Hidden layers 3 with dropout
        # tf.keras.layers.Dense(hidden_units[512], activation=activation_function["relu"]),
        # tf.keras.layers.Dropout(dropout["50%"]),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation=activation_function["softmax"])
    ])

    # Compiling model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # # print model summary
    # model.summary()

    return model


if __name__ == "__main__":
    main()

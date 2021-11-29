import os
import numpy as np
import sys
from model import build_model
from keras.utils import np_utils
from keras.datasets import mnist
import time

# starting time
start = time.time()
np.random.seed(123)  # for reproducibility

def load_data():
    # Load pre-shuffled MNIST data into train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape to (img_cols, img_rows, img_depth)
    img_cols = x_train.shape[1]
    img_rows = x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], img_cols, img_rows, 1)
    x_test = x_test.reshape(x_test.shape[0], img_cols, img_rows, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Convert 1-dimensional class arrays to 10-dimensional class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    # conv neural network
    model = build_model()

    epoch = 10
    model.fit(x_train, y_train, batch_size=32, epochs=epoch, verbose=0)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # end time
    end = time.time()
    # total time taken
    print(f"Runtime of the program is {end - start}")

    # serialize model to JSON
    model_json = model.to_json()
    with open("Sequential_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Sequential_model.h5")
    print("Saved model to disk")

if __name__ == "__main__":
    main()
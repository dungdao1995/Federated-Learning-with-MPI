import os
import sys
import time
import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist
from model import build_model


# starting time
start = time.time()
np.random.seed(123)  # for reproducibility

number_processor = 2

#===================Split data for each processors=====================
def splited_data(x_train, y_train):
    # each worker can only see some part of the data
    x_train = np.split(x_train,number_processor)
    y_train = np.split(y_train,number_processor)

    data = []
    for i in range(number_processor):
        x_local, y_local = normalized_data( x_train[i], y_train[i])
        data.append([x_local, y_local])


    return data

def normalized_data(x_train, y_train):
    # reshape to (img_cols, img_rows, img_depth)
    img_cols = x_train.shape[1]
    img_rows = x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], img_cols, img_rows, 1)

    #Normalization
    x_train = x_train.astype('float32')
    x_train /= 255 #in the range 0 to 1

    #One-hot-labels
    # Convert 1-dimensional class arrays to 10-dimensional class matrices
    y_train = np_utils.to_categorical(y_train, 10)

    return (x_train, y_train)

def Decentralized_FL():
    #Loading and splitting the DATA
    # Load pre-shuffled MNIST data into train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #We split data and Normalized the data
    data = splited_data(x_train, y_train)

    #Models of all clients
    models = []
    for i in range(number_processor):
        i = build_model()
        models.append(i)

    epoch = 1
    for e in range(epoch):
        #Update weight of local model in each interation
        weight_updates = []
        print("Epoch: ", e+1)
        for processor in range(number_processor):
            print("Processor ",processor+1, 'is running!')
            #Data for each processor
            x_train_local, y_train_local = data[processor]
            models[processor].fit(x_train_local, y_train_local, batch_size=32, epochs=1, verbose=2)

            #Local weight update: containing number of sample and weight in each processor
            local_updates = [x_train_local.shape[0], models[processor].get_weights()]
            weight_updates.append(local_updates)

        for processor in range(number_processor):
            #Federated Average
            #Initial weight at Server with the Shape of W0
            weights = [np.zeros(l.shape) for l in weight_updates[0][1]]
            #FedSGD
            for i in range(len(weights)):
                N = 0.0
                for n, w in weight_updates: #number of sample and the weight in each processor
                    weights[i] += n * w[i]
                    N += n
                #Average
                weights[i] /= N

            #Set the new Weight in Server
            models[processor].set_weights(weights)


    #after training is over, try to test
    x_test, y_test = normalized_data(x_test,y_test)
    score = models[0].evaluate(x_test, y_test, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    end = time.time()
    # total time taken
    print(f"Runtime of the program is {end - start}")
if __name__ == "__main__":
    Decentralized_FL()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 00:40:32 2021

@author: apple
"""
from mpi4py import MPI
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

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

SERVER = 0 # rank 0 is the server

#===================Split data for each processors=====================
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


def main():
    #Data Decomposition
    if rank ==0:
        # Load pre-shuffled MNIST data into train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        #Split the data into smaller parts
        x_train = np.split(x_train, size,axis = 0)
        #Label
        y_train = np.split(y_train, size,axis = 0)
    else:
        x_train = None
        y_train = None
    #Scatter Data
    x_train = comm.scatter(x_train, root=0)
    y_train = comm.scatter(y_train, root=0)
    # Normalized Data
    x_train, y_train = normalized_data(x_train, y_train)

    model = build_model()

    epoch = 10
    for e in range(epoch):
        if rank == 0:
            print("Server's Epoch: " + str(e + 1))
            sys.stdout.flush() #Flush the buffer

        # broadcast server's model to all processors
        weights = model.get_weights()
        weights = comm.bcast(weights, root=0)
        model.set_weights(weights)

        model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=0)
        comm.Barrier()
        # Federating step for Centralized FL
        #data contain number of sample and weight in each processor
        data = [x_train.shape[0], model.get_weights()]

        #Server will get all weights from all processors and calculated Average
        data = comm.gather(data, root=0)
        comm.Barrier()

        if rank == 0:
            #Initial weight at Server with the Shape of W0
            weights = [np.zeros(l.shape) for l in data[0][1]]
            #FedSGD
            for i in range(len(weights)):
                N = 0.0
                for n, w in data: #number of sample and the weight in each processor
                    weights[i] += n * w[i]
                    N += n
                #Average
                weights[i] /= N
            #Set the new Weight in Server
            model.set_weights(weights)
        comm.Barrier()

    # after training is over, try to test
    if rank == SERVER:
        # end time
        end = time.time()
        # total time taken
        print(f"Runtime of the program is {end - start}")

        x_test, y_test = normalized_data(x_test, y_test)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


if __name__ == "__main__":
    main()

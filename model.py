from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

#====================Model of Neural Network==========================
def build_model():
    # Build the model
    model = Sequential()

    model.add(Convolution2D(32, kernel_size = (3, 3), activation='relu', input_shape=(28, 28, 1)))

    model.add(Convolution2D(32, kernel_size = (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    return model

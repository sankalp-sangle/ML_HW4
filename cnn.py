from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import LeakyReLU


class CNN(object):
    def __init__(self):
        # change these to appropriate values
        self.batch_size = 256
        self.epochs = 4
        self.init_lr= 0.001 #learning rate

        # No need to modify these
        self.model = None

    def get_vars(self):
        return self.batch_size, self.epochs, self.init_lr

    def create_net(self):
        '''
        In this function you are going to build a convolutional neural network based on TF Keras.
        First, use Sequential() to set the inference features on this model. 
        Then, use model.add() to build layers in your own model
        Return: model
        '''

        #TODO: implement this
        model = Sequential()

        model.add(Conv2D(8, (3, 3), input_shape=(28, 28, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.3))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.3))

        model.add(Flatten())

        model.add(Dense(256))

        model.add(LeakyReLU(alpha=0.1))

        model.add(Dropout(0.5))

        model.add(Dense(10))

        model.add(Activation('softmax'))
        
        self.model = model

        return self.model

    def compile_net(self, model):
        '''
        In this function you are going to compile the model you've created.
        Use model.compile() to build your model.
        '''
        self.model = model

        #TODO: implement this
        # Compile model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model

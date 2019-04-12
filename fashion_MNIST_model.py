# -*- coding: utf-8 -*-
from tensorflow import keras
import tensorflow as tf
import numpy as np

class fashion_MNIST_trained_model:
    def __init__(self):
        print(tf.__version__)
        print(keras.__version__)
        print("keras Model")
        self.__model__ = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
            
        self.__model__.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    
    def train(self, train_images, train_labels):
        self.__model__.fit(train_images, train_labels, epochs=5)
    
    def evaluate(self, test_images, test_labels):
        test_loss, test_acc = self.__model__.evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)
        
    def save(self):
        self.__model__.save('./model./model.keras')
        self.__model__.save_weights('./model./model_wigths.keras')
        
    def load(self):
        self.__model__ = keras.models.load_model('./model./model.keras')       
        
    def predict(self, test_images, test_labels):
        self.load()
                
        predictions = self.__model__.predict(test_images)
        
        print(predictions[55])
        
        print(np.argmax(predictions[55]))
        
        print(test_labels[55])

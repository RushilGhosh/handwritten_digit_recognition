import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#premade dataset that we are loading into the environment
mnist = tf.keras.datasets.mnist

#x data represents the image, y data represents the classification of that image
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#preprocessing data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#initializing the model
model = tf.keras.models.Sequential()

#the image would normally be 28x28 pixels, this function turns each pixel into a 784x1 input
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
#final layer has one neuron per possible digit
model.add(tf.keras.layers.Dense(10, activation="softmax"))


#note: this is inefficient because we are retraining the model every time we run the file but idc
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=3)

#model.save('handwritten.model')

# loss, accuracy = model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from streamlit_drawable_canvas import st_canvas

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

model.fit(x_train, y_train, epochs=5)

# model.save('handwritten.model')

# loss, accuracy = model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)

st.set_page_config(page_title="Digit Identifier", page_icon="🔢", layout="centered")
st.title("Digit Identifier")
st.write("Draw a one-digit number in the white space below and AI will guess what number you drew!")
st.write("Please wait if the page turns dim, the page is loading the model.")
canvas = st_canvas(
    fill_color="000000",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    background_image=None,
    update_streamlit="true",
    height=500,
    width=500,
    drawing_mode="freedraw",
    point_display_radius=0,
    key="canvas"
)

btn=st.button("Predict")

if btn:
    st.empty()
    if canvas.image_data is not None:
        cv2.imwrite("canvas.png", canvas.image_data)
        img=cv2.imread("canvas.png", cv2.IMREAD_GRAYSCALE)
        resized=cv2.resize(img, (28,28)) #model's first layer flattens 28x28 image
        normalized=resized/255.0 #normalizes each pixel's rgb value to be between 0 and 1
        reshaped=np.reshape(normalized, (1,28,28))
        prediction=model.predict(reshaped)
        st.write(f"The number you drew is {np.argmax(prediction)}")

    else:
        st.error("Nothing has been drawn yet!")


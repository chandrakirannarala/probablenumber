import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9
(x_train,y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)  # Normalize each sample
x_test = tf.keras.utils.normalize(x_test, axis=1)  # Normalize each sample

#model = tf.keras.models.Sequential()
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),  # Layer 1
  tf.keras.layers.Dense(128, activation='relu'),  # Layer 2
  tf.keras.layers.Dense(128, activation='relu'),  # Layer 3
  tf.keras.layers.Dense(10, activation='softmax') # Layer 4
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=7)

model.save('handwritten_num_reader.keras')
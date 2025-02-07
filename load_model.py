import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os

model = tf.keras.models.load_model('handwritten_num_reader.model')

# loss, accuracy = model.evaluate(x_test, y_test)

# print(f'Loss: {loss}')
# print(f'Accuracy: {accuracy}')


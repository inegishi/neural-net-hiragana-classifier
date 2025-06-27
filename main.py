import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load images
folder_path = "testimg"
hira_to_num = {"A": 0, "I" :1, "U":2}

X_list = []

#load pre-trained model
model = tf.keras.models.load_model("hiragana.keras")

img2 = Image.open(r"C:\Users\inegi_pqetia\Documents\CS Projects\neural-net-hiragana classifier\testimg\kanaU15.jpg")
img_array = np.array(img2).flatten()
X_list.append(img_array)
X = np.array(X_list)
input_dim = X.shape[0]
prediction = model.predict(X_list[0].reshape(1, 6972))
yhat = np.argmax(tf.nn.softmax(prediction))

print(yhat)
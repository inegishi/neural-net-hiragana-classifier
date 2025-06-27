import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
from pathlib import Path


folder_path = "testimg"



hira_to_num = {"A": 0, "I" :1, "U":2}

#Data Sets
X_list = []
Y_list = []

#iterate thorugh pictures and store vectors in to X and answers into Y
for filename in os.listdir(folder_path):
    full_path = os.path.join(folder_path, filename)
    img = Image.open(full_path).convert("L") 
    img_array = np.array(img).flatten()
    X_list.append(img_array)
    core = filename.replace("kana","").replace(".jpg", "")
    label = hira_to_num["".join(filter(str.isalpha,core))]
    Y_list.append(label)

Y = np.array(Y_list)
X = np.array(X_list)
print(X)
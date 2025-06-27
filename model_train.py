import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load images
hira_to_num = {
    "A": 0, "I": 1, "U": 2, "E": 3, "O": 4,
    "KA": 5, "KI": 6, "KU": 7, "KE": 8, "KO": 9,
    "SA": 10, "SHI": 11, "SU": 12, "SE": 13, "SO": 14,
    "TA": 15, "CHI": 16, "TSU": 17, "TE": 18, "TO": 19,
    "NA": 20, "NI": 21, "NU": 22, "NE": 23, "NO": 24,
    "HA": 25, "HI": 26, "FU": 27, "HE": 28, "HO": 29,
    "MA": 30, "MI": 31, "MU": 32, "ME": 33, "MO": 34,
    "YA": 35, "YU": 36, "YO": 37,
    "RA": 38, "RI": 39, "RU": 40, "RE": 41, "RO": 42,
    "WA": 43, "WO": 44, "N": 45,
    "GA": 46, "GI": 47, "GU": 48, "GE": 49, "GO": 50,
    "ZA": 51, "JI": 52, "ZU": 53, "ZE": 54, "ZO": 55,
    "DA": 56, "DE": 57, "DO": 58,
    "BA": 59, "BI": 60, "BU": 61, "BE": 62, "BO": 63,
    "PA": 64, "PI": 65, "PU": 66, "PE": 67, "PO": 68
}



def get_features():
    """
    Gets features X and Y from image dataset folder
    
    Returns:
        X (np.array): Shape (60,6972) 1-D vectors of images
        Y (np.array): Shape (60,) 
    """

    folder_path = "hiragana_images"
    X_list = []
    Y_list = []
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        img = Image.open(full_path).convert("L") 
        img_array = np.array(img).flatten()
        X_list.append(img_array)
        core = filename.replace("kana","").replace(".jpg", "")
        label = hira_to_num["".join(filter(str.isalpha, core))]
        Y_list.append(label)

    Y = np.array(Y_list)
    X = np.array(X_list)
    return X,Y

def init_model(X,Y):
    """
    Creates model architecture and compile model
    Args: 
        X (np.array): Shape (60,6972) 1-D vectors of images
        Y (np.array): Shape (60,) 

    Returns:
        Model
    """
    model = tf.keras.Sequential([
    tf.keras.layers.Reshape((83, 84, 1), input_shape=(83 * 84,)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(69, activation='softmax')
])
    model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

    return model




def test_run(X,Y,model):
    """
    Trains model, and selects 64 random from image dataset to compare.
    Displays visual comparison graph.

    Args: 
        X (np.array): Shape (60,6972) 1-D vectors of images
        Y (np.array): Shape (60,) 
        model: model

    """
    history = model.fit(X, Y, epochs=40)

    # Visualization and testing
    m = X.shape[0]
    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91])

    for i, ax in enumerate(axes.flat):
        random_index = np.random.randint(m)
        reshaped_img = X[random_index].reshape((84,83))  
        ax.imshow(reshaped_img, cmap='gray')

        prediction = model.predict(X[random_index].reshape(1, X.shape[1]))
        yhat = np.argmax(tf.nn.softmax(prediction))

        ax.set_title(f"{Y[random_index]},{yhat}", fontsize=8)
        ax.set_axis_off()

    fig.suptitle("Label vs Predicted", fontsize=14)
    plt.show()


def save_model(model):
    """
    Save model

    Args:   
        model: logistic regression model
    """
    model.save('hiragana.keras')


X,Y = get_features()
model = init_model(X,Y)
test_run(X,Y,model)
save_model(model)

    
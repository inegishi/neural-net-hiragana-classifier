import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

num_to_hira = {0: "A", 1 :"I", 2: "U"}

# Load model
model = tf.keras.models.load_model("hiragana.keras")

# Canvas size (you draw in 280x280, then resize to 83x84)
CANVAS_SIZE = 280
IMG_WIDTH, IMG_HEIGHT = 83, 84

class HiraganaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw Hiragana")

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack()

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_btn = tk.Button(root, text="Predict", command=self.predict)
        self.predict_btn.pack()

        self.clear_btn = tk.Button(root, text="Clear", command=self.clear)
        self.clear_btn.pack()

        self.result_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.result_label.pack()

    def paint(self, event):
        x, y = event.x, event.y
        r = 5
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def predict(self):
        # Resize to model input size
        img_resized = ImageOps.invert(self.image.resize((IMG_WIDTH, IMG_HEIGHT)))
        img_array = np.array(img_resized).flatten()
        img_array = img_array.reshape(1, 6972)
        img_resized.save("test.png")
        prediction = model.predict(img_array)
        yhat = np.argmax(tf.nn.softmax(prediction))
        self.result_label.config(text=f"Predicted Class: {num_to_hira[yhat]}")

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)
        self.result_label.config(text="")

# Launch
if __name__ == "__main__":
    root = tk.Tk()
    app = HiraganaApp(root)
    root.mainloop()

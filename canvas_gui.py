import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

num_to_hira = {
    0: "あ", 1: "い", 2: "う", 3: "え", 4: "お",
    5: "か", 6: "き", 7: "く", 8: "け", 9: "こ",
    10: "さ", 11: "し", 12: "す", 13: "せ", 14: "そ",
    15: "た", 16: "ち", 17: "つ", 18: "て", 19: "と",
    20: "な", 21: "に", 22: "ぬ", 23: "ね", 24: "の",
    25: "は", 26: "ひ", 27: "ふ", 28: "へ", 29: "ほ",
    30: "ま", 31: "み", 32: "む", 33: "め", 34: "も",
    35: "や", 36: "ゆ", 37: "よ",
    38: "ら", 39: "り", 40: "る", 41: "れ", 42: "ろ",
    43: "わ", 44: "を", 45: "ん",
    46: "が", 47: "ぎ", 48: "ぐ", 49: "げ", 50: "ご",
    51: "ざ", 52: "じ", 53: "ず", 54: "ぜ", 55: "ぞ",
    56: "だ", 57: "で", 58: "ど",
    59: "ば", 60: "び", 61: "ぶ", 62: "べ", 63: "ぼ",
    64: "ぱ", 65: "ぴ", 66: "ぷ", 67: "ぺ", 68: "ぽ"
}

# Load model
model = tf.keras.models.load_model("hiragana.keras")

CANVAS_SIZE = 180
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
        r = 2
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

# ====================================================================================
#  Author: Kunal SK Sukhija
# ====================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import missingno as msno
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
#%%
import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw
#%%
model = tf.keras.models.load_model('Digit_model.h5')
#%%
window = tk.Tk()
window.title("Handwritten Digit Recognition")
#%%
canvas = Canvas(window, width=280, height=280, bg="white")
canvas.grid(row=0, column=0, columnspan=2)
#%%
label = Label(window, text="Predicted Digit: ")
label.grid(row=1, column=0, columnspan=2)
#%%
image = Image.new("L", (280, 280),0)
draw = ImageDraw.Draw(image)
#%%
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="white")

#%%
def recognize_digit():
    # Resize the image to 28x28 and convert to a numpy array
    digit_image = image.resize((28, 28))
    digit_image=np.expand_dims(digit_image, 0)
    digit_array = np.array(digit_image).astype('float32') / 255.0
    digit_array = digit_array.reshape((1, 28, 28, 1))


    prediction = model.predict(digit_array)

    predicted_digit = np.argmax(prediction)

    label.config(text="Predicted Digit: " + str(predicted_digit))
#%%
def draw_digit(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
    draw.line([x1, y1, x2, y2], fill="black", width=10)
#%%
canvas.bind("<B1-Motion>", draw_digit)
#%%
recognize_button = Button(window, text="Recognize Digit", command=recognize_digit)
recognize_button.grid(row=2, column=0)

clear_button = Button(window, text="Clear Canvas", command=clear_canvas)
clear_button.grid(row=2, column=1)
#%%
window.mainloop()


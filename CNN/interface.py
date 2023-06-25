import cv2
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

import tensorflow as tf

# Funcție pentru a deschide o imagine
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png")])
    if file_path:
        image = cv2.imread(file_path)
        predict_image(image)

# Funcție pentru a prezice imaginea și a afișa rezultatele
def predict_image(image):
    img_from_ar = Image.fromarray(image, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    test_image = np.expand_dims(resized_image, axis=0)

    # Încarcă modelul
    model = tf.keras.models.load_model('model.h5')

    result = model.predict(test_image)
    prediction = np.argmax(result)

    messagebox.showinfo("Prediction", f"The prediction is: {prediction}")

# Creează o fereastră
window = tk.Tk()

# Buton pentru a deschide o imagine
open_button = tk.Button(window, text="Open Image", command=open_image)
open_button.pack()

window.mainloop()

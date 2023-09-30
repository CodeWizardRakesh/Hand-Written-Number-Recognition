import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import os.path
import random as rd

model = tf.keras.models.load_model('HR#3_neural.model')

def predict_number(image_path):
    img = cv2.imread(image_path)
    try:
        img = cv2.resize(img, (28, 28))
        img2 = np.invert(img)
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #cov img to grayscale image 0-255
        new_img = tf.keras.utils.normalize(gray, axis=1) #0-1
        new_img = np.array(new_img).reshape(-1, 28, 28, 1)

    except:
        img2 = np.invert(img)
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # print(Image.getdata(gray))
        new_img = tf.keras.utils.normalize(gray, axis=1)
        new_img = np.array(new_img).reshape(-1, 28, 28, 1)

    prediction = model.predict(new_img)
    return np.argmax(prediction)

def upload_image():
    # Show the file dialog box
    file_path = filedialog.askopenfilename()

    # Load the image and display it
    img = Image.open(file_path)
    img = img.resize((170,170))
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img

    # Predict the number
    number = predict_number(file_path)
    L = ['the number is probably ' + str(number), 'the number is ' + str(number) + ' am i right?', 'My prediction is ' + str(number),'It is ' + str(number) + '...', 'i see ' + str(number)]
    n = rd.randint(0, len(L) - 1)
    #print(L[n])
    result_label.configure(text=L[n])


root = tk.Tk()
root.title("Handwritten Number Recognition")
root.geometry("400x400")
root.configure(bg="#36454F") # #7f8c8d


frame = tk.Frame(root, bg="#15317E", width=300, height=850) #3498db #2916F5
frame.pack(expand=True, padx=50, pady=50)


panel = tk.Label(frame, )
panel.pack(padx=10, pady=10)

# Create the button to upload the image
button = tk.Button(frame, text="Upload Image", font=10,bg="#FF00FF", command=upload_image)
button.pack(padx=10, pady=10)

# Create the label to display the result

result_label = tk.Label(frame, text="Predicted number: ", bg="#15317E",fg="white", font=10)
result_label.pack(padx=10, pady=10)

# Start the main event loop
root.mainloop()

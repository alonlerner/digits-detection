import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

models = []
for i in range(10):
  models.append(keras.models.load_model(f"./models/model{i}.h5"))

img_list = os.listdir('./test-images')
for i in range(len(img_list)):
    if img_list[i].endswith(".png"):
        image = cv2.imread(f'./test-images/{img_list[i]}')
        grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        preprocessed_digits = []
        print(len(contours))
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)

            # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
            # cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
            
            # Cropping out the digit from the image corresponding to the current contours in the for loop
            digit = thresh[y:y+h, x:x+w]

            if w < h:
                digit = np.pad(digit, ((0,0),(int((h-w)/2),int((h-w)/2))), "constant", constant_values=0)
            else:
                digit = np.pad(digit, ((int((w-h)/2),int((w-h)/2)),(0,0)), "constant", constant_values=0)
            
            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(digit, (18,18))
            
            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
            padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
            
            # Adding the preprocessed digit to the list of preprocessed digits
            preprocessed_digits.append(padded_digit)

            max_pred = 0
            max_model = -1
            for i in range(10):
                prediction = models[i].predict(padded_digit.reshape(1, 28, 28, 1))
                if prediction[0][1] > max_pred:
                    max_pred, max_model = prediction[0][1], i
            if max_pred > 0.8:
                cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
                cv2.putText(image, str(max_model), (x, y), cv2.FONT_HERSHEY_PLAIN, 5, 0, 3)

        plt.imshow(image, cmap="gray")
        plt.show()
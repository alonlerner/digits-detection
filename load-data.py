import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# load MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()

# normalize the pixel intensities so it will be in 0-1 range
X_train, X_test = X_train / 255.0, X_test / 255.0

r = np.random.randint(0,len(X_train)-1) # select random record
plt.imshow(X_train[r], cmap='gray') # Import the image
plt.show() # Plot the image
print(f'image classification: {y_train[r]}')

with open('data.npy', 'wb') as f:
    np.save(f, X_train)
    np.save(f, y_train)
    np.save(f, X_test)
    np.save(f, y_test)
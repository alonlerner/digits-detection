import tensorflow as tf
from tensorflow import keras
import numpy as np
import copy

# load data
with open('data.npy', 'rb') as f:
    X_train = np.load(f)
    y_train = np.load(f)
    X_test = np.load(f)
    y_test = np.load(f)

evals = []

# create a model for each digit
for i in range(10):
    y_train_temp = copy.deepcopy(y_train)
    y_test_temp = copy.deepcopy(y_test)
    # set y as 1 if the image is as the class of the model, otherwise 0
    for j in range(len(y_train_temp)):
        y_train_temp[j] = 1 if y_train_temp[j] == i else 0
    for j in range(len(y_test_temp)):
        y_test_temp[j] = 1 if y_test_temp[j] == i else 0
    # build the neural network
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(2, activation="softmax"))

    # use compile method to specify the loss function and the optimizer. loss function is set to sparse categorical crossentropy, the optimizer is set to adam, and evaluation it will measure the accuracy.
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train the model using 5 epochs and return add the evaluation to the evals array
    evals.append(model.fit(X_train, y_train_temp, epochs=5, batch_size=32, validation_data=(X_test, y_test_temp)))
    model.save(f'./models/model{i}.h5')

# save evaluations into npy file
with open('evals.npy', 'wb') as f:
    np.save(f, evals)

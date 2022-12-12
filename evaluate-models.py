import tensorflow as tf
from tensorflow import keras
import numpy as np

# load evaluation data
with open('evals.npy', 'rb') as f:
    evals = np.load(f, allow_pickle=True)

# print evaluation data
print('evaluations on test data:')
print('model\tloss\t\t\taccuracy')
for i in range(len(evals)):
    print(f'{i}\t{evals[i].history["val_loss"][-1]}\t{evals[i].history["val_accuracy"][-1]}')

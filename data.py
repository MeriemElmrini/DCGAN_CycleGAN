import numpy as np
import cv2
from keras.datasets import mnist
import h5py

(X_imgs, _) , (_, _) = mnist.load_data()
path = 'dataset/usps-dataset/usps.h5'
with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        Y = train.get('data')[:]

# Mnist Dataset
# Rescale -1 to 1
X_imgs = X_imgs / (255 / 2) - 1.
# Reshape (nb_imgs, nb_rows, nb_cols, 1)
X_imgs = np.expand_dims(X_imgs, axis=3)

# USPS Dataset
# Reshape (nb_imgs, nb_rows, nb_cols)
Y = Y.reshape(Y.shape[0], np.sqrt(Y.shape[1]).astype(int),
        np.sqrt(Y.shape[1]).astype(int))

# Resize to (nb_imgs, 28, 28) (so that a mnist img and an usps img have the same dimensions)
Y_imgs = np.zeros(shape=(Y.shape[0], 28, 28))
for i in range(len(Y_imgs)):
    Y_imgs[i] = cv2.resize(Y[i], dsize=(28, 28))
# Rescale -1 to 1
Y_imgs = Y_imgs / (255 / 2) - 1.
# Reshape (nb_imgs, nb_rows, nb_cols, 1)
Y_imgs = np.expand_dims(Y_imgs, axis=3)

from tensorflow.keras.datasets import mnist
from typing import Tuple
import numpy as np

def load_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

	# Load the dataset from keras dataset
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0

	# Flatten the input data
	x_train: np.ndarray = x_train.reshape(x_train.shape[0], -1)  # (60000, 784)
	x_test: np.ndarray = x_test.reshape(x_test.shape[0], -1)     # (10000, 784)
	return (x_train, y_train), (x_test, y_test)

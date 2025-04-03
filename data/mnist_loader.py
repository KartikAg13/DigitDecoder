import tensorflow as tf
from tensorflow.keras.datasets import mnist
from typing import Tuple
import numpy as np

def load_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
	(x_train, y_train), (x_test, y_test) = load_mnist()
	print("Training Data Shape: ", x_train.shape)
	print("Testing Data Shape: ", x_test.shape)
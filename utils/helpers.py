import numpy as np
import matplotlib.pyplot as plt
from model.mlp import NeuralNetwork

def plot_predictions(model: NeuralNetwork, x_test: np.ndarray, y_test: np.ndarray, number_of_samples: int):
	x_test_images: np.ndarray = x_test.reshape(-1, 28, 28)
	predictions: np.ndarray = model.predict(x_test)
	
	_, axes = plt.subplots(1, number_of_samples, figsize=(10, 3))
	for i in range(number_of_samples):
		axis = axes[i]
		axis.imshow(x_test_images[i], cmap='gray')
		axis.set_title(f"Predicted: {predictions[i]}\nTrue: {y_test[i]}")
		axis.axis('off')
	plt.show()
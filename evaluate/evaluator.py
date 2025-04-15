import numpy as np
from model.mlp import NeuralNetwork

def evaluate_model(model: NeuralNetwork, x: np.ndarray, y: np.ndarray) -> float:
	predictions: np.ndarray = model.predict(x)
	accuracy: float = np.mean(predictions == y)
	return accuracy
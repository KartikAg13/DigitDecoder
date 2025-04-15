from model.mlp import NeuralNetwork
import numpy as np

def loss_function(y_onehot: np.ndarray, A2: np.ndarray) -> float:
	return -np.mean(np.sum(y_onehot * np.log(A2 + 1e-10), axis=1))

def train_model(model: NeuralNetwork, epochs: int, learning_rate: float, x_train: np.ndarray, y_train: np.ndarray):
	y_onehot: np.ndarray = np.eye(10)[y_train]
	for epoch in range(epochs):
		A2: np.ndarray = model.forward_propagation(x_train)
		loss: float = loss_function(y_onehot, A2)
		model.backward_propagation(x_train, y_onehot)
		model.update_parameters(learning_rate)
		if epoch % 50 == 0:
			print(f"Epoch: {epoch}, Loss: {loss}")
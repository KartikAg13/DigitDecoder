from model.mlp import NeuralNetwork
import numpy as np

def loss_function(y_onehot, A2) -> float:
	return -np.mean(np.sum(y_onehot * np.log(A2 + 1e-10), axis=1))

def train_model(model: NeuralNetwork, epochs: int, learning_rate: float, x_train, y_train):
	y_onehot = np.eye(10)[y_train]
	for epoch in range(epochs):
		A2 = model.forward_propogation(x_train)
		
		loss: float = loss_function(y_onehot, A2)

		model.backward_propogation(x_train, y_onehot)
		model.update_parameters(learning_rate)

		if epoch % 50 == 0:
			print(f"Epoch: {epoch}, Loss: {loss}")
		
	predictions = model.predict(x_train)

	return predictions
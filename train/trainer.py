from model.mlp import NeuralNetwork
import numpy as np

def loss_function(y_onehot: np.ndarray, A2: np.ndarray) -> float:
	return -np.mean(np.sum(y_onehot * np.log(A2 + 1e-10), axis=1))

def train_model(model: NeuralNetwork, epochs: int, learning_rate: float, 
				x_train: np.ndarray, y_train: np.ndarray, batch_size: int, 
				number_of_samples: int, epsilon: float):
	y_onehot: np.ndarray = np.eye(10)[y_train]
	
	for epoch in range(epochs):
		# Shuffle the data
		indices: np.ndarray = np.random.permutation(number_of_samples)
		x_train_shuffled: np.ndarray = x_train[indices]
		y_onehot_shuffled: np.ndarray = y_onehot[indices]
		
		# Process mini-batches
		for i in range(0, number_of_samples, batch_size):
			x_batch: np.ndarray = x_train_shuffled[i:i + batch_size]
			y_batch: np.ndarray = y_onehot_shuffled[i:i + batch_size]
			
			A2: np.ndarray = model.forward_propagation(x_batch)
			model.backward_propagation(x_batch, y_batch)
			model.update_parameters(learning_rate)
		
		# Compute and print loss on the full dataset
		A2_full: np.ndarray = model.forward_propagation(x_train)
		# loss: float = loss_function(y_onehot, A2_full)
		current_loss: float = loss_function(y_onehot, A2_full)
		if current_loss > epsilon:
			loss = current_loss
		else:
			break
		if epoch % 50 == 0:
			print(f"Epoch: {epoch}, Loss: {loss}")
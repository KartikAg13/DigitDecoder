import numpy as np
class NeuralNetwork:
	def __init__(self, input_size, hidden_size, output_size):
		self.W1 = np.random.randn(input_size, hidden_size) * 0.01
		self.b1 = np.zeros(hidden_size)
		self.W2 = np.random.randn(hidden_size, output_size) * 0.01
		self.b2 = np.zeros(output_size)
	
	# Sigmoid activation function
	def sigmoid(self, z):
		return (1 / (1 + np.exp(-z)))
	
	# Softmax activation function for multi-class probabilities
	def softmax(self, z):
		exp_z: float = np.exp(z - np.max(z, axis=1, keepdims=True))
		return (exp_z / np.sum(exp_z, axis=1, keepdims=True))
	
	# Implementation of forward propogation
	def forward_propogation(self, x_train):
		# Layer 1
		self.Z1 = x_train.dot(self.W1) + self.b1
		self.A1 = self.sigmoid(self.Z1)
		
		# Layer 2
		self.Z2 = self.A1.dot(self.W2) + self.b2
		self.A2 = self.softmax(self.Z2)

		return self.A2
	
	def backward_propogation(self, x_train, y_train):
		length: int = x_train.shape[0]		# 60000

		dZ2 = self.A2 - y_train
		
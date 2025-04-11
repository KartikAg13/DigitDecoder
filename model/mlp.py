import numpy as np

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.W1: np.ndarray = np.random.randn(input_size, hidden_size) * 0.01
        self.b1: np.ndarray = np.zeros(hidden_size)
        self.W2: np.ndarray = np.random.randn(hidden_size, output_size) * 0.01
        self.b2: np.ndarray = np.zeros(output_size)
    
    # Sigmoid activation function
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    # Softmax activation function for multi-class probabilities
    def softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z: np.ndarray = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # Implementation of forward propagation
    def forward_propagation(self, x_train: np.ndarray) -> np.ndarray:
        # Layer 1
        self.Z1: np.ndarray = x_train.dot(self.W1) + self.b1
        self.A1: np.ndarray = self.sigmoid(self.Z1)
        # Layer 2
        self.Z2: np.ndarray = self.A1.dot(self.W2) + self.b2
        self.A2: np.ndarray = self.softmax(self.Z2)
        return self.A2
    
    # Implementation of backpropagation using Stochastic Gradient Descent
    def backward_propagation(self, x_train: np.ndarray, y_train: np.ndarray):
        length: int = x_train.shape[0]  # 60000

        dZ2: np.ndarray = self.A2 - y_train
        self.dW2: np.ndarray = (self.A1.T.dot(dZ2)) / length
        self.db2: np.ndarray = (np.sum(dZ2, axis=0)) / length

        dA1: np.ndarray = dZ2.dot(self.W2.T)
        dZ1: np.ndarray = dA1 * self.A1 * (1 - self.A1)
        self.dW1: np.ndarray = (x_train.T.dot(dZ1)) / length
        self.db1: np.ndarray = (np.sum(dZ1, axis=0)) / length

    # Updating the weights and biases
    def update_parameters(self, learning_rate: float):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    # Make the predictions for the training data
    def predict(self, x_train: np.ndarray) -> np.ndarray:
        A2: np.ndarray = self.forward_propagation(x_train)
        return np.argmax(A2, axis=1)
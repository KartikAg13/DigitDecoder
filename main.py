from data.mnist_loader import load_mnist
from evaluate.evaluator import evaluate_model
from model.mlp import NeuralNetwork
from train.trainer import train_model
from utils.helpers import plot_predictions

def main():
	# Load and preprocess the data
	(x_train, y_train), (x_test, y_test) = load_mnist()
	
	# Print the size of the data
	train_shape = x_train.shape		# (60000, 28, 28)
	test_shape = x_test.shape		# (10000, 28, 28)
	print("Training Data Shape: ", train_shape)
	print("Testing Data Shape: ", test_shape)

	# Create an instance of the model
	input_size: int = 784	# 28 * 28
	hidden_size: int = 128
	output_size: int = 10
	model: NeuralNetwork = NeuralNetwork(input_size, hidden_size, output_size)

	# Train the model
	epochs: int = 1000
	learning_rate: float = 0.01
	predictions = train_model(model, epochs, learning_rate, x_train, y_train)

if __name__ == "__main__":
	main()
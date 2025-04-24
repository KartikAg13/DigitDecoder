from data.mnist_loader import load_mnist
from evaluate.evaluator import evaluate_model
from model.mlp import NeuralNetwork
from train.trainer import train_model
from utils.helpers import plot_predictions

def main():
	# Load and preprocess the data
	(x_train, y_train), (x_test, y_test) = load_mnist()

	# Print the size of the data
	train_shape: tuple = x_train.shape
	test_shape: tuple = x_test.shape
	print("Training Data Shape: ", train_shape)  # (60000, 784)
	print("Testing Data Shape: ", test_shape)    # (10000, 784)

	# Create an instance of the model
	input_size: int = 784
	hidden_size: int = 128
	output_size: int = 10
	model: NeuralNetwork = NeuralNetwork(input_size, hidden_size, output_size)

	# Train the model
	epochs: int = 1000
	learning_rate: float = 0.01
	train_model(model, epochs, learning_rate, x_train, y_train)

	train_accuracy: float = evaluate_model(model, x_train, y_train)
	print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

	test_accuracy: float = evaluate_model(model, x_test, y_test)
	print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

	number_of_samples: int = 10
	plot_predictions(model, x_test, y_test, number_of_samples)

if __name__ == "__main__":
	main()
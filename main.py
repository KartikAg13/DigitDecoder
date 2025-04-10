from data.mnist_loader import load_mnist
from evaluate.evaluator import evaluate_model
from model.mlp import NeuralNetwork
from train.trainer import train_model
from utils.helpers import plot_predictions

def main():
	# Load and preprocess the data
	(x_train, y_train), (x_test, y_test) = load_mnist()
	
	train_shape = x_train.shape		# (60000, 28, 28)
	test_shape = x_test.shape		# (10000, 28, 28)

	print("Training Data Shape: ", x_train.shape)
	print("Testing Data Shape: ", x_test.shape)

if __name__ == "__main__":
	main()
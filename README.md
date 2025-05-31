# DigitDecoder

A neural network implementation for handwritten digit recognition using the MNIST dataset. This project implements a Multi-Layer Perceptron (MLP) from scratch using NumPy for educational purposes.

## 🎯 Overview

DigitDecoder is a complete machine learning pipeline that:
- Loads and preprocesses the MNIST handwritten digit dataset
- Implements a 2-layer neural network with ReLU and Softmax activations
- Trains the model using mini-batch gradient descent
- Evaluates model performance on test data
- Visualizes predictions with sample images
- Saves trained model weights for future use

## 🏗️ Architecture

The neural network consists of:
- **Input Layer**: 784 neurons (28×28 flattened pixel values)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (for 10 digit classes)

## 📁 Project Structure

```
DigitDecoder/
├── data/
│   ├── __init__.py
│   └── mnist_loader.py      # MNIST data loading and preprocessing
├── model/
│   ├── __init__.py
│   └── mlp.py              # Neural network implementation
├── train/
│   ├── __init__.py
│   └── trainer.py          # Training logic and loss function
├── evaluate/
│   ├── __init__.py
│   └── evaluator.py        # Model evaluation utilities
├── utils/
│   ├── __init__.py
│   └── helpers.py          # Visualization and helper functions
├── weights/                # Directory for saved model weights
├── main.py                 # Main execution script
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy tensorflow matplotlib
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/KartikAg13/DigitDecoder.git
cd DigitDecoder
```

2. Create weights directory:
```bash
mkdir weights
```

### Usage

Run the complete training and evaluation pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess the MNIST dataset
2. Create and train the neural network
3. Display training and test accuracy
4. Generate prediction visualizations
5. Save trained weights to the `weights/` directory

## 🔧 Configuration

You can modify the following hyperparameters in `main.py`:

- `hidden_size`: Number of neurons in the hidden layer (default: 128)
- `epochs`: Maximum number of training epochs (default: 1000)
- `learning_rate`: Learning rate for gradient descent (default: 0.1)
- `batch_size`: Mini-batch size for training (default: 32)
- `epsilon`: Early stopping threshold (default: 1e-4)

## 📊 Model Performance

The model typically achieves:
- Training Accuracy: ~98%
- Test Accuracy: ~97%

Training includes early stopping when the loss falls below the epsilon threshold.

## 🖼️ Visualizations

The project generates `predictions.png` showing sample test images with their predicted and true labels for visual verification of model performance.

## 🧠 Implementation Details

### Forward Propagation
- Linear transformation followed by ReLU activation in hidden layer
- Linear transformation followed by Softmax activation in output layer

### Backward Propagation
- Computes gradients using chain rule
- Updates weights and biases using gradient descent

### Training Features
- Mini-batch gradient descent
- Data shuffling each epoch
- Cross-entropy loss function
- Early stopping mechanism

## 📈 Key Features

- **Pure NumPy Implementation**: Educational focus on understanding neural networks
- **Modular Design**: Clean separation of concerns across modules
- **Comprehensive Pipeline**: From data loading to model evaluation
- **Visualization**: Visual feedback on model predictions
- **Model Persistence**: Save/load trained weights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MNIST dataset provided by Yann LeCun and Corinna Cortes
- TensorFlow/Keras for convenient dataset loading
- NumPy for numerical computations

## 🔬 Educational Purpose

This implementation is designed for learning and understanding the fundamentals of neural networks. For production use, consider using established frameworks like TensorFlow or PyTorch which offer optimized implementations and additional features.

# DigitDecoder

DigitDecoder is a simple Multi-Layer Perceptron (MLP) implemented from scratch using NumPy to classify handwritten digits from the MNIST dataset. The project demonstrates the fundamentals of neural networks, including forward and backward propagation, without relying on high-level machine learning libraries for the model itself. TensorFlow is used solely for loading the MNIST dataset.

## Features
- MLP with one hidden layer, implemented from scratch using NumPy.
- Sigmoid activation for the hidden layer and softmax for the output layer.
- Training and evaluation scripts with accuracy metrics.
- Visualization of predictions using Matplotlib.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/DigitDecoder.git
   cd DigitDecoder
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run the main script**:
   ```bash
   python main.py
   ```

2. **Expected output**:
   - Training progress with loss printed every 50 epochs.
   - Final training and test accuracy.
   - A plot showing 10 sample predictions with true labels.

## Project Structure
- `mnist_loader.py`: Loads and preprocesses the MNIST dataset.
- `mlp.py`: Defines the `NeuralNetwork` class with forward and backward propagation.
- `trainer.py`: Contains the training loop and loss function.
- `evaluator.py`: Evaluates the model's accuracy.
- `helpers.py`: Provides utility functions for plotting predictions.
- `main.py`: Orchestrates data loading, model training, evaluation, and visualization.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License.
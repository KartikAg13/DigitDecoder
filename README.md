# DigitDecoder

Explanation of Each Component

    data/mnist_loader.py
        Purpose: Loads and preprocesses the MNIST dataset (e.g., normalizing pixel values, splitting into training and test sets).
        Why it’s separate: Keeps data-related logic isolated, making it easy to modify if you change datasets or preprocessing steps.
    model/mlp.py
        Purpose: Defines the Multi-Layer Perceptron (MLP) class, including the architecture (input layer, hidden layers, output layer), activation functions (e.g., sigmoid), and methods for forward and backward propagation.
        Why it’s separate: Isolates the neural network’s structure, allowing you to tweak the model without affecting other parts.
    train/trainer.py
        Purpose: Implements the training loop—forward propagation, loss calculation (e.g., cross-entropy), backpropagation, and weight updates. It can also log progress (e.g., loss per epoch).
        Why it’s separate: Separates training logic from the model and evaluation, keeping the code modular and readable.
    evaluate/evaluator.py
        Purpose: Evaluates the trained model on the test set, calculating metrics like accuracy and optionally visualizing results (e.g., predicted digits).
        Why it’s separate: Keeps evaluation distinct, making it easier to test and debug.
    utils/helpers.py
        Purpose: Contains utility functions, such as plotting sample digits with predictions or saving/loading model weights.
        Why it’s separate: Organizes reusable helper code, reducing clutter in other modules.
    main.py
        Purpose: The entry point of the project. It ties everything together—loads data, initializes the model, trains it, and evaluates it.
        Why it’s separate: Provides a clear, high-level script to run the entire project.
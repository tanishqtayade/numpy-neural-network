Custom Neural Network Framework (NumPy)
A modular, lightweight deep learning framework built from scratch using pure Python and NumPy. No TensorFlow, no PyTorch—just linear algebra and backpropagation.

This project was built to understand the mathematical foundations of AI, specifically focusing on how neural networks learn, optimize, and generalize on complex data.

🚀 Key Features
Modular Architecture: Build networks layer by layer (Dense, ReLU, Softmax).

Backpropagation Engine: Fully manual implementation of the chain rule.

Adam Optimizer: Adaptive learning rate implementation for faster convergence.

Mini-Batch Training: Efficient data handling to process large datasets without memory bottlenecks.



🧠 Core Components
1. network.py (The Engine)
This is the heart of the framework. It acts as a container for your layers and manages the flow of data.

Forward Pass: Iterates through layers, applying layer.forward() to calculate the activation at each step.

Backward Pass: Calculates the gradients by reversing through the layers. It propagates the error from the output layer back to the input, updating weights based on the loss derivative.

Modularity: By decoupling the layers, you can swap a 2-layer network for a 5-layer network just by changing the add() sequence.

2. train_mnist.py (The Training Pipeline)
This script orchestrates the learning process. It handles:
Data Preprocessing: Converts binary IDX files to NumPy tensors and normalizes pixel intensities from $[0, 255]$ to $[0, 1]$.
Shuffling: Shuffles the dataset at the start of every epoch to ensure the network doesn't memorize the order of the images.
Mini-Batching: Rather than calculating one massive gradient for 60,000 images, it slices the data into batches of 128, providing 468 weight updates per epoch. This is why the loss drops so quickly and smoothly.

💡 What I Learned
Building this forced me to move beyond "importing" libraries. I had to solve dimension mismatches, debug gradient vanishing/explosion, and understand how matrix multiplication forms the basis of all modern artificial intelligence.

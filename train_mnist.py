import numpy as np
import gzip
import os

# Import your custom framework
from core.network import Network
from core.layers import Dense
from core.activations import ReLU, Softmax
from core.losses import categorical_cross_entropy, categorical_cross_entropy_derivative
from core.optimizers import Adam

def load_mnist():
    """Robustly loads MNIST using absolute paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    
    # Load images and labels
    with gzip.open(os.path.join(data_dir, files[0]), "rb") as f: 
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    with gzip.open(os.path.join(data_dir, files[1]), "rb") as f: 
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(os.path.join(data_dir, files[2]), "rb") as f: 
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    with gzip.open(os.path.join(data_dir, files[3]), "rb") as f: 
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)
        
    return x_train, y_train, x_test, y_test

# 1. Load the FULL dataset
print("Loading all 60,000 MNIST images...")
x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_mnist()

# Normalize pixel values to [0, 1]
x_train = x_train_raw.astype(np.float32) / 255.0
x_test = x_test_raw.astype(np.float32) / 255.0

# One-hot encode training labels
y_train = np.zeros((y_train_raw.size, 10))
y_train[np.arange(y_train_raw.size), y_train_raw] = 1

# 2. Build the Maximized Deep Network Architecture
print("Building Deep Neural Network (2 Hidden Layers)...")
net = Network()

# Hidden Layer 1: Captures broad spatial strokes (784 -> 256)
net.add(Dense(784, 256))
net.add(ReLU())

# Hidden Layer 2: Combines strokes into digit sub-features (256 -> 128)
net.add(Dense(256, 128))
net.add(ReLU())

# Output Layer: Maps features to 10 distinct digit probabilities (128 -> 10)
net.add(Dense(128, 10))
net.add(Softmax())

# Configure Loss and Hyperparameters
net.use_loss(categorical_cross_entropy, categorical_cross_entropy_derivative)
optimizer = Adam(lr=0.001)  # 0.001 is the sweet spot for Adam with deeper networks

# 3. Optimized Mini-Batch Training Loop
epochs = 20  
batch_size = 128
num_samples = x_train.shape[0]

print(f"Starting Training with Mini-Batches (Size: {batch_size}) for {epochs} Epochs...")
for epoch in range(epochs):
    # Shuffle entire dataset at the start of every epoch to break ordering bias
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    
    epoch_loss = 0
    num_batches = int(np.ceil(num_samples / batch_size))
    
    for i in range(0, num_samples, batch_size):
        x_batch = x_train_shuffled[i : i + batch_size]
        y_batch = y_train_shuffled[i : i + batch_size]
        
        # Train on current mini-batch
        error = net.fit(x_batch, y_batch, epochs=1, optimizer=optimizer)
        
        if error is not None:
            epoch_loss += error
            
    # Print status updates
    if error is not None:
        average_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1:02d}/{epochs} | System Average Loss: {average_loss:.6f}")
    else:
        print(f"Epoch {epoch+1:02d}/{epochs} processing completed successfully.")

# 4. Evaluate on Full Test Set
print("\nEvaluating Model on Unseen Test Data...")
preds = np.argmax(net.predict(x_test), axis=1)
accuracy = np.mean(preds == y_test_raw) * 100
print(f"Final Optimized Test Accuracy: {accuracy:.2f}%")

# 5. Save the trained model to disk
import pickle
print("\nSaving your trained model weights...")
with open("mnist_model.pkl", "wb") as f:
    pickle.dump(net, f)
print("Saved successfully as 'mnist_model.pkl'!")
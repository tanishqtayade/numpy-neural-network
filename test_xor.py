import numpy as np
from core.network import Network
from core.layers import Dense
from core.activations import ReLU, Sigmoid
from core.losses import mse, mse_derivative
from core.optimizers import Adam

# 1. The XOR Data
# X = Inputs, Y = Expected Outputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# 2. Build the Network
net = Network()
net.add(Dense(2, 10))  # Input layer: 2 features. Hidden layer: 10 neurons.
net.add(ReLU())        # Non-linearity
net.add(Dense(10, 1))  # Hidden layer -> Output layer (1 neuron).
net.add(Sigmoid())     # Squish output to a probability between 0 and 1.

# 3. Compile and Train
net.use_loss(mse, mse_derivative)

print("Starting training...")
# Notice how we pass the Adam optimizer instance now, and we only need 500 epochs!
optimizer = Adam(lr=0.05)
net.fit(X, Y, epochs=500, optimizer=optimizer)

# 4. Test the Predictions
print("\nFinal Predictions:")
predictions = net.predict(X)

for x, y_true, y_pred in zip(X, Y, predictions):
    print(f"Input: {x} | Target: {y_true[0]} | Prediction: {y_pred[0]:.4f}")
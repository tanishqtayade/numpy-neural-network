import numpy as np

# --- Mean Squared Error (For Regression) ---

def mse(y_true, y_pred):
    """Computes the Mean Squared Error."""
    return np.mean(np.power(y_true - y_pred, 2))

def mse_derivative(y_true, y_pred):
    """
    Computes the derivative of MSE with respect to the predictions.
    This is the very first gradient that gets passed into the network's backward pass.
    """
    return 2 * (y_pred - y_true) / y_true.size

# --- Binary Cross-Entropy (For Binary Classification) ---

def binary_cross_entropy(y_true, y_pred):
    """Computes Binary Cross-Entropy."""
    # We clip predictions slightly to avoid log(0) which causes math domain errors (NaNs)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

def binary_cross_entropy_derivative(y_true, y_pred):
    """Computes the derivative of BCE."""
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # The calculus derivative of the BCE formula
    return ((1 - y_true) / (1 - y_pred_clipped) - y_true / y_pred_clipped) / y_true.size

# --- Categorical Cross-Entropy (For Multi-Class Classification) ---

def categorical_cross_entropy(y_true, y_pred):
    """Computes Categorical Cross-Entropy."""
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # Average the loss across the entire batch
    return -np.sum(y_true * np.log(y_pred_clipped)) / len(y_true)

def categorical_cross_entropy_derivative(y_true, y_pred):
    """Computes the derivative of CCE."""
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # The gradient of CCE, averaged over the batch size
    return -y_true / y_pred_clipped / len(y_true)
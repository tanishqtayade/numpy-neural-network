import numpy as np
from core.layers import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data
        # f(x) = max(0, x)
        self.output = np.maximum(0, self.input)
        return self.output

    def backward(self, output_error):
        # The derivative of ReLU is 1 if input > 0, otherwise 0
        relu_derivative = self.input > 0
        
        # Multiply incoming error by the derivative (chain rule)
        return output_error * relu_derivative

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data
        # f(x) = 1 / (1 + e^-x)
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backward(self, output_error):
        # Derivative of sigmoid is: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_derivative = self.output * (1 - self.output)
        
        # Chain rule
        return output_error * sigmoid_derivative
    
class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data
        # Subtracting the max for numerical stability (prevents np.exp from overflowing)
        shifted_input = self.input - np.max(self.input, axis=1, keepdims=True)
        exp_values = np.exp(shifted_input)
        
        # Normalize to get a probability distribution
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, output_error):
        # Create an empty array to store the gradients
        input_error = np.empty_like(output_error)
        
        # We must calculate the Jacobian matrix for each sample in the batch
        for i, (single_output, single_error) in enumerate(zip(self.output, output_error)):
            # Flatten the output to a 1D column vector
            single_output = single_output.reshape(-1, 1)
            
            # Calculate Jacobian matrix: Softmax derivative formula
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Multiply by the incoming error
            input_error[i] = np.dot(jacobian_matrix, single_error)
            
        return input_error
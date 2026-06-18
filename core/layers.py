import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_error):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
        # Placeholders for gradients (to be accessed by the optimizer)
        self.dweights = None
        self.dbias = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error):
        # Calculate gradients using the chain rule
        input_error = np.dot(output_error, self.weights.T)
        self.dweights = np.dot(self.input.T, output_error)
        self.dbias = np.sum(output_error, axis=0, keepdims=True)
        
        # Return input error to pass to the previous layer
        return input_error
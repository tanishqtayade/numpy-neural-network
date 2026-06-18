import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.lr * layer.dweights
                layer.bias -= self.lr * layer.dbias

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step / iteration counter

    def update(self, layers):
        self.t += 1  # Increment time step for bias correction
        
        for layer in layers:
            # Only update layers that contain trainable weights
            if not hasattr(layer, 'weights'):
                continue
                
            # Dynamically initialize moment states on the layer if they don't exist
            if not hasattr(layer, 'm_w'):
                layer.m_w = np.zeros_like(layer.weights)
                layer.v_w = np.zeros_like(layer.weights)
                layer.m_b = np.zeros_like(layer.bias)
                layer.v_b = np.zeros_like(layer.bias)

            # --- Update Weights ---
            # 1. Update biased first moment estimate
            layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * layer.dweights
            # 2. Update biased second raw moment estimate
            layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * (layer.dweights ** 2)
            # 3. Compute bias-corrected first moment estimate
            m_w_corrected = layer.m_w / (1 - self.beta1 ** self.t)
            # 4. Compute bias-corrected second raw moment estimate
            v_w_corrected = layer.v_w / (1 - self.beta2 ** self.t)
            
            # --- Update Biases ---
            layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.dbias
            layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.dbias ** 2)
            m_b_corrected = layer.m_b / (1 - self.beta1 ** self.t)
            v_b_corrected = layer.v_b / (1 - self.beta2 ** self.t)

            # --- Apply Adaptive Step Updates ---
            layer.weights -= self.lr * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
            layer.bias -= self.lr * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
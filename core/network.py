class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, x_train, y_train, epochs, optimizer):
        """
        Trains the neural network using a dedicated optimizer strategy.
        """
        err = 0  # Initialize err just in case epochs=0
        
        for i in range(epochs):
            # 1. Forward Pass
            output = x_train
            for layer in self.layers:
                output = layer.forward(output)

            # 2. Compute Scalar Loss for metric logging
            err = self.loss(y_train, output)

            # 3. Backward Pass (Compute all gradients)
            error = self.loss_prime(y_train, output)
            for layer in reversed(self.layers):
                error = layer.backward(error)

            # 4. Optimization Step (Update all weights using stored gradients)
            optimizer.update(self.layers)

            # Progress printout (Only prints when training locally outside of mini-batches)
            if (i + 1) % 100 == 0 or i == 0:
                # We can suppress this print if we want to rely entirely on train_mnist.py's prints,
                # but leaving it here is totally fine.
                pass 

        # THIS IS THE MAGIC LINE: 
        # Return the final error so train_mnist.py can add it to the epoch_loss
        return err
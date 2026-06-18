import numpy as np
import pickle
import os
from PIL import Image, ImageOps

def predict_custom_image(image_path):
    # 1. Load the saved neural network
    if not os.path.exists("mnist_model.pkl"):
        print("Error: 'mnist_model.pkl' not found. Please run train_mnist.py first to save the model.")
        return
        
    with open("mnist_model.pkl", "rb") as f:
        net = pickle.load(f)
    
    # 2. Process the custom image to match MNIST standards
    print(f"Processing image: {image_path}...")
    img = Image.open(image_path).convert('L') # Convert to Grayscale
    img = img.resize((28, 28))                # Resize to exactly 28x28 pixels
    
    # MNIST digits are white on a black background. 
    # If you drew a black digit on a white background, we need to invert the colors!
    img = ImageOps.invert(img)
    
    # Convert image to numpy array, flatten to 784, and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    input_vector = img_array.reshape(1, 784)
    
    # 3. Feed forward through your custom framework
    probabilities = net.predict(input_vector)
    prediction = np.argmax(probabilities, axis=1)[0]
    confidence = probabilities[0][prediction] * 100
    
    print("\n--- NETWORK DECISION ---")
    print(f"Predicted Digit: {prediction}")
    print(f"Confidence Level: {confidence:.2f}%")
    print("------------------------")

# Test it on a file named digit.png in your folder
if __name__ == "__main__":
    predict_custom_image("digit.png")
import urllib.request
import os
import gzip

def verify_and_download():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    for file in files:
        file_path = os.path.join(data_dir, file)
        print(f"Downloading {file}...")
        urllib.request.urlretrieve(f"{base_url}{file}", file_path)
        
        # Verify if it's a valid Gzip file
        try:
            with gzip.open(file_path, 'rb') as f:
                f.read(1) # Try to read 1 byte
            print(f"Success: {file} is valid.")
        except Exception as e:
            print(f"FAILED: {file} is corrupted. Please download manually from {base_url}")

if __name__ == "__main__":
    verify_and_download()
import os
import pandas as pd
import numpy as np
import cv2
import sys




def load_dataset(file_path):

    print(f"Loading dataset: {file_path}...")
    _, extension = os.path.splitext(file_path.lower())

    # png loading
    if extension == '.png':
        data, metadata = decode_image(file_path)
        metadata['type'] = 'image'
        metadata['dims'] = 3
        return data, metadata
        
    # csv loading
    elif extension == '.csv':
        df = pd.read_csv(file_path)
        data = df.to_numpy()
        metadata = {'type': 'tabular', 'dims': data.shape[1]}
        return data, metadata
        
    else:
        raise ValueError(f"Unsupported file format {extension}. Please provide a PNG or CSV.")
    



def decode_image(input_image_path):
    # load the image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"error: could not load image from {input_image_path}")
        sys.exit(1)

    # ensure the image is rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # save dimensions
    height, width, channels = image.shape

    # convert from image to array
    pixel_data = image.reshape((height * width, channels))

    # cast to float
    pixel_data = pixel_data.astype(np.float32)

    # normalize between 0 and 1
    pixel_data /= 255.0

    # pack the dimensions into metadata
    metadata = {
        'original_shape': (height, width, channels)
    }

    return pixel_data, metadata




def encode_image(labels, centers, metadata, output_image_path):

    # unpack the dimensions directly from the metadata dictionary
    height, width, channels = metadata['original_shape']

    # map each pixel to its assigned cluster color
    reconstructed_pixels = centers[labels]
    
    # reshape back to original image dimensions
    reconstructed_image = reconstructed_pixels.reshape((height, width, channels))
    
    # denormalize from 0.0-1.0 float back to 0-255 integers
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
    
    # convert RGB back to BGR
    reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR)
    
    # ensure the output directory exists
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    
    # save the image
    cv2.imwrite(output_image_path, reconstructed_image)
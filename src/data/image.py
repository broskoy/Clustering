import numpy as np
import cv2
import sys
import os




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

    return pixel_data, height, width, channels




def encode_image(labels, centers, height, width, channels, output_image_path):
    # map each pixel to its assigned cluster color
    reconstructed_pixels = centers[labels]
    
    # reshape back to original image dimensions
    reconstructed_image = reconstructed_pixels.reshape((height, width, channels))
    
    # denormalize from 0.0-1.0 float back to 0-255 integers
    reconstructed_image = (reconstructed_image * 255.0).astype(np.uint8)
    
    # convert RGB back to BGR
    reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR)
    
    # ensure the output directory exists
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    
    # save the image
    cv2.imwrite(output_image_path, reconstructed_image)
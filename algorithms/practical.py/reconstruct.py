import numpy as np
import cv2
import os

def reconstruct_image(labels, centers, height, width, channels, output_image_path):
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
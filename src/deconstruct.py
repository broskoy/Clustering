import numpy as np
import cv2
import sys

def deconstruct_image(input_image_path):
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
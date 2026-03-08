import numpy as np
import cv2
import sys
import os

def deconstruct_image(input_image_path, output_data_path, output_shape_path):
    # load the image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"error: could not load image from {input_image_path}")
        sys.exit(1)

    # ensure the image is rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # save dimensions
    height, width, channels = image.shape

    # convert from image to array of floats and normalize to 0.0 - 1.0
    pixel_data = image.reshape((height * width, channels)).astype(np.float32) / 255.0

    # ensure the output directory exists
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

    # save the flattened data and the original shape as binary .npy files
    np.save(output_data_path, pixel_data)
    np.save(output_shape_path, np.array([height, width, channels]))
    
    print(f"successfully deconstructed: {input_image_path}")
    print(f"pixels saved to: {output_data_path}")
    print(f"shape saved to: {output_shape_path}")

if __name__ == "__main__":
    # check if the user provided the correct number of terminal arguments
    if len(sys.argv) != 4:
        print("usage: python src/deconstruct.py <input.png> <output_data.npy> <output_shape.npy>")
        sys.exit(1)
    
    # run the function with the provided terminal arguments
    deconstruct_image(sys.argv[1], sys.argv[2], sys.argv[3])
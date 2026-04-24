import os

# We will write these two files next
from .image import decode_image
from .uber import decode_uber

def load_dataset(file_path):
    """
    Reads a dataset file and routes it to the correct decoder.
    Returns:
        data_matrix (ndarray): The N x d data points.
        metadata (dict): Information needed to reconstruct the visual later.
    """
    print(f"Loading dataset: {file_path}...")
    _, extension = os.path.splitext(file_path.lower())

    if extension in ['.png', '.jpg', '.jpeg']:
        data_matrix, metadata = decode_image(file_path)
        return data_matrix, metadata
        
    elif extension in ['.csv']:
        # We assume CSVs are the Uber tabular data for now
        data_matrix, metadata = decode_uber(file_path)
        return data_matrix, metadata
        
    else:
        raise ValueError(f"Unsupported file format: {extension}. Please provide an image or CSV.")
import numpy as np

def extract_lightweight_coreset(data, coreset_size, seed=None):
    """
    Extracts a Lightweight Coreset from an N x d dataset using the 
    Bachem, Lucic, and Krause (2018) methodology.
    
    Returns:
        coreset (ndarray): The sampled points.
        weights (ndarray): The mathematical weights of the sampled points.
    """
    if seed is not None:
        np.random.seed(seed)
        
    N = data.shape[0]
    
    # 1. Find the global mean of the dataset
    mean = np.mean(data, axis=0)
    
    # 2. Calculate the squared Euclidean distance of every point to the mean
    distances_sq = np.sum((data - mean) ** 2, axis=1)
    sum_distances_sq = np.sum(distances_sq)
    
    # 3. Calculate sampling probabilities
    # The probability formula balances uniform sampling (1/2N) with 
    # importance sampling for outliers (distance to mean).
    # We add 1e-10 to prevent division by zero in perfectly uniform data.
    prob = (1.0 / (2 * N)) + (distances_sq / (2 * sum_distances_sq + 1e-10))
    
    # Ensure probabilities sum exactly to 1.0 (NumPy requirement)
    prob = prob / np.sum(prob)
    
    # 4. Sample the coreset indices based on the calculated probabilities
    coreset_indices = np.random.choice(N, size=coreset_size, replace=True, p=prob)
    coreset = data[coreset_indices]
    
    # 5. Calculate the mathematical weights for the sampled points
    weights = 1.0 / (coreset_size * prob[coreset_indices])
    
    return coreset, weights
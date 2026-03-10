import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans




def get_initial_approximation(data, k):
    # run k-means to get a constant factor approximation
    kmeans = MiniBatchKMeans(
        n_clusters=k, 
        init='k-means++', 
        n_init=1, 
        max_iter=10, 
        batch_size=1024, 
        random_state=42
    )
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    return labels, centers




def calculate_cluster_radius(cluster_points, center, epsilon):
    # calculate the cost for all points in this cluster
    distances_sq = np.sum((cluster_points - center) ** 2, axis=1)
    cost = np.sum(distances_sq)
    
    # calculate the boundary radius (small constant added to prevent division by zero)
    radius = np.sqrt(cost / (epsilon * len(cluster_points) + 1e-9))
    return radius, distances_sq




def partition_cluster(cluster_points, distances_sq, radius):
    # create the masks based on radius
    inner_mask = distances_sq <= (radius ** 2)
    outer_mask = ~inner_mask
    
    # separate points inside and outside
    inner_points = cluster_points[inner_mask]
    outer_points = cluster_points[outer_mask]
    outer_distances_sq = distances_sq[outer_mask]
    
    return inner_points, outer_points, outer_distances_sq




def sample_inner_points(inner_points, target_size):
    number_inner = len(inner_points)
    
    if number_inner == 0:
        return np.empty((0, inner_points.shape[1])), np.array([])
        
    # if the target size is larger than the available points take all of them
    actual_size = min(target_size, number_inner)
    
    # uniform random sampling
    indices = np.random.choice(number_inner, size=actual_size, replace=False)
    sampled_points = inner_points[indices]
    
    # uniform weight assignment
    weight_value = number_inner / actual_size
    weights = np.full(actual_size, weight_value)
    
    return sampled_points, weights




def sample_outer_points(outer_points, center, outer_distances_sq, target_size):
    number_outer = len(outer_points)
    
    if number_outer == 0:
        return np.empty((0, outer_points.shape[1])), np.array([])
        
    actual_size = min(target_size, number_outer)
    cost_outer = np.sum(outer_distances_sq)
    
    if cost_outer == 0:
        # Fallback to uniform if all outer points sit exactly on the radius boundary
        return sample_inner_points(outer_points, target_size)
        
    # calculate non-uniform probability distribution
    probabilities = outer_distances_sq / cost_outer
    
    # sample based on probabilities
    indices = np.random.choice(number_outer, size=actual_size, replace=False, p=probabilities)
    sampled_points = outer_points[indices]
    
    # calculate inversely proportional weights
    sampled_distances_sq = outer_distances_sq[indices]
    weights = cost_outer / (actual_size * sampled_distances_sq)
    
    return sampled_points, weights




def build_coreset(data, k, epsilon, inner_sample_size, outer_sample_size):
    labels, centers = get_initial_approximation(data, k)
    
    coreset_points = []
    coreset_weights = []
    
    for i in range(k):
        # Extract points belonging to the current rough cluster
        cluster_points = data[labels == i]
        if len(cluster_points) == 0:
            continue
            
        center = centers[i]
        
        # Partition the data
        radius, distances_sq = calculate_cluster_radius(cluster_points, center, epsilon)
        inner_pts, outer_pts, outer_dist_sq = partition_cluster(cluster_points, distances_sq, radius)
        
        # Sample the partitions
        sampled_inner, weights_inner = sample_inner_points(inner_pts, inner_sample_size)
        sampled_outer, weights_outer = sample_outer_points(outer_pts, center, outer_dist_sq, outer_sample_size)
        
        # Store results
        if len(sampled_inner) > 0:
            coreset_points.append(sampled_inner)
            coreset_weights.append(weights_inner)
            
        if len(sampled_outer) > 0:
            coreset_points.append(sampled_outer)
            coreset_weights.append(weights_outer)
            
    # Concatenate all sampled blocks into final arrays
    final_points = np.vstack(coreset_points)
    final_weights = np.concatenate(coreset_weights)
    
    return final_points, final_weights
import time
from sklearn.cluster import KMeans
from deconstruct import deconstruct_image
from reconstruct import reconstruct_image
from coreset import build_coreset


def compress(input_path, output_path, k, epsilon, inner_sample, outer_sample):

    print("1 Loading image data...")
    pixels, h, w, c = deconstruct_image(input_path)
    print(f"Original shape: {h}x{w} ({len(pixels)} pixels)")

    print("\n2 Extracting coreset...")
    start_time = time.time()
    coreset_points, coreset_weights = build_coreset(pixels, k, epsilon, inner_sample, outer_sample)
    coreset_time = time.time() - start_time
    print(f"Coreset size: {len(coreset_points)} points")
    print(f"Extraction time: {coreset_time:.2f} seconds")

    print("\n3 Running KMeans on the small coreset...")
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=300, random_state=42)
    kmeans.fit(coreset_points, sample_weight=coreset_weights)
    clustering_time = time.time() - start_time
    print(f"Clustering time: {clustering_time:.2f} seconds")

    print("\n4 Mapping colors back to original pixels...")
    labels = kmeans.predict(pixels)
    centers = kmeans.cluster_centers_

    print("\n5. Reconstructing image...")
    reconstruct_image(labels, centers, h, w, c, output_path)
    print("Test finished successfully!")
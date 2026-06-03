import time
import os
import csv
import numpy as np
from sklearn.cluster import KMeans

from src.converter import load_dataset, encode_image
from src.coreset import build_coreset
from src.visualize import generate_plot 




# Execution Parameters
iterations = 5
k_values = [2, 4, 8, 16, 32, 64]
q_budgets = [64, 128, 256, 512, 1024, 2048, 4096]

# Feature Toggles
enable_metrics = True
enable_output = False




# Executes a standard loyd method for clustering
def run_standard_loyd(data, k):
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=100, random_state=42)
    kmeans.fit(data)

    labels = kmeans.predict(data)
    centers = kmeans.cluster_centers_
        
    cost_loyd= np.sum((data - centers[labels])**2)
    time_loyd = time.time() - start_time
    
    return cost_loyd, time_loyd




# Executes a single coreset extraction and clustering run
def run_biased_coreset(data, k, q):
    start_time = time.time()
    
    coreset_points, coreset_weights = build_coreset(data, k, q)
    
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=100, random_state=42)
    kmeans.fit(coreset_points, sample_weight=coreset_weights)
    
    labels = kmeans.predict(data)
    centers = kmeans.cluster_centers_
    
    cost_biased = np.sum((data - centers[labels])**2)
    time_biased = time.time() - start_time
    
    return cost_biased, time_biased




# Executes uniform random sampling then clusters
def run_uniform_coreset(data, k, q):
    start_time = time.time()
    
    # Uniformly select random indices
    indices = np.random.choice(len(data), size=q, replace=False)
    random_points = data[indices]
    
    # Run K-Means without weights
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=100, random_state=42)
    kmeans.fit(random_points)
    
    labels = kmeans.predict(data)
    centers = kmeans.cluster_centers_
    
    cost_uniform = np.sum((data - centers[labels])**2)
    time_uniform = time.time() - start_time
    
    return cost_uniform, time_uniform




# will record the performance of each algorithm in metrics
def dataset_metrics(input_path, file_name, writer1, writer2, writer3):

    print(f"\n{'='*30}\nSTARTING BATCH FOR [{file_name}]\n{'='*30}")

    data, metadata = load_dataset(input_path)
    print(f"Data shape: {data.shape}, Type: {metadata['type']}")

    for k in k_values:

        for q in q_budgets:

            print(f"\n  Testing k={k}, |Q|={q}")
            
            for i in range(iterations):
                try:
                    #run the loyd version
                    cost_loyd, time_loyd = run_standard_loyd(data, k)

                    # log the results for loyd
                    writer1.writerow([
                        file_name, k, q, i, cost_loyd, round(time_loyd, 4)
                    ])

                    # run the biased coreset version
                    cost_biased, time_biased = run_biased_coreset(data, k, q)

                    # log the results for biased
                    writer2.writerow([
                        file_name, k, q, i, cost_biased, round(time_biased, 4)
                    ])
                    
                    # run the uniform coreset version
                    cost_uniform, time_uniform = run_uniform_coreset(data, k, q)
                    
                    # log the results for uniform
                    writer3.writerow([
                        file_name, k, q, i, cost_uniform, round(time_uniform, 4)
                    ])
                        
                    print(f"    Iteration {i}: {time_loyd:.3f} | {time_biased:.3f} | {time_uniform:.3f}")
                            
                except Exception as e:
                    print(f"    Iteration {i} Failed: {e}")




# will generate visualizations of the clusters in output
def dataset_output(file_path, file_output, file_name):
    data, metadata = load_dataset(file_path)

    q = 4096

    for k in k_values:
        coreset_points, coreset_weights = build_coreset(data, k, q)
    
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=300, random_state=42)
        kmeans.fit(coreset_points, sample_weight=coreset_weights)
        
        labels = kmeans.predict(data)
        centers = kmeans.cluster_centers_

        plot_path = os.path.join(file_output, f"{file_name}-k{k}-q{q}.png")
        generate_plot(data, centers, metadata, plot_path, title=f"{file_name} (k={k}, |Q|={q})")
        
        if metadata['type'] == 'image':
            image_path = os.path.join(file_output, f"{file_name}-k{k}-q{q}-reconstructed.png")
            encode_image(labels, centers, metadata, image_path)




def main():

    metrics_file1 = None
    metrics_file2 = None
    metrics_file3 = None
    writer1 = None
    writer2 = None
    writer3 = None

    if (enable_metrics):
        metrics_file1 = open("metrics/metrics_loyd.csv", mode='w', newline='')
        writer1 = csv.writer(metrics_file1)
        writer1.writerow([
            "Dataset", "Clusters", "Budget", "Iteration", "Cost", "Time"
        ])

        metrics_file2 = open("metrics/metrics_biased.csv", mode='w', newline='')
        writer2 = csv.writer(metrics_file2)
        writer2.writerow([
            "Dataset", "Clusters", "Budget", "Iteration", "Cost", "Time"
        ])

        metrics_file3 = open("metrics/metrics_uniform.csv", mode='w', newline='')
        writer3 = csv.writer(metrics_file3)
        writer3.writerow([
            "Dataset", "Clusters", "Budget", "Iteration", "Cost", "Time"
        ])

    # iterate through input datasets
    for filename in os.listdir("input"):
        # get the path and name
        file_path = os.path.join("input", filename)
        file_name = os.path.splitext(filename)[0]

        # create the specificc output folder
        file_output = os.path.join("output", file_name)
        os.makedirs(file_output, exist_ok=True)
        
        if enable_metrics:
            dataset_metrics(file_path, file_name, writer1, writer2, writer3)
                
        if enable_output:
            dataset_output(file_path, file_output, file_name)

    if metrics_file1 is not None:
        metrics_file1.close()

    if metrics_file2 is not None:
        metrics_file2.close()

    if metrics_file3 is not None:
        metrics_file3.close()




if __name__ == "__main__":
    main()
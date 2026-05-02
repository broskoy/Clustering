import time
import os
import csv
import numpy as np
from sklearn.cluster import KMeans

from src.loader import load_dataset, encode_image
from src.coreset import build_coreset
from src.visualize import generate_plot 

def process_dataset(input_path, file_name, out_dir, k_values, q_budgets, iterations, csv_writer):
    print(f"\n{'='*30}")
    print(f"STARTING BATCH FOR [{file_name}]")
    print(f"{'='*30}")

    data, metadata = load_dataset(input_path)
    print(f"Data shape: {data.shape}, Type: {metadata['type']}")

    # Loop over k_values first, as the baseline cost depends entirely on k
    for k in k_values:
        print(f"\n--- Running Baseline for k={k} ---")
        start_baseline = time.time()
        baseline_kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=300, random_state=42)
        baseline_kmeans.fit(data)
        
        cost_p_c = baseline_kmeans.inertia_
        print(f"  Baseline Cost(P,C): {cost_p_c:.2f}")

        # The 2D Grid Search Loop: |Q| vs k
        for q in q_budgets:
            # Distribute the budget evenly, enforcing a minimum of 1 point to prevent crashes
            inner_sample = max(1, (q // 2) // k)
            outer_sample = max(1, (q // 2) // k)

            # find a good batch size for initializing the coreset
            target_batch = k * 1024
            max_safe_batch = max(1024, len(data) // 10) 
            dynamic_batch_size = min(target_batch, max_safe_batch)

            print(f"\n  Testing k={k}, |Q|={q} (Inner: {inner_sample*k}, Outer: {outer_sample*k}, Batch: {dynamic_batch_size})")
            
            for i in range(iterations):
                try:
                    start_time = time.time()
                    
                    # Epsilon removed from the function call
                    coreset_points, coreset_weights = build_coreset(data, k, inner_sample, outer_sample, dynamic_batch_size)
                    
                    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=300, random_state=42)
                    kmeans.fit(coreset_points, sample_weight=coreset_weights)
                    
                    labels = kmeans.predict(data)
                    centers = kmeans.cluster_centers_
                    
                    cost_q_c_prime = kmeans.inertia_
                    cost_p_c_prime = np.sum((data - centers[labels])**2)
                    exec_time = time.time() - start_time
                    
                    # Epsilon removed from the CSV row output
                    csv_writer.writerow([
                        file_name, metadata['type'], k, q, i, 
                        round(exec_time, 4), cost_p_c, cost_q_c_prime, cost_p_c_prime
                    ])
                    print(f"    Iter {i}: Ratio: {cost_p_c_prime / cost_p_c:.3f}")
                except Exception as e:
                    print(f"    Iter {i} FAILED: {e}")

def main():
    # Target the Uber dataset
    input_dirs = ["input/real"]
    base_output_dir = "output"
    metric_dir = "metrics"
    os.makedirs(metric_dir, exist_ok=True)

    iterations = 5 
    
    # The New Grid Parameters: K vs Q
    k_values = [2, 4, 8, 16, 32, 64]
    q_budgets = [8192, 4096, 2048, 1024, 512, 256, 128, 64]

    csv_file_path = os.path.join(metric_dir, "metrics.csv")
    
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Epsilon removed from the CSV headers
        writer.writerow([
            "Dataset_Name", "Dataset_Type", "K_Value", "Q_Budget", "Iteration", 
            "Execution_Time", "Cost_P_C", "Cost_Q_C_prime", "Cost_P_C_prime"
        ])

        for folder in input_dirs:
            if not os.path.exists(folder): continue
            out_dir = os.path.join(base_output_dir, os.path.basename(folder))
            os.makedirs(out_dir, exist_ok=True)

            for filename in os.listdir(folder):
                # Filter specifically for Uber to isolate the run
                if not filename.lower().endswith('.csv') or "uber" not in filename.lower():
                    continue
                input_path = os.path.join(folder, filename)
                file_name = os.path.splitext(filename)[0]
                process_dataset(input_path, file_name, out_dir, k_values, q_budgets, iterations, writer)

if __name__ == "__main__":
    main()
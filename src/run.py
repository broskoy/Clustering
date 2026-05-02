import time
import os
import csv
import numpy as np
from sklearn.cluster import KMeans

from src.loader import load_dataset, encode_image
from src.coreset import build_coreset
from src.visualize import generate_plot 

# ================================
# 1 CONFIGURATION CONSTANTS
# ================================
#Folders
input_folder = "input"
output_folder = "output"
metric_folder = "metrics"

# Execution Parameters
iterations = 5
k_values = [2, 4, 8, 16, 32, 64]
q_budgets = [4096]

# Feature Toggles
enable_logging = False
enable_plotting = False 




# ================================
# 2 EXPERIMENT RUNNERS
# ================================
def run_coreset_experiment(data, k, inner_sample, outer_sample, batch_size):
    """Executes a single coreset extraction and clustering run"""
    start_time = time.time()
    
    coreset_points, coreset_weights = build_coreset(
        data, k, inner_sample, outer_sample, batch_size
    )
    
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=300, random_state=42)
    kmeans.fit(coreset_points, sample_weight=coreset_weights)
    
    labels = kmeans.predict(data)
    centers = kmeans.cluster_centers_
    
    cost_q_c_prime = kmeans.inertia_
    cost_p_c_prime = np.sum((data - centers[labels])**2)
    exec_time = time.time() - start_time
    
    return {
        'exec_time': exec_time,
        'cost_q': cost_q_c_prime,
        'cost_p': cost_p_c_prime,
        'labels': labels,
        'centers': centers
    }




def run_random_baseline(data, k, q_budget):
    """Executes uniform random sampling for comparing"""
    start_time = time.time()
    
    # Uniformly select random indices
    indices = np.random.choice(len(data), size=q_budget, replace=False)
    random_points = data[indices]
    
    # Run K-Means without weights
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=300, random_state=42)
    kmeans.fit(random_points)
    
    labels = kmeans.predict(data)
    centers = kmeans.cluster_centers_
    
    cost_random = np.sum((data - centers[labels])**2)
    exec_time = time.time() - start_time
    
    return cost_random, exec_time





# ================================
# 3 METRICS REPORTING
# ================================
def log_metrics(writer, file_name, metadata, k, q, iteration, base_cost, coreset_res, rand_cost, rand_time):
    """Handles all CSV writing logic"""
    writer.writerow([
        file_name, 
        metadata['type'], 
        k, 
        q, 
        iteration, 
        round(coreset_res['exec_time'], 4), 
        base_cost, 
        coreset_res['cost_q'], 
        coreset_res['cost_p'],
        rand_cost,
        round(rand_time, 4)
    ])




def process_dataset(input_path, output_path, file_name, csv_writer):
    """Handles all of the work for one dataset"""

    print(f"\n{'='*30}\nSTARTING BATCH FOR [{file_name}]\n{'='*30}")

    data, metadata = load_dataset(input_path)
    print(f"Data shape: {data.shape}, Type: {metadata['type']}")

    for k in k_values:
        print(f"\n--- Running Baseline for k={k} ---")
        start_baseline = time.time()
        baseline_kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=300, random_state=42)
        baseline_kmeans.fit(data)
        
        cost_p_c = baseline_kmeans.inertia_
        print(f"  Baseline Cost(P,C): {cost_p_c:.2f} ({time.time() - start_baseline:.2f}s)")

        for q in q_budgets:
            # tune the inner/outer split here
            inner_sample = max(1, (q // 2) // k)
            outer_sample = max(1, (q // 2) // k)
            
            # dynamic batch size
            target_batch = k * 1024
            max_safe_batch = max(1024, len(data) // 10) 
            dynamic_batch_size = min(target_batch, max_safe_batch)

            print(f"\n  Testing k={k}, |Q|={q}")
            
            for i in range(iterations):
                try:
                    # run the biased coreset version
                    coreset_results = run_coreset_experiment(
                        data, k, inner_sample, outer_sample, dynamic_batch_size
                    )
                    
                    # run the uniform coreset version
                    rand_cost, rand_time = run_random_baseline(data, k, q)
                    
                    # log the results
                    if enable_logging:
                        log_metrics(
                            csv_writer, file_name, metadata, k, q, i, 
                            cost_p_c, coreset_results, rand_cost, rand_time
                        )
                        
                    print(f"    Iter {i}: Coreset Ratio: {coreset_results['cost_p'] / cost_p_c:.3f} | Random Ratio: {rand_cost / cost_p_c:.3f}")
                    
                    # generate visuals
                    if enable_plotting and i == 0:
                        plot_path = os.path.join(output_path, f"{file_name}-k{k}-q{q}.png")
                        generate_plot(data, coreset_results['centers'], metadata, plot_path, title=f"{file_name} (k={k}, |Q|={q})")
                        
                        if metadata['type'] == 'image':
                            image_path = os.path.join(output_path, f"{file_name}-k{k}-q{q}-reconstructed.png")
                            encode_image(coreset_results['labels'], coreset_results['centers'], metadata, image_path)
                            
                except Exception as e:
                    print(f"    Iter {i} FAILED: {e}")





# ================================
# 4 MAIN EXECUTION
# ================================
def main():
    # create metrics csv
    csv_file_path = os.path.join(metric_folder, "metrics.csv")
    
    #open CSV safely
    with open(csv_file_path, mode='w', newline='') as file:

        # create the columns of the metrics 
        writer = csv.writer(file)
        if enable_logging:
            writer.writerow([
                "Dataset_Name", "Dataset_Type", "K_Value", "Q_Budget", "Iteration", 
                "Exec_Time_Coreset", "Cost_P_C_Baseline", "Cost_Q_C_prime", 
                "Cost_P_C_prime", "Cost_Random", "Exec_Time_Random"
            ])

        # iterate through input datasets
        for filename in os.listdir(input_folder):
            # get the path and name
            input_path = os.path.join(input_folder, filename)
            file_name = os.path.splitext(filename)[0]

            # create the specificc output folder
            specific_output_folder = os.path.join(output_folder, file_name)
            os.makedirs(specific_output_folder, exist_ok=True)
            
            process_dataset(input_path, specific_output_folder, file_name, writer)




if __name__ == "__main__":
    main()
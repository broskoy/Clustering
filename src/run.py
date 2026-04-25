import time
import os
import csv
from sklearn.cluster import KMeans

# Updated imports to match the new tree structure
from src.loader import load_dataset, encode_image
from src.coreset import build_coreset
from src.visualize import generate_plot 


def process_dataset(input_path, file_name, out_dir, k_values, epsilon, total_inner, total_outer, iterations, csv_writer):
    print(f"\n{'='*30}")
    print(f"STARTING BATCH FOR [{file_name}]")
    print(f"{'='*30}")

    # 1. Load data generically
    data, metadata = load_dataset(input_path)
    print(f"Data shape: {data.shape}, Type: {metadata['type']}")

    # run compression for each k value
    for k in k_values:
        inner_sample = total_inner // k
        outer_sample = total_outer // k

        print(f"\n--- Running k={k} ---")
        
        for i in range(iterations):
            try:
                start_time = time.time()
                
                # 2. Extract Coreset
                coreset_points, coreset_weights = build_coreset(data, k, epsilon, inner_sample, outer_sample)
                
                # 3. Run KMeans on the weighted coreset
                kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=300, random_state=42)
                kmeans.fit(coreset_points, sample_weight=coreset_weights)
                
                # 4. Predict labels for the full dataset and get centers
                labels = kmeans.predict(data)
                centers = kmeans.cluster_centers_
                
                exec_time = time.time() - start_time
                
                # Write the row directly to the CSV
                csv_writer.writerow([file_name, metadata['type'], k, i, round(exec_time, 4)])
                print(f"  Iteration {i}: {exec_time:.2f} seconds")
                
                # 5. Visualization Phase (Only run on iteration 0)
                if i == 0:
                    # Always generate the mathematical plot (2D or 3D)
                    plot_path = os.path.join(out_dir, f"{file_name}-{k}_plot.png")
                    generate_plot(data, centers, metadata, plot_path, title=f"Dataset: {file_name} (k={k})")
                    
                    # If it is an image, also rebuild the visual PNG
                    if metadata['type'] == 'image':
                        image_path = os.path.join(out_dir, f"{file_name}-{k}_reconstructed.png")
                        encode_image(labels, centers, metadata, image_path)
                
            except Exception as e:
                print(f"  Iteration {i} FAILED: {e}")


def main():
    # define the input directories to scan
    # input_dirs = ["input/image", "input/real", "input/synthetic"]
    input_dirs = ["input/synthetic"]
    base_output_dir = "output"
    plot_dir = "plots/execution_times"
    
    os.makedirs(plot_dir, exist_ok=True)

    k_values = [2, 4, 8, 16, 32, 64]
    epsilon = 0.5
    total_inner = 4096
    total_outer = 4096
    iterations = 1

    # set up the CSV file
    csv_file_path = os.path.join(plot_dir, "execution_times.csv")
    
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset_Name", "Dataset_Type", "K_Value", "Iteration", "Execution_Time"])

        # iterate through every input directory
        for folder in input_dirs:
            if not os.path.exists(folder):
                continue
                
            # Create a matching output sub-folder (e.g., output/image, output/synthetic)
            folder_type = os.path.basename(folder)
            out_dir = os.path.join(base_output_dir, folder_type)
            os.makedirs(out_dir, exist_ok=True)

            for filename in os.listdir(folder):
                if not filename.lower().endswith(('.png', '.csv')):
                    continue
                
                input_path = os.path.join(folder, filename)
                file_name = os.path.splitext(filename)[0]

                process_dataset(
                    input_path, file_name, out_dir, 
                    k_values, epsilon, total_inner, total_outer, 
                    iterations, writer
                )
            
    print(f"\nAll tests complete. Data saved to {csv_file_path}")

if __name__ == "__main__":
    main()
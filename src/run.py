import time
import os
import csv
from compress import compress


def process_single_image(input_path, file_name, output_dir, k_values, epsilon, total_inner, total_outer, iterations, csv_writer):
    print(f"\n{'='*30}")
    print(f"STARTING BATCH FOR [{file_name}.png]")
    print(f"{'='*30}")

    # run compression for each k value
    for k in k_values:
        output_filename = f"{file_name}-{k}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # dynamically calculate the budget per cluster
        inner_sample = total_inner // k
        outer_sample = total_outer // k

        print(f"\n--- Running k={k} ---")
        
        for i in range(iterations):
            try:
                start_time = time.time()
                compress(input_path, output_path, k, epsilon, inner_sample, outer_sample)
                compress_time = time.time() - start_time
                
                # Write the row directly to the CSV
                csv_writer.writerow([file_name, k, i, round(compress_time, 4)])
                print(f"  Iteration {i}: {compress_time:.2f} seconds")
                
            except Exception as e:
                print(f"  Iteration {i} FAILED: {e}")


def main():
    # directory paths
    input_dir = "input"
    output_dir = "output"
    data_dir = "data"
    
    # ensure output directories exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # define the parameters for testing
    k_values = [2, 4, 8, 16, 32, 64]
    epsilon = 0.5
    total_inner = 4096
    total_outer = 4096
    iterations = 10

    # set up the CSV file
    csv_file_path = os.path.join(data_dir, "execution_times.csv")
    
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the column headers
        writer.writerow(["Image_Name", "K_Value", "Iteration", "Execution_Time"])

        # iterate through every file in the input directory
        for filename in os.listdir(input_dir):
            # skip other files
            if not filename.lower().endswith(('.png')):
                continue
            
            # extract the name without the extension
            input_path = os.path.join(input_dir, filename)
            file_name = os.path.splitext(filename)[0]

            # pass the workload to the helper function
            process_single_image(
                input_path, file_name, output_dir, 
                k_values, epsilon, total_inner, total_outer, 
                iterations, writer
            )
            
    print(f"\nAll tests complete. Data saved to {csv_file_path}")


if __name__ == "__main__":
    main()
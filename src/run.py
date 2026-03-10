import time
import os
from compress import compress


def main():
    input_dir = "input"
    output_dir = "output"
    
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # define the parametes for testing
    k_values = [2, 4, 8, 16, 32, 64]
    epsilon = 0.5
    total_inner = 4096
    total_outer = 4096

    # iterate through every file in the input directory
    for filename in os.listdir(input_dir):
        # skip other files
        if not filename.lower().endswith(('.png')):
            continue
        
        # extract the name without the extension
        input_path = os.path.join(input_dir, filename)
        file_name = os.path.splitext(filename)[0]

        print(f"\n{'='*20}")
        print(f"BATCH PROCESSING STARTING FOR: {filename}")
        print(f"{'='*20}\n")

        # run compression for each k value
        for k in k_values:
            output_filename = f"{file_name}-{k}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # dynamically calculate the budget per cluster
            inner_sample = total_inner // k
            outer_sample = total_outer // k

            print(f"Starting compression (k={k})")
            start_time = time.time()
            compress(input_path, output_path, k, epsilon, inner_sample, outer_sample)
            compress_time = time.time() - start_time
            print(f"Saved result to {output_path}")
            print(f"Time taken: {compress_time:.2f}\n")


if __name__ == "__main__":
    main()
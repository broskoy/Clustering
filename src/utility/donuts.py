import numpy as np
import pandas as pd
import os



# creates a synthetic dataset with three fuzzy donuts
def generate_synthetic(n_points, output_path):
    all_data = []
    
    # center_x, center_y, radius_mean, radius_std, weight
    configs = [
        (0, 0, 10.0, 2.0, 0.65),
        (18, 18, 5.0, 1.4, 0.25),
        (0, 23, 3.0, 1.0, 0.10)
    ]

    for c_x, c_y, r_mean, r_std, weight in configs:
        points_count = int(n_points * weight)
        
        angles = np.random.uniform(0, 2 * np.pi, points_count)
        radii = np.random.normal(r_mean, r_std, points_count)

        x = c_x + radii * np.cos(angles)
        y = c_y + radii * np.sin(angles)
        
        all_data.append(np.column_stack((x, y)))

    final_data = np.vstack(all_data)
    df = pd.DataFrame(final_data, columns=['x', 'y'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(final_data)} points to {output_path}")





if __name__ == "__main__":
    generate_synthetic(
        n_points=1000000, 
        output_path="input/synthetic/donuts.csv"
    )
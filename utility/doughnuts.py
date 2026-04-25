import numpy as np
import pandas as pd
import os

def generate_complex_synthetic(n_points, output_path):
    """
    Generates a 4-doughnut system: 
    1. Large Fuzzy (Origin)
    2. Small Dense (Origin)
    3. Medium Balanced (Satellite)
    4. Tiny Sharp (Satellite)
    """
    all_data = []
    
    # Configuration: (center_x, center_y, radius_mean, radius_std, weight)
    configs = [
        (0, 0, 15.0, 4.0, 0.4),   # Large Fuzzy (40% of points)
        (0, 0, 5.0, 0.5, 0.2),    # Small Dense (20% of points)
        (30, 20, 10.0, 1.5, 0.25), # Medium Satellite (25% of points)
        (-25, 10, 3.0, 0.2, 0.15)  # Tiny Sharp Satellite (15% of points)
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
    generate_complex_synthetic(
        n_points=200000, 
        output_path="input/synthetic/doughnuts.csv"
    )
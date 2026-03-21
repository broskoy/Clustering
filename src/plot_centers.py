import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.cluster import KMeans




def generate_density_projections(input_path, output_path, k):
    print(f"Loading image...")
    
    img = imread(input_path)
    if img.max() > 1.0:
        img = img / 255.0
        
    h, w, c = img.shape
    flat_image = img.reshape(h * w, c)

    # Sample for the K-Means calculation (10,000 points for speed)
    np.random.seed(42)
    calc_indices = np.random.choice(flat_image.shape[0], 10000, replace=False)
    calc_pixels = flat_image[calc_indices]

    print(f"Calculating cluster centers...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(calc_pixels)
    rgb_centers = kmeans.cluster_centers_

    # Sample for the Background Density Plot (100,000 points for dense visualization)
    print("Sampling background points for rendering...")
    render_indices = np.random.choice(flat_image.shape[0], 100000, replace=False)
    bg_pixels = flat_image[render_indices]

    print("Generating RGB planar density projections...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"RGB Density Distribution with {k} Cluster Centers", fontsize=16, fontweight='bold')

    # Define plotting parameters for clean reuse
    bg_size = 5
    bg_alpha = 0.02
    center_size = 250
    center_color = 'black'
    center_edge = 'white'

    # --- Plot 1: Red vs Green ---
    axes[0].scatter(bg_pixels[:, 0], bg_pixels[:, 1], c=bg_pixels, s=bg_size, alpha=bg_alpha, edgecolors='none')
    axes[0].scatter(rgb_centers[:, 0], rgb_centers[:, 1], marker='*', c=center_color, s=center_size, edgecolors=center_edge, linewidth=1.5)
    
    axes[0].set_title("Red vs. Green Plane")
    axes[0].set_xlabel("Red Channel")
    axes[0].set_ylabel("Green Channel")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)

    # --- Plot 2: Red vs Blue ---
    axes[1].scatter(bg_pixels[:, 0], bg_pixels[:, 2], c=bg_pixels, s=bg_size, alpha=bg_alpha, edgecolors='none')
    axes[1].scatter(rgb_centers[:, 0], rgb_centers[:, 2], marker='*', c=center_color, s=center_size, edgecolors=center_edge, linewidth=1.5)
    
    axes[1].set_title("Red vs. Blue Plane")
    axes[1].set_xlabel("Red Channel")
    axes[1].set_ylabel("Blue Channel")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)

    # --- Plot 3: Green vs. Blue ---
    axes[2].scatter(bg_pixels[:, 1], bg_pixels[:, 2], c=bg_pixels, s=bg_size, alpha=bg_alpha, edgecolors='none')
    axes[2].scatter(rgb_centers[:, 1], rgb_centers[:, 2], marker='*', c=center_color, s=center_size, edgecolors=center_edge, linewidth=1.5)
    
    axes[2].set_title("Green vs. Blue Plane")
    axes[2].set_xlabel("Green Channel (0.0 - 1.0)")
    axes[2].set_ylabel("Blue Channel (0.0 - 1.0)")
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].set_xlim(-0.05, 1.05)
    axes[2].set_ylim(-0.05, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {output_path}")




if __name__ == "__main__":
    generate_density_projections("input/birb.png", "plot/birb_centers.png", 16)
    generate_density_projections("input/cherry.png", "plot/cherry_centers.png", 16)
    generate_density_projections("input/forest.png", "plot/forest_centers.png", 16)
    generate_density_projections("input/windows.png", "plot/windows_centers.png", 16)
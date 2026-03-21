import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.cluster import KMeans




def generate_centers_plot(image_path, k=32, output_path="plots/3d_centers.png"):
    print(f"Loading image from {image_path}...")
    
    img = imread(image_path)
    
    # Strip Alpha channel if present
    if img.shape[-1] == 4:
        img = img[:, :, :3]
        
    if img.max() > 1.0: 
        img = img / 255.0
        
    flat_image = img.reshape(-1, 3)

    # Sample data
    np.random.seed(42)
    calc_pixels = flat_image[np.random.choice(flat_image.shape[0], 10000, replace=False)]
    bg_pixels = flat_image[np.random.choice(flat_image.shape[0], 50000, replace=False)]

    print(f"Calculating {k} cluster centers...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(calc_pixels)
    rgb_centers = kmeans.cluster_centers_

    print("Rendering 3D Plots...")
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"3D RGB Spatial Distribution of {k} Clusters", fontsize=16, fontweight='bold')

    # Helper function to format 3D axes cleanly for PDFs
    def format_3d_axes(ax, title, elev, azim):
        ax.scatter(bg_pixels[:, 0], bg_pixels[:, 1], bg_pixels[:, 2], 
                   c=bg_pixels, s=5, alpha=0.02, edgecolors='none')
        
        ax.scatter(rgb_centers[:, 0], rgb_centers[:, 1], rgb_centers[:, 2], 
                   c='black', marker='D', s=100, edgecolors='white', linewidth=1.5, depthshade=False)

        ax.set_title(title, pad=10)
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        # Remove the gray background panes for clean printing
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        ax.view_init(elev=elev, azim=azim)

    # Plot 1: Front-Right Angle
    ax1 = fig.add_subplot(121, projection='3d')
    format_3d_axes(ax1, "Angle 1: Azimuth 45°", elev=25, azim=45)

    # Plot 2: Back-Left Angle (Rotated 90 degrees)
    ax2 = fig.add_subplot(122, projection='3d')
    format_3d_axes(ax2, "Angle 2: Azimuth 135°", elev=25, azim=135)

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {output_path}")




if __name__ == "__main__":
    generate_centers_plot("input/birb.png", "plot/birb_centers.png", 16)
    generate_centers_plot("input/cherry.png", "plot/cherry_centers.png", 16)
    generate_centers_plot("input/forest.png", "plot/forest_centers.png", 16)
    generate_centers_plot("input/windows.png", "plot/windows_centers.png", 16)
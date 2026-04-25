import os
import numpy as np
import matplotlib.pyplot as plt




def generate_plot(data, centers, metadata, output_path, title="Cluster Visualization"):
    """
    Master routing function for visualizations.
    Reads the dataset dimensions and routes to the correct chart type.
    """
    dims = metadata.get('dims', data.shape[1])
    
    if dims == 2:
        _plot_2d(data, centers, output_path, title)
    elif dims == 3:
        _plot_3d_rgb(data, centers, output_path, title)
    else:
        print(f"Visualization skipped: No plotting logic for {dims}D data.")




def _plot_2d(data, centers, output_path, title):
    plt.figure(figsize=(10, 10))
    
    # Original data in the background
    plt.scatter(data[:, 0], data[:, 1], c='gray', s=1, alpha=0.01, label='Original Data')
    
    # Cluster centers in the foreground
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='D', s=50, 
                edgecolors='white', label='Cluster Centers')
    
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"2D Plot saved to {output_path}")




def _plot_3d_rgb(data, centers, output_path, title):
    # Downsample background pixels for performance and clean plotting
    np.random.seed(42)
    sample_size = min(50000, data.shape[0])
    bg_pixels = data[np.random.choice(data.shape[0], sample_size, replace=False)]

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    def format_3d_axes(ax, view_title, elev, azim):
        # Background pixels colored by their actual RGB values
        ax.scatter(bg_pixels[:, 0], bg_pixels[:, 1], bg_pixels[:, 2], 
                   c=bg_pixels, s=5, alpha=0.02, edgecolors='none')
        
        # Black diamonds for the computed cluster centers
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
                   c='black', marker='D', s=100, edgecolors='white', linewidth=1.5, depthshade=False)

        ax.set_title(view_title, pad=10)
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        ax.view_init(elev=elev, azim=azim)

    # Plot 45 degrees
    ax1 = fig.add_subplot(121, projection='3d')
    format_3d_axes(ax1, "Angle: 45°", elev=25, azim=45)

    # Plot 135 degrees
    ax2 = fig.add_subplot(122, projection='3d')
    format_3d_axes(ax2, "Angle: 135°", elev=25, azim=135)

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Prevent memory leaks during loops
    print(f"3D RGB Plot saved to {output_path}")
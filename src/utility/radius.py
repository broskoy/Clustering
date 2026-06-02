import numpy as np
import matplotlib.pyplot as plt

def generate_partition_diagram():
    # Set random seed for consistent point placement
    np.random.seed(42)

    # Define the core geometric variables
    center = np.array([0.0, 0.0])
    radius = 2.5

    # Generate inner points (C_in) using a tight normal distribution
    # We strictly filter out any points that randomly spawn outside the radius
    inner_candidates = np.random.normal(loc=center, scale=0.9, size=(1000, 2))
    distances = np.linalg.norm(inner_candidates - center, axis=1)
    inner_points = inner_candidates[distances <= radius][:350]

    # Generate outer points (C_out) using a wider distribution
    # We strictly filter out any points that spawn inside the radius
    outer_candidates = np.random.normal(loc=center, scale=2.5, size=(1000, 2))
    outer_distances = np.linalg.norm(outer_candidates - center, axis=1)
    outer_points = outer_candidates[outer_distances > radius][:120]

    # Initialize the plot with a square aspect ratio
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the outer points with a secondary color and lower opacity to indicate outliers
    ax.scatter(outer_points[:, 0], outer_points[:, 1], c='#ff7f0e', alpha=0.5, s=40, label='$C_{out}$ (Outer Points)')

    # Plot the inner points with a solid primary color to indicate density
    ax.scatter(inner_points[:, 0], inner_points[:, 1], c='#1f77b4', alpha=0.9, s=40, label='$C_{in}$ (Inner Points)')

    # Draw the boundary radius as a dashed circle
    circle = plt.Circle(center, radius, color='black', fill=False, linestyle='--', linewidth=2.0)
    ax.add_patch(circle)

    # Plot the mathematical center coordinate
    ax.plot(center[0], center[1], marker='X', color='black', markersize=12, label='Center ($c_i$)')

    # Draw the radius line from the center to the boundary at a 45-degree angle
    angle = np.pi / 4
    edge_x = radius * np.cos(angle)
    edge_y = radius * np.sin(angle)
    ax.plot([center[0], edge_x], [center[1], edge_y], color='black', linestyle='-', linewidth=2.0)

    # Insert mathematical LaTeX annotations directly onto the coordinate plane
    ax.text(center[0] - 0.15, center[1] - 0.35, '$c_i$', fontsize=18, fontweight='bold')
    ax.text(edge_x / 2 - 0.2, edge_y / 2 + 0.2, '$r_i$', fontsize=18, fontweight='bold')
    
    # Label the regions
    ax.text(center[0] - 1.2, center[1] + 1.0, '$C_{in}$', fontsize=20, fontweight='bold', color='#1f77b4')
    ax.text(center[0] + radius + 0.4, center[1] + radius - 0.8, '$C_{out}$', fontsize=20, fontweight='bold', color='#d62728')

    # Remove the standard graph axes to create a clean, mathematical diagram look
    ax.set_aspect('equal')
    ax.axis('off')

    # Save the output file
    out_file = 'src/utility/coreset_partitioning.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Successfully generated {out_file}")

if __name__ == "__main__":
    generate_partition_diagram()
import numpy as np
import matplotlib.pyplot as plt

def generate_partition_diagram():
    np.random.seed(42)

    # variables
    center = np.array([0.0, 0.0])
    radius = 2.2

    # generate inner points
    inner_candidates = np.random.normal(loc=center, scale=0.9, size=(1000, 2))
    distances = np.linalg.norm(inner_candidates - center, axis=1)
    inner_points = inner_candidates[distances <= radius][:350]

    # generate outer points
    outer_candidates = np.random.normal(loc=center, scale=2.5, size=(1000, 2))
    outer_distances = np.linalg.norm(outer_candidates - center, axis=1)
    outer_points = outer_candidates[outer_distances > radius][:120]

    # initialize the a square plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # plot the outer points
    ax.scatter(outer_points[:, 0], outer_points[:, 1], c='#ff7f0e', alpha=0.5, s=40, label='$C_{out}$ (Outer Points)')

    # plot the inner points
    ax.scatter(inner_points[:, 0], inner_points[:, 1], c="#2792de", alpha=0.9, s=40, label='$C_{in}$ (Inner Points)')

    # draw the boundary radius
    circle = plt.Circle(center, radius, color='black', fill=False, linestyle='--', linewidth=2.0)
    ax.add_patch(circle)

    # plot the center
    ax.plot(center[0], center[1], marker='D', color='black', markersize=12, label='Center ($c_i$)')
    
    # label the regions
    ax.text(center[0] - 1.2, center[1] + 1.0, '$C_{in}$', fontsize=20, fontweight='bold', color="#175b8b")
    ax.text(center[0] + radius + 0.4, center[1] + radius - 0.8, '$C_{out}$', fontsize=20, fontweight='bold', color="#bd5e0a")

    # remove the graph axes
    ax.set_aspect('equal')
    ax.axis('off')

    # save the output file
    out_file = 'src/utility/coreset_partitioning.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Successfully generated {out_file}")

if __name__ == "__main__":
    generate_partition_diagram()
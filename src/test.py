import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from sklearn.cluster import KMeans

# Load and prepare data (same as your other script)
input_file = "input/birb.png"
img = imread(input_file)

# remove alpha channel
if img.shape[-1] == 4:
    img = img[:, :, :3]

if img.max() > 1.0: img = img / 255.0
flat_image = img.reshape(-1, 3)

np.random.seed(42)
calc_pixels = flat_image[np.random.choice(flat_image.shape[0], 10000, replace=False)]
bg_pixels = flat_image[np.random.choice(flat_image.shape[0], 50000, replace=False)] # Reduced to 50k to prevent crashing

print("Calculating centers...")
kmeans = KMeans(n_clusters=16, random_state=42, n_init="auto").fit(calc_pixels)
rgb_centers = kmeans.cluster_centers_

print("Rendering 3D Plot...")
# Initialize a 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the faint background points
ax.scatter(bg_pixels[:, 0], bg_pixels[:, 1], bg_pixels[:, 2], 
           c=bg_pixels, s=5, alpha=0.02, edgecolors='none')

# Plot the cluster centers
ax.scatter(rgb_centers[:, 0], rgb_centers[:, 1], rgb_centers[:, 2], 
           c='black', marker='D', s=100, edgecolors='white', linewidth=1.5, depthshade=False)

# Format the 3D axes
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# add a corner view
ax.view_init(elev=30, azim=45)

plt.title("3D RGB Color Space")
plt.savefig("plots/3d_test.png", dpi=300)
print("3D plot saved.")
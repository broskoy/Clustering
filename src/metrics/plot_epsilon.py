from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os




def generate_heatmap_q_vs_epsilon(csv_path, output_dir):
    print(f"Loading metrics from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Calculate the crucial Green Line ratio
    df['Cost_Ratio'] = df['Cost_Biased'] / df['Cost_Loyd']
    
    # Group by Q and Epsilon, taking the mean across the iterations
    grouped = df.groupby(['Budget', 'Epsilon'])['Cost_Ratio'].mean().reset_index()
    
    # Pivot the data into a 2D matrix format for the heatmap
    pivot_table = grouped.pivot(index='Budget', columns='Epsilon', values='Cost_Ratio')
    
    # Sort the Y-axis so the largest coreset is at the top
    pivot_table = pivot_table.sort_index(ascending=False)
    
    # Create the visual
    plt.figure(figsize=(10, 8))
    colors = ["#47c94b","#caf363" , "#ffdd60", "#f39f51", "#e16153"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_RdYlGn", colors)
    plt.imshow(pivot_table.values, cmap='RdYlGn_r', aspect='auto')
    
    # Add the text annotations inside the boxes
    for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
            val = pivot_table.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10)
    
    # Format axes
    plt.colorbar(label='Cost Ratio: Biased / Loyd')
    plt.xticks(ticks=np.arange(len(pivot_table.columns)), labels=pivot_table.columns)
    plt.yticks(ticks=np.arange(len(pivot_table.index)), labels=pivot_table.index)
    
    plt.xlabel('Epsilon (Radius Scaling Factor)')
    plt.ylabel('Coreset Size |Q|')
    plt.title('Impact |Q| vs Epsilon')
    
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "heatmap_epsilon.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {out_file}")




if __name__ == "__main__":
    metrics_file = "metrics/metrics.csv"
    output_directory = "metrics"
    generate_heatmap_q_vs_epsilon(metrics_file, output_directory)
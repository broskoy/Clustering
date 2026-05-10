import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_heatmap_cost(csv_path, output_dir):

    print(f"Loading metrics from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Calculate the crucial Green Line ratio
    df['Cost_Ratio'] = df['Cost_Biased'] / df['Cost_Loyd']
    
    # Group by Q_Budget and K_Value, taking the mean across the iterations
    grouped = df.groupby(['Q_Budget', 'K_Clusters'])['Cost_Ratio'].mean().reset_index()
    
    # Pivot the data into a 2D matrix format for the heatmap
    pivot_table = grouped.pivot(index='Q_Budget', columns='K_Clusters', values='Cost_Ratio')
    
    # Sort the Y-axis so the largest coreset is at the top
    pivot_table = pivot_table.sort_index(ascending=False)
    
    # Create the visual
    plt.figure(figsize=(12, 8))
    
    # vmin=1.0 ensures 1.0 is max green. 
    # vmax=2.5 sets a ceiling so extreme errors don't wash out the gradient.
    plt.imshow(pivot_table.values, cmap='RdYlGn_r', aspect='auto', vmin=1.0, vmax=2.5)
    
    # Add the text annotations inside the boxes
    for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
            val = pivot_table.values[i, j]
            if not np.isnan(val):
                # Dynamic text color based on background intensity for readability
                color = "white" if val > 1.8 else "black"
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight='bold')
    
    # Format axes
    plt.colorbar(label='Cost Ratio: Cost(P,C\') / Cost(P,C)')
    plt.xticks(ticks=np.arange(len(pivot_table.columns)), labels=pivot_table.columns)
    plt.yticks(ticks=np.arange(len(pivot_table.index)), labels=pivot_table.index)
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Coreset Size |Q|')
    plt.title('Cost Comparison: Biased Coreset vs Standard Loyd')
    
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"heatmap_cost_biased.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {out_file}")


def generate_heatmap_cost2(csv_path, output_dir):

    print(f"Loading metrics from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Calculate the crucial Green Line ratio
    df['Cost_Ratio'] = df['Cost_Uniform'] / df['Cost_Loyd']
    
    # Group by Q_Budget and K_Value, taking the mean across the iterations
    grouped = df.groupby(['Q_Budget', 'K_Clusters'])['Cost_Ratio'].mean().reset_index()
    
    # Pivot the data into a 2D matrix format for the heatmap
    pivot_table = grouped.pivot(index='Q_Budget', columns='K_Clusters', values='Cost_Ratio')
    
    # Sort the Y-axis so the largest coreset is at the top
    pivot_table = pivot_table.sort_index(ascending=False)
    
    # Create the visual
    plt.figure(figsize=(12, 8))
    
    # vmin=1.0 ensures 1.0 is max green. 
    # vmax=2.5 sets a ceiling so extreme errors don't wash out the gradient.
    plt.imshow(pivot_table.values, cmap='RdYlGn_r', aspect='auto', vmin=1.0, vmax=2.5)
    
    # Add the text annotations inside the boxes
    for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
            val = pivot_table.values[i, j]
            if not np.isnan(val):
                # Dynamic text color based on background intensity for readability
                color = "white" if val > 1.8 else "black"
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight='bold')
    
    # Format axes
    plt.colorbar(label='Cost Ratio: Cost(P,C\') / Cost(P,C)')
    plt.xticks(ticks=np.arange(len(pivot_table.columns)), labels=pivot_table.columns)
    plt.yticks(ticks=np.arange(len(pivot_table.index)), labels=pivot_table.index)
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Coreset Size |Q|')
    plt.title('Cost Comparison: Uniform Coreset vs Standard Loyd')
    
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"heatmap_cost_uniform.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {out_file}")



if __name__ == "__main__":
    generate_heatmap_cost("metrics/metrics.csv", "metrics/cost")
    generate_heatmap_cost2("metrics/metrics.csv", "metrics/cost")
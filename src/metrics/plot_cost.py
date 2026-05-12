import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os




def plot_biased_heatmap():

    df = pd.read_csv("metrics/metrics.csv")
    
    # Calculate the ratio
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
    
    out_file = os.path.join("metrics/cost", f"heatmap_cost_biased.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {out_file}")




def plot_uniform_heatmap():

    df = pd.read_csv("metrics/metrics.csv")
    
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
    
    out_file = os.path.join("metrics/cost", f"heatmap_cost_uniform.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {out_file}")




def plot_lines_k():
    df = pd.read_csv("metrics/metrics.csv")

     # Average the time across all Q budgets to show how scaling K impacts total runtime
    agg_k = df.groupby('K_Clusters')[['Cost_Loyd', 'Cost_Biased', 'Cost_Uniform']].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot all three lines on the same axis
    plt.plot(agg_k['K_Clusters'], agg_k['Cost_Loyd'], marker='o', label='Standard Lloyd', color='#d62728', linewidth=2.5)
    plt.plot(agg_k['K_Clusters'], agg_k['Cost_Biased'], marker='s', label='Biased Coreset', color='#1f77b4', linewidth=2.5)
    plt.plot(agg_k['K_Clusters'], agg_k['Cost_Uniform'], marker='^', label='Uniform Coreset', color='#2ca02c', linewidth=2.5)
    
    plt.title('Cost Comparison Scaling (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Cost')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    out_file_line = os.path.join("metrics/cost", 'cost_lines_k.png')
    plt.savefig(out_file_line, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file_line}")




def plot_lines_q():
    df = pd.read_csv("metrics/metrics.csv")

     # Average the time across all Q budgets to show how scaling K impacts total runtime
    agg_k = df.groupby('Q_Budget')[['Cost_Loyd', 'Cost_Biased', 'Cost_Uniform']].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot all three lines on the same axis
    plt.plot(agg_k['Q_Budget'], agg_k['Cost_Loyd'], marker='o', label='Standard Lloyd', color='#d62728', linewidth=2.5)
    plt.plot(agg_k['Q_Budget'], agg_k['Cost_Biased'], marker='s', label='Biased Coreset', color='#1f77b4', linewidth=2.5)
    plt.plot(agg_k['Q_Budget'], agg_k['Cost_Uniform'], marker='^', label='Uniform Coreset', color='#2ca02c', linewidth=2.5)
    
    plt.title('Cost Comparison Scaling |Q|')
    plt.xlabel('Coreset Size |Q|')
    plt.ylabel('Average Cost')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    out_file_line = os.path.join("metrics/cost", 'cost_lines_q.png')
    plt.savefig(out_file_line, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file_line}")




if __name__ == "__main__":

    plot_biased_heatmap()

    plot_uniform_heatmap()
    
    plot_lines_k()

    plot_lines_q()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os




def plot_single_heatmap(grouped, data_col, title, filename, cmap):
        # Pivot the data into a 2D matrix format
        pivot = grouped.pivot(index='Q_Budget', columns='K_Clusters', values=data_col)
        pivot = pivot.sort_index(ascending=False) # Largest Q at the top
        
        plt.figure(figsize=(12, 8))
        
        # Plot the heatmap
        plt.imshow(pivot.values, cmap=cmap, aspect='auto')
        
        # Add the text annotations inside the boxes
        min_val = np.nanmin(pivot.values)
        max_val = np.nanmax(pivot.values)
        
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    # Dynamic text color: Light text on dark backgrounds, Dark text on light backgrounds
                    normalized_val = (val - min_val) / (max_val - min_val + 1e-9)
                    color = "white" if normalized_val > 0.6 else "black"
                    plt.text(j, i, f"{val:.2f}s", ha="center", va="center", color=color, fontsize=10, fontweight='bold')
        
        # Format axes
        plt.colorbar(label='Execution Time (Seconds)')
        plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
        plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
        
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Coreset Size |Q|')
        plt.title(title)
        
        out_file = os.path.join("metrics/time", filename)
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_file}")




def generate_time_plots():
    print(f"Loading metrics from {"metrics/metrics.csv"}...")
    df = pd.read_csv("metrics/metrics.csv")
    
    # group by Q and K then average the execution times across iterations
    grouped = df.groupby(['Q_Budget', 'K_Clusters'])[['Time_Loyd', 'Time_Biased', 'Time_Uniform']].mean().reset_index()
    
    # Generate the isolated heatmaps with distinct color schemes
    plot_single_heatmap(grouped, 'Time_Loyd', 'Execution Time: Standard Lloyd (Full Dataset)', 'heatmap_time_loyd.png', 'Reds')
    plot_single_heatmap(grouped, 'Time_Biased', 'Execution Time: Biased Coreset (Our Algorithm)', 'heatmap_time_biased.png', 'Blues')
    plot_single_heatmap(grouped, 'Time_Uniform', 'Execution Time: Uniform Sampling', 'heatmap_time_uniform.png', 'Greens')

    
    # Average the time across all Q budgets to show how scaling K impacts total runtime
    agg_k = df.groupby('K_Clusters')[['Time_Loyd', 'Time_Biased', 'Time_Uniform']].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot all three lines on the same axis
    plt.plot(agg_k['K_Clusters'], agg_k['Time_Loyd'], marker='o', label='Lloyd (Full Dataset)', color='#d62728', linewidth=2.5)
    plt.plot(agg_k['K_Clusters'], agg_k['Time_Biased'], marker='s', label='Biased Coreset', color='#1f77b4', linewidth=2.5)
    plt.plot(agg_k['K_Clusters'], agg_k['Time_Uniform'], marker='^', label='Uniform Sampling', color='#2ca02c', linewidth=2.5)
    
    plt.title('Execution Time Comparison: Scaling Complexity (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Execution Time (Seconds)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    out_file_line = os.path.join("metrics/time", 'line_time_vs_k.png')
    plt.savefig(out_file_line, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file_line}")




if __name__ == "__main__":
    generate_time_plots()
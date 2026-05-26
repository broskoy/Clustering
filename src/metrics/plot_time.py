import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# the fully merged dataset 
global_df = None


def merge_data():
    global global_df

    # load the files
    df_loyd = pd.read_csv("metrics/metrics_loyd.csv")
    df_biased = pd.read_csv("metrics/metrics_biased.csv")
    df_uniform = pd.read_csv("metrics/metrics_uniform.csv")
    df_egb = pd.read_csv("metrics/metrics_egb.csv")
    df_lightweight = pd.read_csv("metrics/metrics_lightweight.csv")
    df_ranked = pd.read_csv("metrics/metrics_ranked.csv")

    # rename columns to prevent suffix collisions
    df_loyd = df_loyd.rename(columns={'Cost': 'Cost_Loyd', 'Time': 'Time_Loyd'})
    df_biased = df_biased.rename(columns={'Cost': 'Cost_Biased', 'Time': 'Time_Biased'})
    df_uniform = df_uniform.rename(columns={'Cost': 'Cost_Uniform', 'Time': 'Time_Uniform'})
    df_egb = df_egb.rename(columns={'Cost': 'Cost_Egb', 'Time': 'Time_Egb'})
    df_lightweight = df_lightweight.rename(columns={'Cost': 'Cost_Lightweight', 'Time': 'Time_Lightweight'})
    df_ranked = df_ranked.rename(columns={'Cost': 'Cost_Ranked', 'Time': 'Time_Ranked'})

    # define the common keys
    merge_keys = ['Dataset', 'Clusters', 'Budget', 'Iteration']

    # merge consecutively
    df_merged = pd.merge(df_loyd, df_biased, on=merge_keys)
    df_merged = pd.merge(df_merged, df_uniform, on=merge_keys)
    df_merged = pd.merge(df_merged, df_egb, on=merge_keys)
    df_merged = pd.merge(df_merged, df_lightweight, on=merge_keys)
    df_merged = pd.merge(df_merged, df_ranked, on=merge_keys)

    # filer for specific dataset
    df_merged = df_merged[df_merged['Dataset'] == 'uber']

    global_df = df_merged




def plot_biased_heatmap():
    df = global_df.copy()

    # Calculate the ratio
    df['Time_Ratio_Biased'] = df['Time_Loyd'] / df['Time_Biased']
    
    # group by Q and K then average the execution times across iterations
    grouped = df.groupby(['Budget', 'Clusters'])[['Time_Ratio_Biased']].mean().reset_index()
    
    # Pivot the data into a 2D matrix format
    pivot = grouped.pivot(index='Budget', columns='Clusters', values='Time_Ratio_Biased')
    pivot = pivot.sort_index(ascending=False) # Largest Q at the top
    
    plt.figure(figsize=(12, 8))
    
    # Plot the heatmap
    plt.imshow(pivot.values, cmap='Blues', aspect='auto')
        
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
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight='bold')
    
    # Format axes
    plt.colorbar(label='Execution Speedup')
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Coreset Size |Q|')
    plt.title('Biased Coreset Execution Time')
    
    out_file = os.path.join("metrics/time", 'time_heatmap_biased.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")



def plot_uniform_heatmap():
    df = global_df.copy()

    # Calculate the ratio
    df['Time_Ratio_Uniform'] = df['Time_Loyd'] / df['Time_Uniform']
    
    # group by Q and K then average the execution times across iterations
    grouped = df.groupby(['Budget', 'Clusters'])[['Time_Ratio_Uniform']].mean().reset_index()
    
    # Pivot the data into a 2D matrix format
    pivot = grouped.pivot(index='Budget', columns='Clusters', values='Time_Ratio_Uniform')
    pivot = pivot.sort_index(ascending=False) # Largest Q at the top
    
    plt.figure(figsize=(12, 8))
    
    # Plot the heatmap
    plt.imshow(pivot.values, cmap='Greens', aspect='auto')
        
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
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight='bold')
    
    # Format axes
    plt.colorbar(label='Execution Speedup')
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Coreset Size |Q|')
    plt.title('Biased Coreset Execution Time')
    
    out_file = os.path.join("metrics/time", 'time_heatmap_uniform.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")



def plot_lines_k():
    df = global_df.copy()

     # Average the time across all Q budgets to show how scaling K impacts total runtime
    agg_k = df.groupby('Clusters')[['Time_Loyd', 'Time_Biased', 'Time_Uniform', 'Time_Egb', 'Time_Lightweight', 'Time_Ranked']].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot all three lines on the same axis
    plt.plot(agg_k['Clusters'], agg_k['Time_Loyd'], marker='o', label='Standard Lloyd', color='#d62728', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Time_Biased'], marker='s', label='Biased Coreset', color='#1f77b4', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Time_Uniform'], marker='^', label='Uniform Coreset', color='#2ca02c', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Time_Egb'], marker='^', label='EGB Coreset', color='#eed142', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Time_Lightweight'], marker='^', label='Lightweight Coreset', color='#ffa43d', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Time_Ranked'], marker='^', label='Ranked Coreset', color="#42eed4", linewidth=2.5)
    
    plt.title('Execution Time Scaling (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Execution Time (Seconds)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    out_file_line = os.path.join("metrics/time", 'time_lines_k.png')
    plt.savefig(out_file_line, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file_line}")




def plot_lines_q():
    df = global_df.copy()

     # Average the time across all Q budgets to show how scaling K impacts total runtime
    agg_k = df.groupby('Budget')[['Time_Loyd', 'Time_Biased', 'Time_Uniform', 'Time_Egb', 'Time_Lightweight', 'Time_Ranked']].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot all three lines on the same axis
    plt.plot(agg_k['Budget'], agg_k['Time_Loyd'], marker='o', label='Standard Lloyd', color='#d62728', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Time_Biased'], marker='s', label='Biased Coreset', color='#1f77b4', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Time_Uniform'], marker='^', label='Uniform Coreset', color='#2ca02c', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Time_Egb'], marker='^', label='EGB Coreset', color='#eed142', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Time_Lightweight'], marker='^', label='Lightweight Coreset', color='#ffa43d', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Time_Ranked'], marker='^', label='Ranked Coreset', color="#42eed4", linewidth=2.5)
    
    plt.title('Execution Time Scaling |Q|')
    plt.xlabel('Coreset Size |Q|')
    plt.ylabel('Average Execution Time (Seconds)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    out_file_line = os.path.join("metrics/time", 'time_lines_q.png')
    plt.savefig(out_file_line, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file_line}")




if __name__ == "__main__":
    merge_data()

    plot_biased_heatmap()

    plot_uniform_heatmap()

    plot_lines_k()

    plot_lines_q()
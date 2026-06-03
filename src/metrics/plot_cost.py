import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
    df_kchen = pd.read_csv("metrics/metrics_kchen.csv")


    # rename columns to prevent suffix collisions
    df_loyd = df_loyd.rename(columns={'Cost': 'Cost_Loyd', 'Time': 'Time_Loyd'})
    df_biased = df_biased.rename(columns={'Cost': 'Cost_Biased', 'Time': 'Time_Biased'})
    df_uniform = df_uniform.rename(columns={'Cost': 'Cost_Uniform', 'Time': 'Time_Uniform'})
    df_egb = df_egb.rename(columns={'Cost': 'Cost_Egb', 'Time': 'Time_Egb'})
    df_lightweight = df_lightweight.rename(columns={'Cost': 'Cost_Lightweight', 'Time': 'Time_Lightweight'})
    df_ranked = df_ranked.rename(columns={'Cost': 'Cost_Ranked', 'Time': 'Time_Ranked'})
    df_kchen = df_kchen.rename(columns={'Cost': 'Cost_Kchen', 'Time': 'Time_Kchen'})

    # define the common keys
    merge_keys = ['Dataset', 'Clusters', 'Budget', 'Iteration']

    # merge consecutively
    df_merged = pd.merge(df_loyd, df_biased, on=merge_keys)
    df_merged = pd.merge(df_merged, df_uniform, on=merge_keys)
    df_merged = pd.merge(df_merged, df_egb, on=merge_keys)
    df_merged = pd.merge(df_merged, df_lightweight, on=merge_keys)
    df_merged = pd.merge(df_merged, df_ranked, on=merge_keys)
    df_merged = pd.merge(df_merged, df_kchen, on=merge_keys)

    # filer for specific dataset
    df_merged = df_merged[df_merged['Dataset'] == 'balloons']

    global_df = df_merged

    



def plot_biased_heatmap():

    df = global_df.copy()
    
    # Calculate the ratio
    df['Cost_Ratio'] = df['Cost_Biased'] / df['Cost_Loyd']
    
    # Group to take the mean across the iterations
    grouped = df.groupby(['Budget', 'Clusters'])['Cost_Ratio'].mean().reset_index()
    
    # Pivot the data into a 2D matrix format for the heatmap
    pivot_table = grouped.pivot(index='Budget', columns='Clusters', values='Cost_Ratio')
    
    # Sort the Y-axis so the largest coreset is at the top
    pivot_table = pivot_table.sort_index(ascending=False)
    
    # Create the visual
    plt.figure(figsize=(12, 8))
    
    # color gradient from green to red
    colors = ["#47c94b","#caf363" , "#ffdd60", "#f39f51", "#e16153"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_RdYlGn", colors)
    plt.imshow(pivot_table.values, cmap=custom_cmap, aspect='auto', vmin=1.0, vmax=2.5)
    
    # Add the text annotations inside the boxes
    for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
            val = pivot_table.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    # Format axes
    plt.colorbar(label='Cost Ratio: Cost(P,C\') / Cost(P,C)')
    plt.xticks(ticks=np.arange(len(pivot_table.columns)), labels=pivot_table.columns)
    plt.yticks(ticks=np.arange(len(pivot_table.index)), labels=pivot_table.index)
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Coreset Size |Q|')
    plt.title('Cost Comparison: Biased Coreset vs Standard Loyd')
    
    out_file = os.path.join("metrics/cost", f"cost_heatmap_biased.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")




def plot_uniform_heatmap():

    df = global_df.copy()
    
    # Calculate the ratio
    df['Cost_Ratio'] = df['Cost_Uniform'] / df['Cost_Loyd']
    
    # Group to take the mean across the iterations
    grouped = df.groupby(['Budget', 'Clusters'])['Cost_Ratio'].mean().reset_index()
    
    # Pivot the data into a 2D matrix format for the heatmap
    pivot_table = grouped.pivot(index='Budget', columns='Clusters', values='Cost_Ratio')
    
    # sort the y axis so the largest coreset is at the top
    pivot_table = pivot_table.sort_index(ascending=False)
    
    # Create the visual
    plt.figure(figsize=(12, 8))
     
    # color gradient from green to red
    colors = ["#47c94b","#caf363" , "#ffdd60", "#f39f51", "#e16153"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_RdYlGn", colors)
    plt.imshow(pivot_table.values, cmap=custom_cmap, aspect='auto', vmin=1.0, vmax=2.0)
    
    # Add the text annotations inside the boxes
    for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
            val = pivot_table.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    # Format axes
    plt.colorbar(label='Cost Ratio: Cost(P,C\') / Cost(P,C)')
    plt.xticks(ticks=np.arange(len(pivot_table.columns)), labels=pivot_table.columns)
    plt.yticks(ticks=np.arange(len(pivot_table.index)), labels=pivot_table.index)
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Coreset Size |Q|')
    plt.title('Cost Comparison: Uniform Coreset vs Standard Loyd')
    
    out_file = os.path.join("metrics/cost", f"cost_heatmap_uniform.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")




def plot_lines_k():

    df = global_df.copy()

    # Average the time across all Q budgets to show how scaling K impacts total runtime
    agg_k = df.groupby('Clusters')[['Cost_Loyd', 'Cost_Biased', 'Cost_Uniform', 'Cost_Egb', 'Cost_Lightweight', 'Cost_Ranked', 'Cost_Kchen']].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot all three lines on the same axis
    plt.plot(agg_k['Clusters'], agg_k['Cost_Loyd'], marker='o', label='Standard Lloyd', color='#d62728', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Cost_Biased'], marker='s', label='Biased Coreset', color='#1f77b4', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Cost_Uniform'], marker='^', label='Uniform Coreset', color='#2ca02c', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Cost_Egb'], marker='^', label='EGB Coreset', color='#eed142', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Cost_Lightweight'], marker='^', label='Lightweight Coreset', color='#ffa43d', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Cost_Ranked'], marker='^', label='Ranked Coreset', color="#42eed4", linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Cost_Kchen'], marker='^', label='Kchen Coreset', color="#7842ee", linewidth=2.5)


    plt.title('Cost Comparison Scaling (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Cost')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    out_file_line = os.path.join("metrics/cost", 'cost_lines_k.png')
    plt.savefig(out_file_line, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file_line}")




def plot_lines_q():
    df = global_df.copy()

    # Average the time across all Q budgets
    agg_k = df.groupby('Budget')[['Cost_Loyd', 'Cost_Biased', 'Cost_Uniform', 'Cost_Egb', 'Cost_Lightweight', 'Cost_Ranked', 'Cost_Kchen']].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot all three lines on the same axis
    plt.plot(agg_k['Budget'], agg_k['Cost_Loyd'], marker='o', label='Standard Lloyd', color='#d62728', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Cost_Biased'], marker='s', label='Biased Coreset', color='#1f77b4', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Cost_Uniform'], marker='^', label='Uniform Coreset', color='#2ca02c', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Cost_Egb'], marker='^', label='EGB Coreset', color="#eed142", linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Cost_Lightweight'], marker='^', label='Lightweight Coreset', color="#ffa43d", linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Cost_Ranked'], marker='^', label='Ranked Coreset', color="#42eed4", linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Cost_Kchen'], marker='^', label='Kchen Coreset', color="#7842ee", linewidth=2.5)

    plt.title('Cost Comparison Scaling |Q|')
    plt.xlabel('Coreset Size |Q|')
    plt.ylabel('Average Cost')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    out_file_line = os.path.join("metrics/cost", 'cost_lines_q.png')
    plt.savefig(out_file_line, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file_line}")




if __name__ == "__main__":
    merge_data()

    plot_biased_heatmap()

    plot_uniform_heatmap()
    
    plot_lines_k()

    plot_lines_q()
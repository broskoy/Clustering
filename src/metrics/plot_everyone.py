import pandas as pd
import numpy as np
import seaborn as sns
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
    # df_merged = df_merged[df_merged['Dataset'] == 'uber']

    global_df = df_merged




def cost_plot_lines_k():

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
    
    out_file_line = os.path.join("metrics/everyone", 'cost_lines_k.png')
    plt.savefig(out_file_line, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file_line}")




def cost_plot_lines_q():
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
    
    out_file_line = os.path.join("metrics/everyone", 'cost_lines_q.png')
    plt.savefig(out_file_line, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file_line}")




def time_plot_lines_k():
    df = global_df.copy()

     # Average the time across all Q budgets to show how scaling K impacts total runtime
    agg_k = df.groupby('Clusters')[['Time_Loyd', 'Time_Biased', 'Time_Uniform', 'Time_Egb', 'Time_Lightweight', 'Time_Ranked', 'Time_Kchen']].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot all three lines on the same axis
    plt.plot(agg_k['Clusters'], agg_k['Time_Loyd'], marker='o', label='Standard Lloyd', color='#d62728', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Time_Biased'], marker='s', label='Biased Coreset', color='#1f77b4', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Time_Uniform'], marker='^', label='Uniform Coreset', color='#2ca02c', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Time_Egb'], marker='^', label='EGB Coreset', color='#eed142', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Time_Lightweight'], marker='^', label='Lightweight Coreset', color='#ffa43d', linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Time_Ranked'], marker='^', label='Ranked Coreset', color="#42eed4", linewidth=2.5)
    plt.plot(agg_k['Clusters'], agg_k['Time_Kchen'], marker='^', label='Kchen Coreset', color="#7842ee", linewidth=2.5)

    plt.title('Execution Time Scaling (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Execution Time (Seconds)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    out_file_line = os.path.join("metrics/everyone", 'time_lines_k.png')
    plt.savefig(out_file_line, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file_line}")




def time_plot_lines_q():
    df = global_df.copy()

    # Average the time across all Q budgets to show how scaling K impacts total runtime
    agg_k = df.groupby('Budget')[['Time_Loyd', 'Time_Biased', 'Time_Uniform', 'Time_Egb', 'Time_Lightweight', 'Time_Ranked', 'Time_Kchen']].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot all three lines on the same axis
    plt.plot(agg_k['Budget'], agg_k['Time_Loyd'], marker='o', label='Standard Lloyd', color='#d62728', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Time_Biased'], marker='s', label='Biased Coreset', color='#1f77b4', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Time_Uniform'], marker='^', label='Uniform Coreset', color='#2ca02c', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Time_Egb'], marker='^', label='EGB Coreset', color='#eed142', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Time_Lightweight'], marker='^', label='Lightweight Coreset', color='#ffa43d', linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Time_Ranked'], marker='^', label='Ranked Coreset', color="#42eed4", linewidth=2.5)
    plt.plot(agg_k['Budget'], agg_k['Time_Kchen'], marker='^', label='Kchen Coreset', color="#7842ee", linewidth=2.5)

    plt.title('Execution Time Scaling |Q|')
    plt.xlabel('Coreset Size |Q|')
    plt.ylabel('Average Execution Time (Seconds)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    out_file_line = os.path.join("metrics/everyone", 'time_lines_q.png')
    plt.savefig(out_file_line, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file_line}")




if __name__ == "__main__":
    merge_data()

    cost_plot_lines_k()

    cost_plot_lines_q()

    time_plot_lines_k()

    time_plot_lines_q()
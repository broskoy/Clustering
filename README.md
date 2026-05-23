## Overview

This repository evaluates the performance of different of clustering algorithms. It is specifically designed to compare the execution time and error costs of standard Lloyd's $k$-means against biased and uniform coreset sampling techniques. To ensure data integrity across multiple executions, the pipeline separates the logging for each algorithm into independent metric files. These are later merged in memory for global performance comparisons.

<br><br>

## Getting Started

1. Ensure you have Python installed and configure your environment.
2. Install the necessary packages:
`pip install -r requirements.txt`
3. Place your target datasets into the `input/` directory.
4. Execute the main runner to begin the testing sequence:
`python src/run.py`

<br><br>

## Workflow Pipeline

**Data Ingestion:** Tabular data and images are placed in the `input/` directory and processed by the loader utility.  
**Execution:** The primary runner script tests various cluster counts ($k$) and coreset memory budgets ($Q$).  
**Metric Logging:** Performance measurements (Cost and Time) are stored in isolated CSVs within the `metrics/` directory.  
**Visualization:** Scatterplots, projections, and reconstructed images are automatically generated and saved to the `output/` and `metrics/` directories. (Only 2d, 3d and 6d data works)  

<br><br>

## Standardized CSV Data Schema

To ensure that results from all algorithms can be merged seamlessly for global comparisons, every algorithm must log its metrics using the exact same CSV column headers. The plotting scripts use the first four columns as shared keys to perfectly align the independent runs.

| Column Name | Description | Data Type |
| --- | --- | --- |
| **Dataset** | The exact name of the input file (excluding the extension). | String |
| **Clusters** | The number of clusters configured for the trial (k). | Integer |
| **Budget** | The coreset memory size or sampling limit (Q) | Integer |
| **Iteration** | The specific trial run number (used to average out variances). | Integer |
| **Cost** | The final calculated sum of squared errors. | Float |
| **Time** | The total execution time of the algorithm in seconds. | Float |

**Merging Note:** The plotting pipeline dynamically renames the `Cost` and `Time` columns based on the file name (changing `Cost` to `Cost_Biased`) before joining them. You do need to hardcode the algorithm's name into the codebase, but no changes need to be applied to the CSV.

<br><br>

## Repository Structure

Clustering/  
│  
├── src/                  # source code  
│   │  
│   ├── utility/          # this is used for cleaning / creating data  
│   ├── metrics/          # this contains all of the metrics plotting  
│   ├── coreset.py        # finds a biased coreset  
│   ├── loader.py         # converting datasets to raw points and metadata  
│   ├── visualize.py      # handles drawing the graphs and scatter plots  
│   └── run.py            # run all of the tests  
│  
├── input/                # initial datasets  
│  
├── output/               # solution centers visualized  
│  
├── metrics/              # here are measurements of performance  
│  
├── requirements.txt      # list of dependencies  
└── README.md             # project overview 
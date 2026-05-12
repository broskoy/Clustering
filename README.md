



## Project Structure

Clustering/  
│  
├── src/                  # source code  
|   |
│   ├── utility/          # this is used for cleaning / creating data
│   ├── metrics/          # this containg all of the metrics plotting
│   ├── coreset.py        # finds a biased coreset
│   ├── loader.py         # converting datasets to raw points and metadata
│   ├── visualize.py      # handles drawing the graphs and scatter plots
│   └── run.py            # run all of the tests  
│  
├── input/                # initial datasets  
|  
├── output/               # solution centers visualized  
|
├── metrics/              # here are measurements of performance
│  
├── documentation.md      # detailed explanations here  
├── requirements.txt      # list of dependencies  
└── README.md             # project overview 
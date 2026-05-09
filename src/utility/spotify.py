import pandas as pd
import os

def clean_spotify_data(input_path, output_path):
    print(f"Reading raw data from {input_path}...")
    
    # The 6 dimensions are all natively between 0.0 and 1.0
    target_columns = [
        "danceability", 
        "valence", 
        "energy", 
        "acousticness", 
        "instrumentalness", 
        "speechiness"
    ]
    
    try:
        # usecols saves memory by only loading the columns we want
        df = pd.read_csv(input_path, usecols=target_columns)
    except ValueError:
        print("Header mismatch, loading all and filtering...")
        df = pd.read_csv(input_path)
        df = df[target_columns]

    initial_count = len(df)
    
    # Drop any rows with missing values to ensure K-Means does not crash
    df = df.dropna()
    final_count = len(df)

    print(f"Dropped {initial_count - final_count} invalid rows.")
    print(f"Final dataset size: {final_count} tracks.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save without index to keep it purely mathematical for the loader
    df.to_csv(output_path, index=False)
    print(f"Cleaned 6D data saved to {output_path}")

if __name__ == "__main__":
    # Update these paths based on your actual file locations
    raw_file = "utility/spotify-raw.csv"
    clean_file = "input/real/spotify.csv"
    
    if os.path.exists(raw_file):
        clean_spotify_data(raw_file, clean_file)
    else:
        print(f"Error: {raw_file} not found. Please check your path.")
import pandas as pd
import os




def clean_uber_data(input_path, output_path):
    print(f"Reading raw data from {input_path}...")
    
    try:
        df = pd.read_csv(input_path, usecols=["Lat", "Lon"])
    except ValueError:
        print("Header mismatch, attempting positional load...")
        df = pd.read_csv(input_path)
        df = df[["Lat", "Lon"]]

    initial_count = len(df)
    
    # Drop missing values
    df = df.dropna()

    # Apply the geographic bounding box
    df = df[
        (df['Lat'] >= 40.65) & (df['Lat'] <= 40.85) &
        (df['Lon'] >= -74.05) & (df['Lon'] <= -73.85)
    ]
    
    final_count = len(df)
    print(f"Dropped {initial_count - final_count} invalid or out-of-bounds rows.")
    print(f"Final dataset size: {final_count} points.")

    # Flip the columns to (X, Y) which is (Lon, Lat)
    df = df[["Lon", "Lat"]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save without index or headers
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")




if __name__ == "__main__":
    # file locations
    raw_file = "utility/uber-raw.csv"
    clean_file = "input/real/uber.csv"
    
    if os.path.exists(raw_file):
        clean_uber_data(raw_file, clean_file)
    else:
        print(f"Error: {raw_file} not found. Please check your path.")
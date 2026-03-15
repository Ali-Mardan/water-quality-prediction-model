import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess_data(target_file, landsat_file, terraclimate_file):
    """
    Loads datsets and merges them into a single dataframe, preparing them
    for time-series sequence generation by resampling.
    """
    print("Loading datasets...")
    target_df = pd.read_csv(target_file)
    target_df['Sample Date'] = pd.to_datetime(target_df['Sample Date'], format='mixed')
    
    landsat_df = pd.read_csv(landsat_file)
    landsat_df['Sample Date'] = pd.to_datetime(landsat_df['Sample Date'], format='mixed')
    
    terraclimate_df = pd.read_csv(terraclimate_file)
    terraclimate_df['Sample Date'] = pd.to_datetime(terraclimate_df['Sample Date'], format='mixed')

    print("Merging datasets on Latitude, Longitude, and Sample Date...")
    merged_df = pd.merge(target_df, landsat_df, on=['Latitude', 'Longitude', 'Sample Date'], how='inner')
    merged_df = pd.merge(merged_df, terraclimate_df, on=['Latitude', 'Longitude', 'Sample Date'], how='inner')
    
    print("Creating Location ID based on Lat/Lon...")
    merged_df['Loc_ID'] = merged_df['Latitude'].astype(str) + "_" + merged_df['Longitude'].astype(str)

    return merged_df

def create_sequences(df, target_col, sequence_length=3, resample_freq='MS'):
    """
    Groups data by Location ID, resamples to a regular frequency to handle 
    satellite gaps, interpolates missing values, and creates sliding window sequences.
    
    Args:
        df: Input dataframe containing 'Loc_ID', 'Sample Date', and target columns.
        target_col: The column name to predict.
        sequence_length: Number of time steps to look back (e.g., 3 months).
        resample_freq: Pandas offset string. 'MS' is Month Start.
    """
    print(f"Resampling data to '{resample_freq}' and creating sequences of length {sequence_length}...")
    X, y = [], []
    
    # Sort just in case
    df = df.sort_values(by=['Loc_ID', 'Sample Date'])
    
    # Identify all feature columns (excluding identifiers)
    exclude_cols = ['Loc_ID', 'Latitude', 'Longitude', 'Sample Date']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Ensure target_col is the first feature so we can easily grab its index for y
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    feature_cols = [target_col] + feature_cols
    
    loc_groups = df.groupby('Loc_ID')
    
    for loc_id, group_df in loc_groups:
        group_df = group_df.set_index('Sample Date')
        
        # Resample to regular intervals. This is CRITICAL for time-series models
        # It handles the irregular gaps in satellite overpasses.
        resampled_df = group_df[feature_cols].resample(resample_freq).mean()
        
        # Interpolate missing values created by resampling gaps
        resampled_df = resampled_df.interpolate(method='linear').ffill().bfill()
        
        if len(resampled_df) <= sequence_length:
            continue
            
        values = resampled_df.values
        
        # Create sliding windows
        for i in range(len(values) - sequence_length):
            seq_x = values[i : i + sequence_length]
            seq_y = values[i + sequence_length, 0] # Assuming target_col is index 0
            
            X.append(seq_x)
            y.append(seq_y)
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    target_file = 'water_quality_training_dataset.csv'
    landsat_file = 'landsat_features_training.csv'
    terraclimate_file = 'terraclimate_features_training.csv'
    
    df = load_and_preprocess_data(target_file, landsat_file, terraclimate_file)
    
    # We take the first available target column as an example
    print("Columns available:", df.columns.tolist())
    target_column = df.columns[3] # Usually Dissolved Oxygen or something similar
    
    if target_column in df.columns:
        X, y = create_sequences(df, target_col=target_column, sequence_length=3, resample_freq='MS')
        print(f"Sequence Data Shape - X: {X.shape}, y: {y.shape}")
        
        # Save as numpy arrays for the PyTorch model
        np.save('X_sequences.npy', X)
        np.save('y_targets.npy', y)
        print("Saved sequences to X_sequences.npy and y_targets.npy")
    else:
        print(f"Target column '{target_column}' not found. Please update the target_column variable.")

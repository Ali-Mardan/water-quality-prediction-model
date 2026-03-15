import pandas as pd
import numpy as np
import os

def calculate_indices(df):
    """
    Calculate remote sensing indices from Landsat bands.
    """
    eps = 1e-10
    
    # Existing indices
    if 'nir' in df.columns and 'swir16' in df.columns:
        df['NDMI'] = (df['nir'] - df['swir16']) / (df['nir'] + df['swir16'] + eps)
    
    if 'green' in df.columns and 'swir16' in df.columns:
        df['MNDWI'] = (df['green'] - df['swir16']) / (df['green'] + df['swir16'] + eps)
    
    # New scientific indices
    # NDVI (Vegetation cover)
    if 'nir' in df.columns and 'red' in df.columns:
        df['NDVI'] = (df['nir'] - df['red']) / (df['nir'] + df['red'] + eps)
    
    # NDTI (Turbidity proxy)
    if 'red' in df.columns and 'green' in df.columns:
        df['NDTI'] = (df['red'] - df['green']) / (df['red'] + df['green'] + eps)
        
    # Salinity Indices
    if 'blue' in df.columns and 'red' in df.columns:
        df['SI'] = np.sqrt(df['blue'] * df['red'])
        df['SI1'] = np.sqrt(df['blue']**2 + df['red']**2)
        
    if 'blue' in df.columns and 'green' in df.columns and 'red' in df.columns:
        df['SI2'] = np.sqrt(df['blue']**2 + df['green']**2 + df['red']**2)
        
    # Lithology / Carbonate proxy (SWIR ratios)
    if 'swir16' in df.columns and 'swir22' in df.columns:
        df['SWIR_Ratio'] = df['swir16'] / (df['swir22'] + eps)
        
    return df

def main():
    # File paths
    landsat_file = 'expanded_landsat_features_training.csv'
    checkpoint_file = 'expanded_landsat_checkpoint.csv'
    climate_file = 'expanded_terraclimate_features_training.csv'
    targets_file = 'water_quality_training_dataset.csv'
    output_file = 'final_training_data_enriched.csv'
    
    # Check if files exist
    if not os.path.exists(landsat_file) and os.path.exists(checkpoint_file):
        print(f"Using checkpoint file for Landsat: {checkpoint_file}")
        landsat_file = checkpoint_file
    for f in [landsat_file, climate_file, targets_file]:
        if not os.path.exists(f):
            print(f"Warning: {f} not found. Some features will be missing.")
    
    # Load and process Landsat
    if os.path.exists(landsat_file):
        landsat_df = pd.read_csv(landsat_file)
        landsat_df = calculate_indices(landsat_df)
        print("Calculated spectral indices.")
    else:
        landsat_df = pd.DataFrame()
        
    # Load Climate
    if os.path.exists(climate_file):
        climate_df = pd.read_csv(climate_file)
        # We can add climate-specific engineering here (e.g. T_range)
        climate_df['T_range'] = climate_df['tmax'] - climate_df['tmin']
        print("Processed climate features.")
    else:
        climate_df = pd.DataFrame()
        
    # Load Targets
    targets_df = pd.read_csv(targets_file)
    
    # Merge datasets
    # Standardize join columns
    join_cols = ['Latitude', 'Longitude', 'Sample Date']
    
    merged_df = targets_df.copy()
    
    if not landsat_df.empty:
        merged_df = pd.merge(merged_df, landsat_df, on=join_cols, how='left')
        
    if not climate_df.empty:
        merged_df = pd.merge(merged_df, climate_df, on=join_cols, how='left')
        
    # Temporal features
    merged_df['Sample Date'] = pd.to_datetime(merged_df['Sample Date'], dayfirst=True)
    merged_df['Month'] = merged_df['Sample Date'].dt.month
    merged_df['DayOfYear'] = merged_df['Sample Date'].dt.dayofyear
    
    # Clean up
    print(f"Merged dataset shape: {merged_df.shape}")
    
    # Save enriched dataset
    merged_df.to_csv(output_file, index=False)
    print(f"Saved enriched training data to {output_file}")

if __name__ == "__main__":
    main()

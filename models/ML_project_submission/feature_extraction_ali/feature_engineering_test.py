import pandas as pd
import numpy as np
import os

def calculate_indices(df):
    """
    Calculate remote sensing indices from Landsat bands.
    """
    eps = 1e-10
    
    # NDVI (Vegetation cover)
    if 'nir' in df.columns and 'red' in df.columns:
        df['NDVI'] = (df['nir'] - df['red']) / (df['nir'] + df['red'] + eps)
    
    # NDTI (Turbidity proxy)
    if 'red' in df.columns and 'green' in df.columns:
        df['NDTI'] = (df['red'] - df['green']) / (df['red'] + df['green'] + eps)
        
    # Salinity Indices
    if 'blue' in df.columns and 'green' in df.columns and 'red' in df.columns:
        df['SI2'] = np.sqrt(df['blue']**2 + df['green']**2 + df['red']**2)
        
    # SWIR ratios
    if 'swir16' in df.columns and 'swir22' in df.columns:
        df['MNDWI'] = (df['green'] - df['swir16']) / (df['green'] + df['swir16'] + eps)
        df['NDMI'] = (df['nir'] - df['swir16']) / (df['nir'] + df['swir16'] + eps)
        df['SWIR_Ratio'] = df['swir16'] / (df['swir22'] + eps)
        
    return df

def main():
    # File paths
    landsat_file = 'expanded_landsat_features_test.csv'
    climate_file = 'expanded_terraclimate_features_test.csv'
    template_file = 'submission_template.csv'
    output_file = 'final_test_data_enriched.csv'
    
    # Load and process Landsat
    landsat_df = pd.read_csv(landsat_file)
    landsat_df = calculate_indices(landsat_df)
    print("Calculated spectral indices for test set.")
        
    # Load Climate
    climate_df = pd.read_csv(climate_file)
    climate_df['T_range'] = climate_df['tmax'] - climate_df['tmin']
    print("Processed climate features for test set.")
        
    # Load Template
    template_df = pd.read_csv(template_file)
    
    # Merge datasets
    join_cols = ['Latitude', 'Longitude', 'Sample Date']
    
    merged_df = template_df.copy()
    
    # Drop existing target cols if they exist but are empty
    for col in ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']:
        if col in merged_df.columns:
            merged_df = merged_df.drop(columns=[col])

    merged_df = pd.merge(merged_df, landsat_df, on=join_cols, how='left')
    merged_df = pd.merge(merged_df, climate_df, on=join_cols, how='left')
        
    # Temporal features
    merged_df['Sample Date'] = pd.to_datetime(merged_df['Sample Date'], dayfirst=True)
    merged_df['Month'] = merged_df['Sample Date'].dt.month
    merged_df['DayOfYear'] = merged_df['Sample Date'].dt.dayofyear
    
    print(f"Merged test dataset shape: {merged_df.shape}")
    
    # Save enriched dataset
    merged_df.to_csv(output_file, index=False)
    print(f"Saved enriched test data to {output_file}")

if __name__ == "__main__":
    main()

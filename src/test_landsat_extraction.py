import pandas as pd
from extract_expanded_landsat import compute_Landsat_values
import pystac_client
import planetary_computer as pc
import os

def main():
    input_file = 'water_quality_training_dataset.csv'
    df = pd.read_csv(input_file).head(5)
    print(f"Testing Landsat extraction on first 5 samples...")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    results = df.apply(lambda row: compute_Landsat_values(row, catalog), axis=1)
    
    # Add coordinates and date for verification
    results['Latitude'] = df['Latitude']
    results['Longitude'] = df['Longitude']
    results['Sample Date'] = df['Sample Date']
    
    print("Test Results:")
    print(results)
    
    results.to_csv('test_landsat_features.csv', index=False)
    print("Test results saved to test_landsat_features.csv")

if __name__ == "__main__":
    main()

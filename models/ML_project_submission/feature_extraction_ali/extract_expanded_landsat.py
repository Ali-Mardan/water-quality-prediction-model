import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load
from datetime import datetime
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")

def compute_Landsat_values(row, catalog):
    lat = row['Latitude']
    lon = row['Longitude']
    # Use coerce to handle potential date format issues
    sample_date = pd.to_datetime(row['Sample Date'], dayfirst=True, errors='coerce')
    
    if pd.isnull(sample_date):
        return pd.Series({
            "nir": np.nan, "red": np.nan, "blue": np.nan, "green": np.nan, 
            "swir16": np.nan, "swir22": np.nan
        })

    # Buffer size for ~100m 
    bbox_size = 0.00089831  
    bbox = [
        lon - bbox_size / 2,
        lat - bbox_size / 2,
        lon + bbox_size / 2,
        lat + bbox_size / 2
    ]

    try:
        # Search for Landsat scenes
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=bbox,
            datetime="2011-01-01/2015-12-31",
            query={"eo:cloud_cover": {"lt": 10}},
        )
        
        items = search.item_collection()

        if not items:
            return pd.Series({
                "nir": np.nan, "red": np.nan, "blue": np.nan, "green": np.nan, 
                "swir16": np.nan, "swir22": np.nan
            })

        # Convert sample date to UTC
        sample_date_utc = sample_date.tz_localize("UTC") if sample_date.tzinfo is None else sample_date.tz_convert("UTC")

        # Pick the item closest to the sample date
        items = sorted(
            items,
            key=lambda x: abs(pd.to_datetime(x.properties["datetime"]).tz_convert("UTC") - sample_date_utc)
        )
        selected_item = pc.sign(items[0])

        # Load expanded bands
        bands_of_interest = ["red", "blue", "green", "nir08", "swir16", "swir22"]
        data = stac_load([selected_item], bands=bands_of_interest, bbox=bbox).isel(time=0)

        # Compute median values for each band
        results = {}
        # Mapping standard names to loaded band names
        band_map = {
            "red": "red",
            "blue": "blue",
            "green": "green",
            "nir": "nir08",
            "swir16": "swir16",
            "swir22": "swir22"
        }
        
        for key, band_name in band_map.items():
            val = float(data[band_name].astype("float").median(skipna=True).values)
            results[key] = val if val != 0 else np.nan
            
        return pd.Series(results)
    
    except Exception as e:
        return pd.Series({
            "nir": np.nan, "red": np.nan, "blue": np.nan, "green": np.nan, 
            "swir16": np.nan, "swir22": np.nan
        })

from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    input_file = 'water_quality_training_dataset.csv'
    output_file = 'expanded_landsat_features_training.csv'
    checkpoint_file = 'expanded_landsat_checkpoint.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples from {input_file}")

    # Check if we have a checkpoint
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        processed_count = len(checkpoint_df)
        print(f"Resuming from checkpoint: {processed_count} samples already processed.")
        df_to_process = df.iloc[processed_count:]
        all_results = checkpoint_df.to_dict('records')
    else:
        df_to_process = df
        all_results = []

    if df_to_process.empty:
        print("All samples already processed.")
        return

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    batch_size = 50
    num_threads = 12 # Increased to speed up extraction (safe limit)
    
    print(f"Starting extraction with {num_threads} threads...")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Process in chunks to save checkpoints
        for i in range(0, len(df_to_process), batch_size):
            chunk = df_to_process.iloc[i : i + batch_size]
            futures = {executor.submit(compute_Landsat_values, row, catalog): row for _, row in chunk.iterrows()}
            
            chunk_results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {i//batch_size + 1}"):
                row_orig = futures[future]
                try:
                    res = future.result()
                    # Add metadata
                    res['Latitude'] = row_orig['Latitude']
                    res['Longitude'] = row_orig['Longitude']
                    res['Sample Date'] = row_orig['Sample Date']
                    chunk_results.append(res.to_dict())
                except Exception as e:
                    print(f"Error processing row: {e}")
                    # Create empty entry on error
                    res = pd.Series({
                        "nir": np.nan, "red": np.nan, "blue": np.nan, "green": np.nan, 
                        "swir16": np.nan, "swir22": np.nan,
                        "Latitude": row_orig['Latitude'],
                        "Longitude": row_orig['Longitude'],
                        "Sample Date": row_orig['Sample Date']
                    })
                    chunk_results.append(res.to_dict())
            
            all_results.extend(chunk_results)
            
            # Update checkpoint
            pd.DataFrame(all_results).to_csv(checkpoint_file, index=False)
            
    # Final save
    final_df = pd.DataFrame(all_results)
    cols = ['Latitude', 'Longitude', 'Sample Date', 'red', 'blue', 'green', 'nir', 'swir16', 'swir22']
    final_df = final_df[cols]
    final_df.to_csv(output_file, index=False)
    print(f"Saved all expanded features to {output_file}")
    
    # Cleanup checkpoint if finished
    # os.remove(checkpoint_file)

if __name__ == "__main__":
    main()

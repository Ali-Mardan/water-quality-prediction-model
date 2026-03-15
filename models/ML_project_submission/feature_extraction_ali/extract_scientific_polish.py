import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load
from tqdm import tqdm
import os
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

def extract_scientific_polish_features_checkpointed(data_path):
    checkpoint_path = "scientific_polish_checkpoint.csv"
    
    print(f"Reading locations and dates from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Compute Hydro-Soil Interaction immediately (no external API needed)
    df['hydro_soil_interaction'] = df['NDTI'] * df['soil_clay']
    
    # 2. Group for NDCI extraction to save API calls
    df['Sample Date'] = pd.to_datetime(df['Sample Date'])
    df['YearMonth'] = df['Sample Date'].dt.to_period('M')
    
    # Group unique location + month/year
    unique_extractions = df[['Latitude', 'Longitude', 'YearMonth']].drop_duplicates().reset_index(drop=True)
    total_unique = len(unique_extractions)
    print(f"Total samples: {len(df)} | Unique (Loc, Month) groups: {total_unique}")

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        processed_df = pd.read_csv(checkpoint_path)
        print(f"Loaded checkpoint with {len(processed_df)} entries.")
        # Create a set of processed keys
        processed_keys = set(zip(processed_df['Latitude'].round(6), 
                                processed_df['Longitude'].round(6), 
                                processed_df['YearMonth'].astype(str)))
    else:
        processed_df = pd.DataFrame(columns=['Latitude', 'Longitude', 'YearMonth', 'ndci'])
        processed_keys = set()

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    buffer_deg = 0.0001 
    results_list = processed_df.to_dict('records')

    print("Extracting Sentinel-2 Red Edge (B5) and Red (B4) with Checkpointing...")
    
    count = 0
    for idx, row in tqdm(unique_extractions.iterrows(), total=total_unique):
        lat, lon = row['Latitude'], row['Longitude']
        year_month = row['YearMonth']
        ym_str = str(year_month)
        
        # Skip if already in checkpoint
        if (round(lat, 6), round(lon, 6), ym_str) in processed_keys:
            continue
            
        # Search window
        start_date = year_month.to_timestamp().strftime('%Y-%m-%d')
        end_date = (year_month.to_timestamp() + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        
        bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
        
        ndci = np.nan
        try:
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{start_date}/{end_date}",
                query={"eo:cloud_cover": {"lt": 30}}
            )
            
            items = list(search.items())
            
            if items:
                ds = stac_load(
                    [items[0]],
                    bbox=bbox,
                    bands=["B04", "B05"],
                    resolution=20
                )
                b4 = float(ds.B04.mean())
                b5 = float(ds.B05.mean())
                if (b5 + b4) != 0:
                    ndci = (b5 - b4) / (b5 + b4)
        except Exception:
            pass
        
        results_list.append({
            'Latitude': lat,
            'Longitude': lon,
            'YearMonth': ym_str,
            'ndci': ndci
        })
        
        # Save checkpoint every 10 iterations
        count += 1
        if count % 10 == 0:
            pd.DataFrame(results_list).to_csv(checkpoint_path, index=False)
            
    # Final save of checkpoint
    checkpoint_final = pd.DataFrame(results_list)
    checkpoint_final.to_csv(checkpoint_path, index=False)

    # 3. Apply results back to the main dataframe
    print("Mapping results back to main dataset...")
    # Create map from checkpoint
    ndci_map = {(row['Latitude'], row['Longitude'], str(row['YearMonth'])): row['ndci'] 
                for _, row in checkpoint_final.iterrows()}
    
    df['ndci'] = df.apply(lambda r: ndci_map.get((r['Latitude'], r['Longitude'], str(r['YearMonth']))), axis=1)
    
    # Clean up
    df = df.drop(columns=['YearMonth'])
    
    out_path = "master_dataset_final_polished.csv"
    df.to_csv(out_path, index=False)
    print(f"Scientific polish complete. Saved to {out_path}")
    return df

if __name__ == "__main__":
    extract_scientific_polish_features_checkpointed("master_dataset_final.csv")

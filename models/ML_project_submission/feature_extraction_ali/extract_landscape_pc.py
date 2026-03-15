import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")

def extract_landscape_planetary_computer(data_path, buffer_m=2000):
    print(f"Reading unique locations from {data_path}...")
    df = pd.read_csv(data_path)
    unique_coords = df[['Latitude', 'Longitude']].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(unique_coords)} unique locations.")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    results = []
    
    # Approx degrees for the buffer (111km per deg)
    buffer_deg = buffer_m / 111000.0

    print(f"Extracting ESA WorldCover from Planetary Computer...")
    
    for idx, row in tqdm(unique_coords.iterrows(), total=len(unique_coords)):
        lat, lon = row['Latitude'], row['Longitude']
        
        # Bounding box for the point with buffer
        bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
        
        search = catalog.search(
            collections=["esa-worldcover"],
            bbox=bbox
        )
        
        items = list(search.get_items())
        if not items:
            results.append({
                'Latitude': lat, 'Longitude': lon,
                'perc_agri': np.nan, 'perc_urban': np.nan, 
                'perc_grass': np.nan, 'perc_forest': np.nan
            })
            continue
            
        # Load the map data using rioxarray for more control
        try:
            import rioxarray
            import xarray as xr
            
            # Get the COG URL from the first item
            asset_url = items[0].assets["map"].href
            
            # Open the remote raster
            with rioxarray.open_rasterio(asset_url) as rds:
                # Clip to bbox
                # bbox is [minx, miny, maxx, maxy] (lon, lat)
                clipped = rds.rio.clip_box(
                    minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3],
                    crs="EPSG:4326"
                )
                data = clipped.values.flatten()
        except Exception as e:
            print(f"Error loading data for {lat}, {lon}: {e}")
            results.append({
                'Latitude': lat, 'Longitude': lon,
                'perc_agri': np.nan, 'perc_urban': np.nan, 
                'perc_grass': np.nan, 'perc_forest': np.nan
            })
            continue
        
        # Filter out NoData if applicable (usually 0 in WorldCover)
        data = data[data > 0]
        
        if len(data) == 0:
            results.append({
                'Latitude': lat, 'Longitude': lon,
                'perc_agri': 0.0, 'perc_urban': 0.0, 
                'perc_grass': 0.0, 'perc_forest': 0.0
            })
            continue

        counts = pd.Series(data).value_counts()
        total = len(data)
        
        results.append({
            'Latitude': lat,
            'Longitude': lon,
            'perc_agri': (counts.get(40, 0) / total) * 100,
            'perc_urban': (counts.get(50, 0) / total) * 100,
            'perc_grass': (counts.get(30, 0) / total) * 100,
            'perc_forest': (counts.get(10, 0) / total) * 100
        })

    res_df = pd.DataFrame(results)
    out_path = "advanced_landscape_features.csv"
    res_df.to_csv(out_path, index=False)
    print(f"Extraction complete. Saved to {out_path}")
    return res_df

if __name__ == "__main__":
    extract_landscape_planetary_computer("merged_training_data.csv")

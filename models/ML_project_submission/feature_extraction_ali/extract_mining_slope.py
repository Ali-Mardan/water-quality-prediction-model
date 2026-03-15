import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def extract_mining_and_slope(data_path):
    print(f"Reading unique locations from {data_path}...")
    df = pd.read_csv(data_path)
    unique_coords = df[['Latitude', 'Longitude']].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(unique_coords)} unique locations.")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    results = []
    
    # 2km buffer approx 0.018 degrees
    buffer_deg = 0.018 

    print(f"Extracting Barren (Mining proxy) and Topographic Slope...")
    
    for idx, row in tqdm(unique_coords.iterrows(), total=len(unique_coords)):
        lat, lon = row['Latitude'], row['Longitude']
        bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
        
        perc_barren = 0.0
        slope_val = np.nan
        
        try:
            import rioxarray
            import xarray as xr
            
            # --- 1. ESA WorldCover Barren ---
            search_wc = catalog.search(collections=["esa-worldcover"], bbox=bbox)
            items_wc = list(search_wc.items())
            
            if items_wc:
                asset_url = items_wc[0].assets["map"].href
                with rioxarray.open_rasterio(asset_url) as rds:
                    clipped_wc = rds.rio.clip_box(
                        minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], crs="EPSG:4326"
                    )
                    data_wc = clipped_wc.values.flatten()
                    data_wc = data_wc[data_wc > 0] # Remove NoData
                    
                    if len(data_wc) > 0:
                        counts = pd.Series(data_wc).value_counts()
                        perc_barren = (counts.get(60, 0) / len(data_wc)) * 100
                        
            # --- 2. Copernicus DEM (Slope) ---
            search_dem = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
            items_dem = list(search_dem.items())
            
            if items_dem:
                asset_url_dem = items_dem[0].assets["data"].href
                with rioxarray.open_rasterio(asset_url_dem) as rds_dem:
                    clipped_dem = rds_dem.rio.clip_box(
                        minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], crs="EPSG:4326"
                    )
                    elevation = clipped_dem.values[0]
                    y, x = np.gradient(elevation)
                    slope_approx = np.sqrt(x**2 + y**2) / 30.0 
                    mean_slope = np.nanmean(np.degrees(np.arctan(slope_approx)))
                    slope_val = float(mean_slope)
                    
        except Exception as e:
            # print(f"Error extracting for {lat}, {lon}: {e}")
            pass
            
        results.append({
            'Latitude': lat,
            'Longitude': lon,
            'perc_barren': perc_barren,
            'topographic_slope': slope_val
        })

    res_df = pd.DataFrame(results)
    
    # Merge with original dataframe
    print("Mapping results back to main dataset...")
    df_final = pd.merge(df, res_df, on=['Latitude', 'Longitude'], how='left')
    
    out_path = "master_dataset_ultimate.csv"
    df_final.to_csv(out_path, index=False)
    print(f"Extraction complete. Saved to {out_path}")
    return df_final

if __name__ == "__main__":
    extract_mining_and_slope("master_dataset_final_sota.csv")

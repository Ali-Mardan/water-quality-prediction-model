import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load
from datetime import datetime
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

def compute_Landsat_values(row, catalog):
    lat = row['Latitude']
    lon = row['Longitude']
    sample_date = pd.to_datetime(row['Sample Date'], dayfirst=True, errors='coerce')
    
    if pd.isnull(sample_date):
        return pd.Series({"nir": np.nan, "red": np.nan, "blue": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan})

    bbox_size = 0.00089831  
    bbox = [lon - bbox_size / 2, lat - bbox_size / 2, lon + bbox_size / 2, lat + bbox_size / 2]

    try:
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=bbox,
            datetime="2011-01-01/2015-12-31",
            query={"eo:cloud_cover": {"lt": 10}},
        )
        items = search.item_collection()
        if not items:
            return pd.Series({"nir": np.nan, "red": np.nan, "blue": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan})

        sample_date_utc = sample_date.tz_localize("UTC") if sample_date.tzinfo is None else sample_date.tz_convert("UTC")
        items = sorted(items, key=lambda x: abs(pd.to_datetime(x.properties["datetime"]).tz_convert("UTC") - sample_date_utc))
        selected_item = pc.sign(items[0])

        bands_of_interest = ["red", "blue", "green", "nir08", "swir16", "swir22"]
        data = stac_load([selected_item], bands=bands_of_interest, bbox=bbox).isel(time=0)

        results = {}
        band_map = {"red": "red", "blue": "blue", "green": "green", "nir": "nir08", "swir16": "swir16", "swir22": "swir22"}
        for key, band_name in band_map.items():
            val = float(data[band_name].astype("float").median(skipna=True).values)
            results[key] = val if val != 0 else np.nan
        return pd.Series(results)
    except Exception:
        return pd.Series({"nir": np.nan, "red": np.nan, "blue": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan})

def main():
    input_file = 'submission_template.csv'
    output_file = 'expanded_landsat_features_test.csv'
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples from {input_file}")

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    num_threads = 8
    all_results = []
    
    print(f"Starting extraction with {num_threads} threads...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(compute_Landsat_values, row, catalog): row for _, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Landsat Test Extraction"):
            row_orig = futures[future]
            res = future.result()
            res['Latitude'] = row_orig['Latitude']
            res['Longitude'] = row_orig['Longitude']
            res['Sample Date'] = row_orig['Sample Date']
            all_results.append(res.to_dict())

    final_df = pd.DataFrame(all_results)
    cols = ['Latitude', 'Longitude', 'Sample Date', 'red', 'blue', 'green', 'nir', 'swir16', 'swir22']
    final_df = final_df[cols]
    final_df.to_csv(output_file, index=False)
    print(f"Saved test Landsat features to {output_file}")

if __name__ == "__main__":
    main()

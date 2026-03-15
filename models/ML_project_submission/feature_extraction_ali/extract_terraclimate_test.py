import xarray as xr
import pandas as pd
import os
import numpy as np
import planetary_computer as pc
import pystac_client
import warnings

warnings.filterwarnings("ignore")

def load_terraclimate_dataset():
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]

    if "xarray:storage_options" in asset.extra_fields:
        ds = xr.open_zarr(
            asset.href,
            storage_options=asset.extra_fields["xarray:storage_options"],
            consolidated=True,
        )
    else:
        ds = xr.open_dataset(
            asset.href,
            **asset.extra_fields["xarray:open_kwargs"],
        )
    return ds

def filter_terraclimate(ds, vars_of_interest):
    # Select time range
    ds_subset = ds[vars_of_interest].sel(time=slice("2010-01-01", "2015-12-31"))
    # Slice to South Africa region
    ds_subset = ds_subset.sel(lat=slice(-21, -36), lon=slice(15, 34))
    print(f"Dataset subset selected for variables: {vars_of_interest}")
    return ds_subset

def main():
    input_file = 'submission_template.csv'
    output_file = 'expanded_terraclimate_features_test.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples from {input_file}")

    print(f"Loading TerraClimate dataset...")
    ds = load_terraclimate_dataset()
    
    vars_of_interest = ['ppt', 'tmax', 'tmin', 'pet', 'soil', 'def']
    ds_subset = filter_terraclimate(ds, vars_of_interest)

    print("Loading subset into memory...")
    ds_subset = ds_subset.compute()
    
    print("Calculating rolling features...")
    ds_subset['ppt_3mo'] = ds_subset['ppt'].rolling(time=3, center=False).sum()
    ds_subset['ppt_6mo'] = ds_subset['ppt'].rolling(time=6, center=False).sum()
    ds_subset['tmax_3mo'] = ds_subset['tmax'].rolling(time=3, center=False).mean()
    
    vars_of_interest += ['ppt_3mo', 'ppt_6mo', 'tmax_3mo']

    print("Extracting test data using vectorized selection...")
    df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True)
    
    lat_da = xr.DataArray(df['Latitude'].values, dims="sample")
    lon_da = xr.DataArray(df['Longitude'].values, dims="sample")
    time_da = xr.DataArray(df['Sample Date'].values, dims="sample")

    point_data = ds_subset.sel(
        lat=lat_da, 
        lon=lon_da, 
        time=time_da, 
        method='nearest'
    )
    
    expanded_df = point_data.to_dataframe().reset_index()
    
    # Re-attach original metadata
    original_df = pd.read_csv(input_file)
    expanded_df['Latitude'] = original_df['Latitude']
    expanded_df['Longitude'] = original_df['Longitude']
    expanded_df['Sample Date'] = original_df['Sample Date']
    
    final_cols = ['Latitude', 'Longitude', 'Sample Date'] + vars_of_interest
    expanded_df = expanded_df[final_cols]
    
    expanded_df.to_csv(output_file, index=False)
    print(f"Saved test TerraClimate features to {output_file}")

if __name__ == "__main__":
    main()

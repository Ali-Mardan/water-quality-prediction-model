import pandas as pd
import ee
import time
from tqdm import tqdm

# Initialize Earth Engine
try:
    print("Attempting to initialize Earth Engine...")
    ee.Initialize() 
    print("Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error during initialization: {e}")
    print("Attempting to initialize with a dummy project ID...")
    try:
        # Sometimes it needs a project ID, let's try 'ee-shreyas-ds' again but handle if it fails
        ee.Initialize(project='ee-shreyas-ds')
        print("Earth Engine initialized with project 'ee-shreyas-ds'.")
    except Exception as e2:
        print(f"Error initializing with project 'ee-shreyas-ds': {e2}")
        print("Please ensure you have run 'earthengine authenticate' in your terminal and select a project.")
        raise e2

def extract_advanced_landscape_features(data_path, buffer_m=2000):
    print(f"Reading unique locations from {data_path}...")
    df = pd.read_csv(data_path)
    unique_coords = df[['Latitude', 'Longitude']].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(unique_coords)} unique locations.")

    # 1. ESA WorldCover 2021 (10m)
    # Values: 10 (Trees), 20 (Shrubland), 30 (Grassland), 40 (Cropland), 50 (Built-up), 60 (Barren), 70 (Snow/Ice), 80 (Water), 90 (Marsh), 95 (Mangroves), 100 (Moss)
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
    
    # 2. Global Lithological Map (GLiM) - Carbonate classification
    # Note: Using GLiM available in community datasets or similar. 
    # If standard GLiM is not in EE, we'll try to use a proxy of geology or keep it simple with WorldCover for now.
    # Actually, let's focus on WorldCover first as it is highly reliable in EE.
    
    landuse_results = []
    
    # Batch size for GEE requests
    batch_size = 50 
    
    print(f"Extracting Land Cover in {buffer_m}m buffers...")
    
    for i in tqdm(range(0, len(unique_coords), batch_size)):
        batch = unique_coords.iloc[i:i+batch_size]
        
        points = [ee.Feature(ee.Geometry.Point([row['Longitude'], row['Latitude']]).buffer(buffer_m), 
                  {'id': f"{row['Latitude']}_{row['Longitude']}"}) for idx, row in batch.iterrows()]
        
        fc = ee.FeatureCollection(points)
        
        # Calculate area of Crop (40) and Built-up (50)
        def get_stats(feature):
            reduced = worldcover.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=feature.geometry(),
                scale=10,
                maxPixels=1e9
            )
            return feature.set('histogram', reduced.get('label'))

        processed_fc = fc.map(get_stats).getInfo()
        
        for feat in processed_fc['features']:
            hist = feat['properties'].get('histogram', {})
            total = sum(hist.values()) if hist else 1
            
            # Map ESA classes
            perc_agri = (hist.get('40', 0) / total) * 100
            perc_urban = (hist.get('50', 0) / total) * 100
            perc_grass = (hist.get('30', 0) / total) * 100
            perc_forest = (hist.get('10', 0) / total) * 100
            
            landuse_results.append({
                'loc_id': feat['properties']['id'],
                'perc_agri': perc_agri,
                'perc_urban': perc_urban,
                'perc_grass': perc_grass,
                'perc_forest': perc_forest
            })
            
    # Convert results to dataframe and split loc_id back
    res_df = pd.DataFrame(landuse_results)
    res_df[['Latitude', 'Longitude']] = res_df['loc_id'].str.split('_', expand=True).astype(float)
    res_df = res_df.drop(columns=['loc_id'])
    
    # Save the landscape features
    out_path = "advanced_landscape_features.csv"
    res_df.to_csv(out_path, index=False)
    print(f"Extraction complete. Results saved to {out_path}")
    
    return res_df

if __name__ == "__main__":
    extract_advanced_landscape_features("merged_training_data.csv")

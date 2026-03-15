import pandas as pd
from data_sequence_prep import load_and_preprocess_data

def prepare_for_timegpt(target_file, landsat_file, terraclimate_file, target_col):
    """
    Prepares the merged multivariate dataset into the specific Zero-Shot 
    format expected by TimeGPT (Nixtla) or similar libraries.
    """
    print("Loading and merging data...")
    merged_df = load_and_preprocess_data(target_file, landsat_file, terraclimate_file)

    print("Formatting for TimeGPT...")
    # TimeGPT needs specific column names: unique_id, ds, y
    timegpt_df = merged_df.rename(columns={
        'Loc_ID': 'unique_id',
        'Sample Date': 'ds',
        target_col: 'y'
    })

    # Sort and reset index
    timegpt_df = timegpt_df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    
    # Optional: Fill missing values as TimeGPT requires dense time-series
    # Here we are just doing a basic forward/backward fill per group
    timegpt_df = timegpt_df.groupby('unique_id').apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    
    # Select columns to save (unique_id, ds, y, and the exog features)
    cols = ['unique_id', 'ds', 'y'] + [col for col in timegpt_df.columns if col not in ['unique_id', 'ds', 'y', 'Latitude', 'Longitude']]
    timegpt_df = timegpt_df[cols]
    
    output_file = 'timegpt_ready_dataset.csv'
    timegpt_df.to_csv(output_file, index=False)
    print(f"Successfully saved formatting data to {output_file}")
    print(f"Shape: {timegpt_df.shape}")
    print("Head:")
    print(timegpt_df.head(3))
    
    return timegpt_df

if __name__ == "__main__":
    target_file = 'water_quality_training_dataset.csv'
    landsat_file = 'landsat_features_training.csv'
    terraclimate_file = 'terraclimate_features_training.csv'
    
    # Example target
    target_df = pd.read_csv(target_file, nrows=1)
    target_column = target_df.columns[3]
    
    prepare_for_timegpt(target_file, landsat_file, terraclimate_file, target_col=target_column)

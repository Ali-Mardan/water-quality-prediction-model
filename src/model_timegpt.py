import pandas as pd
import numpy as np
from nixtla import NixtlaClient
from sklearn.metrics import r2_score, mean_squared_error
import os

def run_timegpt_baseline():
    dataset_path = 'timegpt_ready_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run timegpt_prep.py first.")
        return

    print("Loading TimeGPT format dataset...")
    df = pd.read_csv(dataset_path)
    df['ds'] = pd.to_datetime(df['ds'])
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Unique Sites: {df['unique_id'].nunique()}")

    print("Initializing NixtlaClient...")
    try:
        nixtla_client = NixtlaClient(api_key="nixak-NQrAApCEVEF3Nk5y8D3oMaExh9MbdgG866fshU5UiaaJyzMfLMMQRHQJbtNybriMBYOxjaMKSt2AC4Sf")
        if hasattr(nixtla_client, 'validate_api_key'):
            valid = nixtla_client.validate_api_key()
            if not valid:
                print("Nixtla validation failed. Ensure the API key is correct.")
                return
    except Exception as e:
        print(f"Failed to initialize NixtlaClient: {e}")
        return

    print("\nRunning TimeGPT cross-validation (Zero-Shot Baseline)...")
    try:
        # Cross validation automatically creates train/test splits sequentially
        cv_df = nixtla_client.cross_validation(
            df=df,
            h=1, # Forecast 1 step (1 month) ahead
            n_windows=1, 
            step_size=1,
            time_col='ds',
            target_col='y',
            id_col='unique_id'
        )
        
        print("\nCross-validation completed!")
        print(cv_df.head())
        
        # Calculate evaluation metrics
        cv_df = cv_df.dropna(subset=['y', 'TimeGPT'])
        if len(cv_df) > 0:
            r2 = r2_score(cv_df['y'], cv_df['TimeGPT'])
            rmse = np.sqrt(mean_squared_error(cv_df['y'], cv_df['TimeGPT']))
            print("\n=== TimeGPT Zero-Shot Evaluation ===")
            print(f"Test R²:   {r2:.4f}")
            print(f"Test RMSE: {rmse:.4f}")
            print("====================================")
        else:
            print("No valid predictions were returned to compute metrics.")

    except Exception as e:
        print(f"\nTimeGPT execution failed: {e}")
        print("Note: Nixtla free tier might have rate limits or restrict the number of unique spatial points.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_timegpt_baseline()

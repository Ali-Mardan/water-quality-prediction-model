import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import os

def main():
    train_file = 'final_training_data_enriched.csv'
    test_file = 'final_test_data_enriched.csv'
    template_file = 'submission_template.csv'
    output_file = 'final_submission.csv'
    
    # Load data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Features (Must match tune_models.py Exactly)
    landsat_feat = ['red', 'blue', 'green', 'nir', 'swir16', 'swir22', 'NDVI', 'SI2', 'NDTI', 'MNDWI', 'NDMI']
    climate_feat = ['pet', 'ppt', 'tmax', 'tmin', 'soil', 'def', 'ppt_3mo', 'ppt_6mo', 'tmax_3mo', 'T_range']
    temporal_feat = ['Month', 'DayOfYear']
    
    features = climate_feat + temporal_feat + landsat_feat
    
    # Preprocess
    X_train_full = train_df[features].fillna(train_df[features].median())
    X_test = test_df[features].fillna(train_df[features].median()) # Impute test with train median
    
    # Target variables
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Tuned Hyperparameters (from tune_models.py output)
    tuned_params = {
        'subsample': 0.7, 
        'n_estimators': 1000, 
        'max_depth': 10, 
        'learning_rate': 0.01, 
        'colsample_bytree': 0.7,
        'random_state': 42
    }
    
    submission_df = pd.read_csv(template_file)
    
    for target in targets:
        print(f"Training final model for {target}...")
        y_train = train_df[target]
        
        # Train on valid indices only
        valid_idx = y_train.notna()
        X_train_v = X_train_full[valid_idx]
        y_train_v = y_train[valid_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_v)
        X_test_s = scaler.transform(X_test)
        
        model = XGBRegressor(**tuned_params)
        model.fit(X_train_s, y_train_v)
        
        preds = model.predict(X_test_s)
        # Handle potential negative predictions (physically impossible for these parameters)
        preds = np.maximum(preds, 0)
        
        submission_df[target] = preds
        print(f"Predictions completed for {target}.")
    
    # Final check on column order
    cols = ['Latitude', 'Longitude', 'Sample Date', 'Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    submission_df = submission_df[cols]
    
    # Save
    submission_df.to_csv(output_file, index=False)
    print(f"Final submission saved to {output_file}")

if __name__ == "__main__":
    main()

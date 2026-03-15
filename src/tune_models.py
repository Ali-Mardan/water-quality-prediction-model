import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os

# Load enriched data
DATA_PATH = 'final_training_data_enriched.csv'

def load_and_preprocess():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run feature_engineering.py first.")
    
    df = pd.read_csv(DATA_PATH)
    
    # Feature set
    landsat_feat = ['red', 'blue', 'green', 'nir', 'swir16', 'swir22', 'NDVI', 'SI2', 'NDTI', 'MNDWI', 'NDMI']
    climate_feat = ['pet', 'ppt', 'tmax', 'tmin', 'soil', 'def', 'ppt_3mo', 'ppt_6mo', 'tmax_3mo', 'T_range']
    temporal_feat = ['Month', 'DayOfYear']
    
    features = climate_feat + temporal_feat + landsat_feat
    
    X = df[features]
    y_dict = {
        'Total Alkalinity': df['Total Alkalinity'],
        'Electrical Conductance': df['Electrical Conductance'],
        'Dissolved Reactive Phosphorus': df['Dissolved Reactive Phosphorus']
    }
    
    # Simple imputation
    X = X.fillna(X.median())
    
    return X, y_dict

def tune_parameter(X, y, target_name):
    # Filter valid target indices
    valid_mask = y.notna()
    X_v = X[valid_mask]
    y_v = y[valid_mask]
    
    X_train, X_test, y_train, y_test = train_test_split(X_v, y_v, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Define Parameter Grids
    rf_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    xgb_grid = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 10],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    
    print(f"\n--- Tuning for {target_name} ---")
    
    # Tune RF
    rf = RandomForestRegressor(random_state=42)
    rf_search = RandomizedSearchCV(rf, rf_grid, n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=42)
    rf_search.fit(X_train_s, y_train)
    rf_best = rf_search.best_estimator_
    rf_r2 = r2_score(y_test, rf_best.predict(X_test_s))
    print(f"RF  Best Params: {rf_search.best_params_}")
    print(f"RF  R2 Score: {rf_r2:.4f}")
    
    # Tune XGB
    xgb = XGBRegressor(random_state=42)
    xgb_search = RandomizedSearchCV(xgb, xgb_grid, n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=42)
    xgb_search.fit(X_train_s, y_train)
    xgb_best = xgb_search.best_estimator_
    xgb_r2 = r2_score(y_test, xgb_best.predict(X_test_s))
    print(f"XGB Best Params: {xgb_search.best_params_}")
    print(f"XGB R2 Score: {xgb_r2:.4f}")
    
    return {
        'target': target_name,
        'rf_r2': rf_r2,
        'xgb_r2': xgb_r2,
        'best_model': 'RF' if rf_r2 > xgb_r2 else 'XGB'
    }

def main():
    X, y_dict = load_and_preprocess()
    results = []
    
    for name, y in y_dict.items():
        res = tune_parameter(X, y, name)
        results.append(res)
        
    print("\n--- Final Summary ---")
    summary_df = pd.DataFrame(results)
    print(summary_df)

if __name__ == "__main__":
    main()

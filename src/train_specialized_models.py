import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def train_specialized_models(master_path):
    print(f"{'='*50}\nTraining Specialized Scientific Engines\n{'='*50}")
    
    df = pd.read_csv(master_path)
    df['spatial_group'] = df['Latitude'].astype(str) + "_" + df['Longitude'].astype(str)
    
    # Define Specialized Feature Sets based on the Hydro-Informatics Report
    model_configs = {
        'Total Alkalinity': {
            'features': [
                'carbonate_index', 'swir22', 'swir16', 
                'perc_forest', 'perc_grass', 'soil_carbon', 'Month', 'soil_ph', 
                'topographic_slope', 'soil_texture', 'ppt_3mo', 'ppt_6mo', 
                'temperature', 'urban_precip_interaction', 'perc_urban'
            ],
            'params': {'learning_rate': 0.03, 'max_depth': 7, 'n_estimators': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8}
        },
        'Electrical Conductance': {
            'features': [
                'perc_urban', 'perc_agri', 'perc_barren', 
                'precipitation', 'ppt_3mo', 'ppt_6mo', 
                'SI1', 'SI2', 'NDSI', 'NDVI', 
                'evap_ratio', 'evapotranspiration'
            ],
            'params': {'learning_rate': 0.07, 'max_depth': 7, 'n_estimators': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8}
        },
        'Dissolved Reactive Phosphorus': {
            'features': [
                'perc_agri', 'soil_clay', 'soil_ph', 
                'hydro_soil_interaction', 'ndci', 'NDTI', 
                'ndci_turbidity_interaction', 'post_2008', 
                'topographic_slope', 'temperature', 'precipitation'
            ],
            'params': {'learning_rate': 0.03, 'max_depth': 7, 'n_estimators': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8}
        }
    }

    results_summary = []

    for target, config in model_configs.items():
        print(f"\n--- Training {target} (Engine Type: {target.split()[-1]}) ---")
        df_target = df.dropna(subset=[target]).copy()
        
        available_features = [f for f in config['features'] if f in df_target.columns]
        missing_features = [f for f in config['features'] if f not in df_target.columns]
        
        if missing_features:
            print(f"Warning: Missing features for {target}: {missing_features}")
            
        X = df_target[available_features].copy()
        y = df_target[target].copy()
        groups = df_target['spatial_group'].copy()
        
        # Spatial 80/20 Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        try:
            train_idx, test_idx = next(gss.split(X, y, groups))
        except StopIteration:
            continue
            
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # Preprocessing: Use training medians for imputation
        medians = X_train.median()
        X_train_imputed = X_train.fillna(medians)
        X_test_imputed = X_test.fillna(medians)
        
        # Train specialized model
        model = XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1, **config['params'])
        model.fit(X_train_imputed, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_imputed)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Top 5 features for this specific engine
        importances = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
        print(f"Top 5 {target} Drivers:")
        print(importances.head(5))
        
        print(f"Results -> R2: {r2:.4f}, RMSE: {rmse:.4f}")
        
        results_summary.append({
            'Target': target,
            'Test_R2': r2,
            'Test_RMSE': rmse,
            'Feature_Count': len(available_features)
        })

    print(f"\n{'='*50}\nSPECIALIZED MODEL PERFORMANCE\n{'='*50}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    train_specialized_models("master_dataset_ultimate.csv")

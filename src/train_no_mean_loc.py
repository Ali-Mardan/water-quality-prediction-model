import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def train_without_loc_mean(master_path):
    print(f"{'='*60}\nEvaluating Model WITHOUT 'loc_mean' Variables\n{'='*60}")
    
    df = pd.read_csv(master_path)
    
    print(f"Original Feature Space Dimension: {df.shape}")
    
    # Identify target variables and exclusion list
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    df['spatial_group'] = df['Latitude'].astype(str) + "_" + df['Longitude'].astype(str)
    
    # Exclude non-feature columns AND the loc_mean variables as requested
    loc_mean_vars = ['loc_mean_cond', 'loc_mean_alk', 'loc_mean_phos']
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin', 'spatial_group'] + loc_mean_vars
    
    available_features = [col for col in df.columns if col not in features_to_drop]
    print(f"Features being used ({len(available_features)} total): {available_features}")
    print(f"Excluding variables: {loc_mean_vars}")

    results_summary = []

    for target in targets:
        print(f"\n--- Evaluating Target: {target} ---")
        df_target = df.dropna(subset=[target]).copy()
        
        X = df_target[available_features].copy()
        y = df_target[target].copy()
        groups = df_target['spatial_group'].copy()
        
        # Spatial 80/20 Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        try:
            train_idx, test_idx = next(gss.split(X, y, groups))
        except StopIteration:
            print(f"Warning: Could not create split for {target}")
            continue
            
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # Preprocessing isolation
        medians = X_train.median()
        X_train_imputed = X_train.fillna(medians)
        X_test_imputed = X_test.fillna(medians)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        print(f"Training XGBoost on {X_train_scaled.shape[1]} features...")
        model = XGBRegressor(
            random_state=42, 
            objective='reg:squarederror',
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        
        # Feature Importance (Top 10)
        importances = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
        print("\nTop 10 Important Features:")
        print(importances.head(10))
        
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\nResults -> R2: {r2:.4f}, Adj_R2: {adj_r2:.4f}, RMSE: {rmse:.4f}")
        
        results_summary.append({
            'Target': target,
            'Test_R2': r2,
            'Test_Adj_R2': adj_r2,
            'Test_RMSE': rmse
        })

    print(f"\n{'='*60}\nRESULTS SUMMARY: WITHOUT LOC_MEAN\n{'='*60}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    train_without_loc_mean("master_dataset_ultimate.csv")

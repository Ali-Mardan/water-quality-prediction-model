import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def finalize_and_evaluate(master_path, landscape_path):
    print(f"{'='*50}\nFinalizing Master Dataset\n{'='*50}")
    
    df_master = pd.read_csv(master_path)
    df_landscape = pd.read_csv(landscape_path)

    # Standardize Coordinates for join
    df_master['Latitude'] = df_master['Latitude'].round(6)
    df_master['Longitude'] = df_master['Longitude'].round(6)
    df_landscape['Latitude'] = df_landscape['Latitude'].round(6)
    df_landscape['Longitude'] = df_landscape['Longitude'].round(6)

    print("Merging Advanced Landscape Features...")
    df_final = pd.merge(df_master, df_landscape, on=['Latitude', 'Longitude'], how='left')
    
    print(f"Final Feature Space Dimension: {df_final.shape}")
    df_final.to_csv("master_dataset_final.csv", index=False)
    print("Saved final dataset to master_dataset_final.csv")

    # 80/20 Spatial Split Evaluation
    df_final['spatial_group'] = df_final['Latitude'].astype(str) + "_" + df_final['Longitude'].astype(str)
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Exclude non-feature columns
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin', 'spatial_group']
    available_features = [col for col in df_final.columns if col not in features_to_drop]

    results_summary = []

    for target in targets:
        print(f"\n--- Evaluating Target: {target} ---")
        df_target = df_final.dropna(subset=[target]).copy()
        
        X = df_target[available_features].copy()
        y = df_target[target].copy()
        groups = df_target['spatial_group'].copy()
        
        # Spatial 80/20 Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # Preprocessing isolation
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("Training XGBoost...")
        model = XGBRegressor(
            random_state=42, 
            objective='reg:squarederror',
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Results -> R2: {r2:.4f}, Adj_R2: {adj_r2:.4f}, RMSE: {rmse:.4f}")
        
        results_summary.append({
            'Target': target,
            'Test_R2': r2,
            'Test_Adj_R2': adj_r2,
            'Test_RMSE': rmse
        })

    print(f"\n{'='*50}\nFINAL STATE-OF-THE-ART RESULTS\n{'='*50}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    finalize_and_evaluate("merged_training_data.csv", "advanced_landscape_features.csv")

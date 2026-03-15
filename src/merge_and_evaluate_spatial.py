import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
import os
warnings.filterwarnings('ignore')

def merge_datasets():
    print(f"{'='*50}\nMerging Datasets\n{'='*50}")
    shreyas_path = "shreyasds/training_dataset_enhanced.csv"
    custom_path = "final_training_data_enriched.csv"
    
    print(f"Loading {shreyas_path}...")
    df_shreyas = pd.read_csv(shreyas_path)
    if 'Unnamed: 0' in df_shreyas.columns:
        df_shreyas = df_shreyas.drop(columns=['Unnamed: 0'])
        
    print(f"Loading {custom_path}...")
    df_custom = pd.read_csv(custom_path)
    if 'Unnamed: 0' in df_custom.columns:
        df_custom = df_custom.drop(columns=['Unnamed: 0'])

    # Standardize Dates
    print("Standardizing Date formats...")
    df_shreyas['Sample Date'] = pd.to_datetime(df_shreyas['Sample Date']).dt.strftime('%Y-%m-%d')
    df_custom['Sample Date'] = pd.to_datetime(df_custom['Sample Date']).dt.strftime('%Y-%m-%d')

    # Standardize Coordinates to 6 decimal places to avoid floating point mismatch
    print("Standardizing Coordinate precision...")
    df_shreyas['Latitude'] = df_shreyas['Latitude'].round(6)
    df_shreyas['Longitude'] = df_shreyas['Longitude'].round(6)
    df_custom['Latitude'] = df_custom['Latitude'].round(6)
    df_custom['Longitude'] = df_custom['Longitude'].round(6)

    # Perform Inner Join
    print(f"Performing Inner Join on ['Latitude', 'Longitude', 'Sample Date']...")
    
    # Drop target columns from custom to avoid duplicates `_x`, `_y` during merge
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    for t in targets:
        if t in df_custom.columns:
            df_custom = df_custom.drop(columns=[t])

    # We need to drop any other columns that exist in both to avoid _x/_y suffixes if they aren't the merge keys
    merge_keys = ['Latitude', 'Longitude', 'Sample Date']
    common_cols = set(df_shreyas.columns).intersection(set(df_custom.columns)) - set(merge_keys)
    df_custom = df_custom.drop(columns=list(common_cols))

    df_merged = pd.merge(df_shreyas, df_custom, on=merge_keys, how='inner')
    
    print(f"Merged Dataset Shape: {df_merged.shape}")
    
    out_path = "merged_training_data.csv"
    df_merged.to_csv(out_path, index=False)
    print(f"Saved merged dataset to {out_path}")
    return df_merged

def run_spatial_pipeline(df):
    print(f"\n{'='*50}")
    print(f"Executing Strict Spatial Pipeline (80/20) on Merged Dataset")
    print(f"{'='*50}")

    df['spatial_group'] = df['Latitude'].astype(str) + "_" + df['Longitude'].astype(str)

    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin', 'spatial_group']
    available_features = [col for col in df.columns if col not in features_to_drop]

    results_summary = []

    for target in targets:
        print(f"\n--- Evaluating Target: {target} ---")
        
        # 1. Select specific target (drop missing labels)
        df_target = df.dropna(subset=[target]).copy()
        
        X = df_target[available_features].copy()
        y = df_target[target].copy()
        groups = df_target['spatial_group'].copy()
        
        # Categorical Encoding
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # 2. Strict Spatial 80/20 Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        
        print(f"Features: {X.shape[1]}")
        print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        
        # 3. Preprocessing (ABSOLUTE LEAKAGE PREVENTION: Fit on Train ONLY)
        # Imputation
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians) # Transform test with Train medians
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) # Transform test with Train scaler
        
        n = X_test_scaled.shape[0]
        p = X_test_scaled.shape[1]

        # 4. Train Model
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
        
        # 5. Evaluate on true spatial holdout set
        y_pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Results -> R2: {r2:.4f}, Adj_R2: {adj_r2:.4f}, RMSE: {rmse:.4f}")
        
        results_summary.append({
            'Target': target,
            'Test_R2': r2,
            'Test_Adj_R2': adj_r2,
            'Test_RMSE': rmse
        })

    print(f"\n{'='*50}\nFINAL RESULTS (MERGED DATASET - 80/20 SPATIAL SPLIT)\n{'='*50}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    df_merged = merge_datasets()
    run_spatial_pipeline(df_merged)

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def finalize_and_evaluate_final(master_path):
    print(f"{'='*50}\nFinalizing State-of-the-Art Master Dataset\n{'='*50}")
    
    df = pd.read_csv(master_path)

    # 1. Feature Engineering: Carbonate Index (Lithology proxy)
    df['carbonate_index'] = df['swir22'] / (df['swir16'] + 1e-6)
    
    # 2. Add NDCI * Turbidity interaction (Biological/Physical soup)
    df['ndci_turbidity_interaction'] = df['ndci'] * df['NDTI']
    
    # 3. The "Phosphorus Shift" (Structural Break)
    # The report mentions a national-scale decrease in dissolved phosphate after 2008.
    df['post_2008'] = (df['year'] > 2008).astype(int)
    
    # 4. Urban Karst Flushing Interaction
    # Interaction between Urban % and Precipitation to capture the flushing of concrete weathering products.
    df['urban_precip_interaction'] = df['perc_urban'] * df['precipitation']
    
    # 5. Normalized Difference Salinity Index (NDSI)
    # Different papers use different bands, but a common one for soil salinity using Sentinel/Landsat is (Red - NIR) / (Red + NIR)
    # Sometimes this is called salinity index, we'll use (Red - NIR) / (Red + NIR) as a proxy for vegetation stress/salinity.
    df['NDSI'] = (df['red'] - df['nir']) / (df['red'] + df['nir'] + 1e-6)

    print(f"Final Feature Space Dimension: {df.shape}")
    df.to_csv("master_dataset_final_sota.csv", index=False)
    print("Saved final SOTA dataset to master_dataset_final_sota.csv")

    # 80/20 Spatial Split Evaluation
    df['spatial_group'] = df['Latitude'].astype(str) + "_" + df['Longitude'].astype(str)
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Exclude non-feature columns
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin', 'spatial_group']
    available_features = [col for col in df.columns if col not in features_to_drop]

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
        # Use simple mean/median for sparse NDCI if needed, or let XGBoost handle it
        # Actually for XGBoost we can leave NaNs as is or use a constant. 
        # But we previously used medians. To be consistent with user request:
        # "Run the imputer and scaler only on the training set and transform the test set."
        
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
            learning_rate=0.03, # Slightly slower LR for better generalization
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

    print(f"\n{'='*50}\nFINAL SCIENTIFIC-ALIGNED RESULTS\n{'='*50}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    finalize_and_evaluate_final("master_dataset_final_polished.csv")

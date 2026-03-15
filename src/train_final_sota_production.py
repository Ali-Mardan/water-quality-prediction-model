import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def finalize_and_evaluate_final_sota(master_path):
    print(f"{'='*50}\nFinal Grand Evaluation: Scaled & Tuned SOTA\n{'='*50}")
    
    df = pd.read_csv(master_path)

    # 80/20 Spatial Split Evaluation
    df['spatial_group'] = df['Latitude'].astype(str) + "_" + df['Longitude'].astype(str)
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Exclude non-feature columns
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin', 'spatial_group']
    available_features = [col for col in df.columns if col not in features_to_drop]

    results_summary = []

    # Tuned Params based on Grid Search results
    best_params = {
        'Total Alkalinity': {'learning_rate': 0.03, 'max_depth': 7, 'n_estimators': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8},
        'Electrical Conductance': {'learning_rate': 0.07, 'max_depth': 7, 'n_estimators': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8},
        'Dissolved Reactive Phosphorus': {'learning_rate': 0.03, 'max_depth': 7, 'n_estimators': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8}
    }

    for target in targets:
        print(f"\n--- Final Evaluation for: {target} ---")
        df_target = df.dropna(subset=[target]).copy()
        
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

        # Preprocessing (Median imputation helps sparse features like NDCI)
        medians = X_train.median()
        X_train_imputed = X_train.fillna(medians)
        X_test_imputed = X_test.fillna(medians)
        
        # XGBoost doesn't strictly need scaling, but we'll stick to the model space
        params = best_params[target]
        model = XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1, **params)
        model.fit(X_train_imputed, y_train)
        
        y_pred = model.predict(X_test_imputed)
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

    print(f"\n{'='*50}\nFINAL SOTA PERFORMANCE (Spatial Holdout)\n{'='*50}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df[['Target', 'Test_R2', 'Test_Adj_R2', 'Test_RMSE']].to_string(index=False))

if __name__ == "__main__":
    finalize_and_evaluate_final_sota("master_dataset_ultimate.csv")

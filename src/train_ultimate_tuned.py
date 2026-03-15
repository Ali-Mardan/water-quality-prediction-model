import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def finalize_and_evaluate_tuned(master_path):
    print(f"{'='*50}\nHyperparameter Tuning: SOTA Native XGBoost\n{'='*50}")
    
    df = pd.read_csv(master_path)

    # 80/20 Spatial Split Evaluation
    df['spatial_group'] = df['Latitude'].astype(str) + "_" + df['Longitude'].astype(str)
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Exclude non-feature columns
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin', 'spatial_group']
    available_features = [col for col in df.columns if col not in features_to_drop]

    results_summary = []

    # Targeted Grid Search Space
    param_grid = {
        'max_depth': [5, 7],
        'learning_rate': [0.03, 0.07],
        'n_estimators': [1000],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    for target in targets:
        print(f"\n--- Optimizing Target: {target} ---")
        df_target = df.dropna(subset=[target]).copy()
        
        X = df_target[available_features].copy()
        y = df_target[target].copy()
        groups = df_target['spatial_group'].copy()
        
        # 1. Outer 80/20 Spatial Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        try:
            train_idx, test_idx = next(gss.split(X, y, groups))
        except StopIteration:
            continue
            
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        groups_train = groups.iloc[train_idx].copy()

        # 2. Inner GridSearchCV with GroupKFold (maintaining spatial independence)
        print("Launching Grid Search with 3-Fold Spatial Cross-Validation...")
        gkf = GroupKFold(n_splits=3)
        
        # Remove scaling/imputation - XGBoost handles NaNs natively
        xgb = XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=gkf,
            scoring='r2',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train, groups=groups_train)
        
        best_model = grid_search.best_estimator_
        print(f"Best Params: {grid_search.best_params_}")
        
        # 3. Final Evaluation
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # Feature Importance (Top 10)
        importances = pd.Series(best_model.feature_importances_, index=available_features).sort_values(ascending=False)
        print("\nTop 10 Important Features:")
        print(importances.head(10))
        
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\nResults -> R2: {r2:.4f}, Adj_R2: {adj_r2:.4f}, RMSE: {rmse:.4f}")
        
        results_summary.append({
            'Target': target,
            'Best_Params': str(grid_search.best_params_),
            'Test_R2': r2,
            'Test_Adj_R2': adj_r2,
            'Test_RMSE': rmse
        })

    print(f"\n{'='*50}\nFINAL TUNED RESULTS\n{'='*50}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df[['Target', 'Test_R2', 'Test_Adj_R2', 'Test_RMSE']].to_string(index=False))

if __name__ == "__main__":
    finalize_and_evaluate_tuned("master_dataset_ultimate.csv")

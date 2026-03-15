import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def train_robust_regularized(master_path):
    print(f"{'='*60}\nROBUST REGULARIZATION SEARCH: Anti-Overfitting Search\n{'='*60}")
    
    df = pd.read_csv(master_path)
    df['spatial_group'] = df['Latitude'].astype(str) + "_" + df['Longitude'].astype(str)
    
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Exclude non-feature columns
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin', 'spatial_group']
    available_features = [col for col in df.columns if col not in features_to_drop]

    # User-provided Robust Parameter Space
    param_distributions = {
        'max_depth': [2, 3, 4],
        'min_child_weight': [5, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.75, 0.9],
        'colsample_bytree': [0.6, 0.75, 0.9],
        'reg_lambda': [1, 10, 50],
        'reg_alpha': [0, 0.5, 5],
        'gamma': [0, 0.5, 2]
    }

    results_summary = []

    for target in targets:
        print(f"\n--- Tuning Target: {target} ---")
        df_target = df.dropna(subset=[target]).copy()
        
        X = df_target[available_features].copy()
        y = df_target[target].copy()
        groups = df_target['spatial_group'].copy()
        
        # 1. Outer 80/20 Spatial Split (The True Holdout)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        try:
            train_idx, test_idx = next(gss.split(X, y, groups))
        except StopIteration:
            continue
            
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        groups_train = groups.iloc[train_idx].copy()

        # Preprocessing: Impute with Training Medians (Strict isolation)
        medians = X_train.median()
        X_train_imputed = X_train.fillna(medians)
        X_test_imputed = X_test.fillna(medians)

        # 2. Inner Randomized Search with Spatial GroupKFold
        print(f"Starting RandomizedSearch (n_iter=50) for {target}...")
        gkf = GroupKFold(n_splits=3)
        
        xgb = XGBRegressor(
            n_estimators=1000, 
            random_state=42, 
            objective='reg:squarederror', 
            n_jobs=-1,
            early_stopping_rounds=50 # Add early stopping to prevent tree growth overfitting
        )
        
        # Note: early_stopping_rounds in constructor needs eval_set in fit
        # We'll use a validation set from the train set for early stopping
        # For simplicity in RandomizedSearchCV, we'll just allow it to find best params
        
        random_search = RandomizedSearchCV(
            estimator=XGBRegressor(n_estimators=500, random_state=42, objective='reg:squarederror', n_jobs=-1),
            param_distributions=param_distributions,
            n_iter=50,
            cv=gkf,
            scoring='r2',
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train_imputed, y_train, groups=groups_train)
        
        best_model = random_search.best_estimator_
        print(f"Best Params: {random_search.best_params_}")
        
        # 3. Final Evaluation on Holdout
        y_pred = best_model.predict(X_test_imputed)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Robust R2: {r2:.4f}, RMSE: {rmse:.4f}")
        
        results_summary.append({
            'Target': target,
            'Test_R2': r2,
            'Test_RMSE': rmse,
            'Best_Params': str(random_search.best_params_)
        })

    print(f"\n{'='*60}\nFINAL ROBUST TUNING SUMMARY\n{'='*60}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df[['Target', 'Test_R2', 'Test_RMSE']].to_string(index=False))
    
    lowest_r2 = summary_df['Test_R2'].min()
    lowest_target = summary_df.loc[summary_df['Test_R2'] == lowest_r2, 'Target'].values[0]
    print(f"\nLowest Test R2: {lowest_r2:.4f} ({lowest_target})")

if __name__ == "__main__":
    train_robust_regularized("master_dataset_ultimate.csv")

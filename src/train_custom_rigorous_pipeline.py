import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def run_custom_dataset_pipeline(data_path):
    print(f"\n{'='*50}")
    print(f"Executing Strict Spatial Pipeline on Custom Dataset")
    print(f"Target: {data_path}")
    print(f"{'='*50}")

    df = pd.read_csv(data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
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

        # 2. Strict Spatial 70/30 Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        
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

        # 4. Train Model with strict known parameters or new search
        print("Training XGBoost...")
        
        # Note: Since the underlying split heavily changed from random to spatial, 
        # we'll perform a quick Grid Search again to find the true spatial optimal parameters
        model = XGBRegressor(random_state=42, objective='reg:squarederror')
        param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [5, 6],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.8, 1.0]
        }
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='r2',
            cv=cv,
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
        # 5. Evaluate on true spatial holdout set
        y_pred = best_model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Results -> R2: {r2:.4f}, Adj_R2: {adj_r2:.4f}, RMSE: {rmse:.4f}")
        print(f"Best Params: {grid_search.best_params_}")
        
        results_summary.append({
            'Target': target,
            'Test_R2': r2,
            'Test_Adj_R2': adj_r2,
            'Test_RMSE': rmse,
            'Best_Params': str(grid_search.best_params_)
        })

    print(f"\n{'='*50}\nFINAL RESULTS (CUSTOM DATASET - STRICT SPATIAL SPLIT)\n{'='*50}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df[['Target', 'Test_R2', 'Test_Adj_R2', 'Test_RMSE']].to_string(index=False))

if __name__ == "__main__":
    run_custom_dataset_pipeline("final_training_data_enriched.csv")

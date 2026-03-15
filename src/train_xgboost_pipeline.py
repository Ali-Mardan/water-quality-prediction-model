import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def run_pipeline(data_path, dataset_name):
    print(f"\n{'='*50}")
    print(f"Running XGBoost Pipeline for: {dataset_name}")
    print(f"{'='*50}")

    # 1. Load Data
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {data_path} with shape {df.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}")
        return

    # Handle unnamed index columns
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Target variables
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Identify non-feature columns
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin']
    
    # Existing features
    available_features = [col for col in df.columns if col not in features_to_drop]
    
    print(f"\nIdentified {len(available_features)} feature columns.")

    results_summary = []

    for target in targets:
        print(f"\n--- Evaluating Target: {target} ---")
        
        # 1. Clean data for specific target
        df_target = df.dropna(subset=[target]).copy()
        
        X = df_target[available_features].copy()
        y = df_target[target].copy()

        # Handle purely empty columns if any
        X = X.dropna(axis=1, how='all')
        
        # Identify categorical columns (if any exist in the alternative dataset) and encode them
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            print(f"Encoded categorical features: {list(categorical_cols)}")

        # Update available features after dropping totally empty columns and encoding
        current_features = X.columns
        
        # 2. Strict 80/20 Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        print(f"Train set: {X_train.shape[0]} samples. Test set: {X_test.shape[0]} samples.")

        # 3. Preprocessing (Fitted ONLY on Train to prevent Data Leakage)
        # Median Imputation
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians) # Apply train medians to test

        # Standardization 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) # Apply train scaling to test
        
         # 4. GridSearchCV for XGBoost
        xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
        
        param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        print("Starting GridSearchCV...")
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring='r2',
            cv=cv,
            n_jobs=-1, # Use all available cores
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        
        # 5. Out-of-Sample Evaluation on 20% Holdout Test Set
        y_pred = best_model.predict(X_test_scaled)
        
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Calculate Adjusted R2
        n = X_test_scaled.shape[0]
        p = X_test_scaled.shape[1]
        test_adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
        
        print(f"Out-of-Sample Results -> R2: {test_r2:.4f}, Adj_R2: {test_adj_r2:.4f}, RMSE: {test_rmse:.4f}")
        
        results_summary.append({
            'Target': target,
            'Test_R2': test_r2,
            'Test_Adj_R2': test_adj_r2,
            'Test_RMSE': test_rmse,
            'Best_Params': str(grid_search.best_params_)
        })

    print(f"\n--- Summary for {dataset_name} ---")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df[['Target', 'Test_R2', 'Test_Adj_R2', 'Test_RMSE']].to_string(index=False))

if __name__ == "__main__":
    # Dataset 1
    # run_pipeline("final_training_data_enriched.csv", "Enriched Dataset (Custom)")
    
    # Dataset 2
    run_pipeline("shreyasds/training_dataset_enhanced.csv", "Enhanced Dataset (Shreyas)")

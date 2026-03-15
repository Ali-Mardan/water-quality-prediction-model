import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def train_all_features_model_spatial(data_path):
    print(f"\n{'='*50}")
    print(f"Training XGBoost with ALL Features (SPATIAL SPLIT)")
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
        print(f"\n--- Target: {target} ---")
        
        # Select target dataset
        df_target = df.dropna(subset=[target]).copy()
        
        # Use all available features (excluding spatial identifiers / targets dropped at the top file level)
        X = df_target[available_features].copy()
        y = df_target[target].copy()
        groups = df_target['spatial_group'].copy()
        
        # Categorical Encoding (if any top features are categorical)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Spatial 70/30 Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        
        # Preprocessing: Imputation & Scaling fitted strictly on train
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        n = X_test_scaled.shape[0]
        p = X_test_scaled.shape[1]

        # Train with Optimal Hardcoded Parameters
        print(f"Training XGBoost on {len(X.columns)} features with SPATIAL split...")
        print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
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
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Results -> R2: {r2:.4f}, Adj_R2: {adj_r2:.4f}, RMSE: {rmse:.4f}")
        
        results_summary.append({
            'Target': target,
            'Num_Features': len(X.columns),
            'Test_R2': r2,
            'Test_Adj_R2': adj_r2,
            'Test_RMSE': rmse
        })

    print(f"\n{'='*50}\nFINAL RESULTS (SPATIAL SPLIT - ALL FEATURES)\n{'='*50}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    train_all_features_model_spatial("shreyasds/training_dataset_enhanced.csv")

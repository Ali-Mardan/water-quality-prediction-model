import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def run_shreyas_xgboost_8020(data_path):
    print(f"\n{'='*50}")
    print(f"Executing XGBoost on {data_path}")
    print(f"Split: 80/20 Random Split (Strict Imputer/Scaler Isolation)")
    print(f"{'='*50}")

    df = pd.read_csv(data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    features_to_drop = targets + ['Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin']
    # Note: Retaining Latitude and Longitude as features since it's a random split
    available_features = [col for col in df.columns if col not in features_to_drop]

    results_summary = []

    for target in targets:
        print(f"\n--- Target: {target} ---")
        
        df_target = df.dropna(subset=[target]).copy()
        
        X = df_target[available_features].copy()
        y = df_target[target].copy()
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # 80/20 Random Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        
        # Strict Preprocessing Leakage Prevention
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        n = X_test_scaled.shape[0]
        p = X_test_scaled.shape[1]

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
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Results -> R2: {r2:.4f}, Adj_R2: {adj_r2:.4f}, RMSE: {rmse:.4f}")
        
        results_summary.append({
            'Target': target,
            'Test_R2': r2,
            'Test_Adj_R2': adj_r2,
            'Test_RMSE': rmse
        })

    print(f"\n{'='*50}\nFINAL RESULTS (SHREYAS DATASET - 80/20 RANDOM SPLIT)\n{'='*50}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_shreyas_xgboost_8020("shreyasds/training_dataset_enhanced.csv")

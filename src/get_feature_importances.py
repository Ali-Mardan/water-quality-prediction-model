import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def get_feature_importances(data_path):
    print(f"\nExtracting Feature Importances from {data_path}...\n")
    df = pd.read_csv(data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin']
    available_features = [col for col in df.columns if col not in features_to_drop]

    for target in targets:
        df_target = df.dropna(subset=[target]).copy()
        X = df_target[available_features].copy()
        y = df_target[target].copy()
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Fill NAs
        X = X.fillna(X.median())
        
        # Fit model
        model = XGBRegressor(
            random_state=42, 
            objective='reg:squarederror',
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8
        )
        model.fit(X, y)
        
        # Get importances
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
        
        print(f"Top 10 Features for {target}:")
        print("-" * 40)
        for idx, row in importance_df.iterrows():
            print(f"{row['Feature']:<30} {row['Importance']:.4f}")
        print("\n")

if __name__ == "__main__":
    get_feature_importances("shreyasds/training_dataset_enhanced.csv")

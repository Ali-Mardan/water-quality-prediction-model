import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def generate_tuning_plots(master_path):
    print(f"Loading data from {master_path}...")
    df = pd.read_csv(master_path)
    df['spatial_group'] = df['Latitude'].astype(str) + "_" + df['Longitude'].astype(str)
    
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Shared settings
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin', 'spatial_group']
    available_features = [col for col in df.columns if col not in features_to_drop]
    
    # Param range to test
    depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    plt.figure(figsize=(18, 5))
    
    for i, target in enumerate(targets):
        print(f"\nProcessing {target}...")
        df_target = df.dropna(subset=[target]).copy()
        
        X = df_target[available_features].copy()
        y = df_target[target].copy()
        groups = df_target['spatial_group'].copy()
        
        # Spatial 80/20 Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Median imputation (isolated)
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
        
        train_scores = []
        test_scores = []
        
        for d in tqdm(depths):
            model = XGBRegressor(
                n_estimators=100, # Use smaller n_estimators for quick plotting
                max_depth=d,
                learning_rate=0.07,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_scores.append(r2_score(y_train, y_train_pred))
            test_scores.append(r2_score(y_test, y_test_pred))
            
        # Plotting
        plt.subplot(1, 3, i+1)
        plt.plot(depths, train_scores, 'o-', label='Train R2 (Spatial)', color='#2ecc71', lw=2)
        plt.plot(depths, test_scores, 's-', label='Test R2 (Spatial)', color='#e74c3c', lw=2)
        plt.title(f'Tuning Curve: {target}', fontsize=12, fontweight='bold')
        plt.xlabel('Max Depth', fontsize=10)
        plt.ylabel('R2 Score', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.ylim(-0.1, 1.05) if min(test_scores) > -0.1 else plt.ylim(min(test_scores)-0.1, 1.05)
        
    plt.tight_layout()
    plot_path = "tuning_curves_all_targets.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved tuning curves to {plot_path}")

if __name__ == "__main__":
    generate_tuning_plots("master_dataset_ultimate.csv")

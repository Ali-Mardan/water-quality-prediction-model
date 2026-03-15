import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("colorblind")

def plot_eda_spatial_map(df, targets):
    print("Generating EDA Spatial Map...")
    fig, axes = plt.subplots(1, len(targets), figsize=(18, 5))
    
    for i, target in enumerate(targets):
        plot_df = df.dropna(subset=[target, 'Latitude', 'Longitude'])
        sc = axes[i].scatter(plot_df['Longitude'], plot_df['Latitude'], 
                             c=plot_df[target], cmap='viridis', s=20, alpha=0.7, edgecolors='none')
        axes[i].set_title(f'Spatial Distribution:\n{target}', fontweight='bold')
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        plt.colorbar(sc, ax=axes[i], label='Concentration / Value')
        
    plt.tight_layout()
    plt.savefig('eda_spatial_map.png', dpi=300)
    plt.close()

def plot_eda_distributions(df, targets):
    print("Generating EDA Distributions...")
    fig, axes = plt.subplots(1, len(targets), figsize=(18, 5))
    
    for i, target in enumerate(targets):
        plot_df = df.dropna(subset=[target]).copy()
        
        # We use a log1p scale to handle massive outliers visually
        plot_df[f'Log1p_{target}'] = np.log1p(plot_df[target] - plot_df[target].min() + 1)
        
        sns.violinplot(y=plot_df[f'Log1p_{target}'], ax=axes[i], color='skyblue', inner='quartile')
        axes[i].set_title(f'Distribution (Log Scaled):\n{target}', fontweight='bold')
        axes[i].set_ylabel('Log1p Value')
        
    plt.tight_layout()
    plt.savefig('eda_target_distributions.png', dpi=300)
    plt.close()

def plot_correlation_heatmap(df, targets):
    print("Generating Correlation Heatmap...")
    # Select key engineered features + targets
    key_features = [
        'perc_barren', 'perc_urban', 'perc_agri', 'carbonate_index', 
        'topographic_slope', 'ppt_6mo', 'NDSI', 'NDTI', 'soil_clay', 'soil_ph'
    ]
    
    available_cols = [c for c in key_features + targets if c in df.columns]
    corr_df = df[available_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Hydro-Informatics Engineering Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('eda_correlation_heatmap.png', dpi=300)
    plt.close()

def plot_overfitting_curves(df, targets):
    print("Generating Overfitting vs. Robustness Curves...")
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin', 'spatial_group']
    df['spatial_group'] = df['Latitude'].astype(str) + "_" + df['Longitude'].astype(str)
    
    # Simple label encoding for this plot
    df_enc = df.copy()
    for col in df_enc.columns:
        if df_enc[col].dtype == 'object' and col not in features_to_drop + ['spatial_group']:
            df_enc[col] = pd.factorize(df_enc[col])[0]
            
    available_features = [col for col in df_enc.columns if col not in features_to_drop]
    
    depths = [2, 4, 6, 8, 10]
    
    fig, axes = plt.subplots(1, len(targets), figsize=(18, 5))
    
    for i, target in enumerate(targets):
        df_target = df_enc.dropna(subset=[target]).copy()
        X = df_target[available_features]
        y = df_target[target]
        groups = df_target['spatial_group']
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        try:
            train_idx, test_idx = next(gss.split(X, y, groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            medians = X_train.median()
            X_train = X_train.fillna(medians)
            X_test = X_test.fillna(medians)
            
            train_r2 = []
            test_r2 = []
            
            for d in depths:
                # We specifically use low regularization here to demonstrate overfitting
                model = XGBRegressor(max_depth=d, n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                train_r2.append(r2_score(y_train, model.predict(X_train)))
                test_r2.append(r2_score(y_test, model.predict(X_test)))
                
            axes[i].plot(depths, train_r2, 'o-', color='blue', label='Train R2 (Overfits)', lw=2)
            axes[i].plot(depths, test_r2, 's-', color='red', label='Holdout R2 (Crashes)', lw=2)
            axes[i].set_title(f'The Simplicity Paradox:\n{target}', fontweight='bold')
            axes[i].set_xlabel('Tree Depth (Complexity)')
            axes[i].set_ylabel('R2 Score')
            axes[i].grid(True, linestyle='--', alpha=0.6)
            axes[i].legend()
            
        except Exception as e:
            print(f"Skipping {target} overwriting plot: {e}")
            
    plt.tight_layout()
    plt.savefig('model_overfitting_analysis.png', dpi=300)
    plt.close()

def plot_evaluation_metrics(df, targets):
    print("Generating Actual vs. Predicted & Feature Importance...")
    # This requires fully training the robust models once to get the predictions.
    # We will use the optimal parameters found during the robust tuning phase.
    
    optimal_params = {
        'Total Alkalinity': {'subsample': 0.6, 'reg_lambda': 50, 'reg_alpha': 0.5, 'min_child_weight': 20, 'max_depth': 4, 'learning_rate': 0.01},
        'Electrical Conductance': {'subsample': 0.75, 'reg_lambda': 50, 'reg_alpha': 0, 'min_child_weight': 5, 'max_depth': 3, 'learning_rate': 0.01},
        'Dissolved Reactive Phosphorus': {'subsample': 0.75, 'reg_lambda': 50, 'reg_alpha': 5, 'min_child_weight': 10, 'max_depth': 2, 'learning_rate': 0.01}
    }
    
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin', 'spatial_group']
    df['spatial_group'] = df['Latitude'].astype(str) + "_" + df['Longitude'].astype(str)
    
    df_enc = df.copy()
    for col in df_enc.columns:
        if df_enc[col].dtype == 'object' and col not in features_to_drop + ['spatial_group']:
            df_enc[col] = pd.factorize(df_enc[col])[0]
            
    available_features = [col for col in df_enc.columns if col not in features_to_drop]
    
    fig_pred, axes_pred = plt.subplots(1, len(targets), figsize=(18, 5))
    fig_feat, axes_feat = plt.subplots(1, len(targets), figsize=(18, 6))
    fig_res, axes_res = plt.subplots(1, len(targets), figsize=(18, 5))
    
    for i, target in enumerate(targets):
        df_target = df_enc.dropna(subset=[target]).copy()
        X = df_target[available_features]
        y = df_target[target]
        groups = df_target['spatial_group']
        lat_lon = df_target[['Latitude', 'Longitude']]
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        try:
            train_idx, test_idx = next(gss.split(X, y, groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            medians = X_train.median()
            X_train = X_train.fillna(medians)
            X_test = X_test.fillna(medians)
            
            model = XGBRegressor(n_estimators=500, random_state=42, n_jobs=-1, **optimal_params[target])
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            
            # --- Plot 1: Actual vs Predicted ---
            axes_pred[i].scatter(y_test, preds, alpha=0.5, color='#2ecc71', edgecolors='k')
            
            # 1:1 Line
            min_val = min(y_test.min(), preds.min())
            max_val = max(y_test.max(), preds.max())
            axes_pred[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (1:1)')
            
            axes_pred[i].set_title(f'{target}\nActual vs predicted', fontweight='bold')
            axes_pred[i].set_xlabel('Actual Value')
            axes_pred[i].set_ylabel('Predicted Value')
            axes_pred[i].legend()
            
            # --- Plot 2: Feature Importance ---
            importance = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False).head(10)
            sns.barplot(x=importance.values, y=importance.index, ax=axes_feat[i], palette='viridis')
            axes_feat[i].set_title(f'Top 10 Drivers:\n{target}', fontweight='bold')
            axes_feat[i].set_xlabel('XGBoost Gain')
            
            # --- Plot 3: Residual Spatial Map ---
            residuals = np.abs(y_test - preds)
            test_lat_lon = lat_lon.iloc[test_idx]
            sc = axes_res[i].scatter(test_lat_lon['Longitude'], test_lat_lon['Latitude'], 
                                     c=residuals, cmap='magma_r', s=30, alpha=0.8)
            axes_res[i].set_title(f'Spatial Residuals:\n{target}', fontweight='bold')
            axes_res[i].set_xlabel('Longitude')
            axes_res[i].set_ylabel('Latitude')
            plt.colorbar(sc, ax=axes_res[i], label='Absolute Error')
            
        except Exception as e:
            print(f"Skipping evaluation plot for {target}: {e}")
            
    fig_pred.tight_layout()
    fig_pred.savefig('eval_actual_vs_predicted.png', dpi=300)
    
    fig_feat.tight_layout()
    fig_feat.savefig('eval_feature_importance.png', dpi=300)
    
    fig_res.tight_layout()
    fig_res.savefig('eval_spatial_residuals.png', dpi=300)
    
    plt.close('all')

if __name__ == "__main__":
    print("Beginning Comprehensive Visualization Suite Generation...")
    master_path = "master_dataset_ultimate.csv"
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    try:
        df = pd.read_csv(master_path)
    except FileNotFoundError:
        print("master_dataset_ultimate.csv not found!")
        exit(1)
        
    plot_eda_spatial_map(df, targets)
    plot_eda_distributions(df, targets)
    plot_correlation_heatmap(df, targets)
    plot_overfitting_curves(df, targets)
    plot_evaluation_metrics(df, targets)
    
    print("\nVisualizations successfully generated and saved to current directory.")

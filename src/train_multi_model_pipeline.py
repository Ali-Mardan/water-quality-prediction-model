import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def save_visuals(results_df, target_name, y_test, y_pred, model_name):
    """
    Saves visual comparisons for the given target and model.
    """
    os.makedirs('plots', exist_ok=True)
    
    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title(f'Actual vs Predicted: {target_name} ({model_name})')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.grid(True)
    plt.savefig(f'plots/{target_name.replace(" ", "_")}_{model_name}_regplot.png')
    plt.close()

def plot_model_comparison(results_df, targets):
    """
    Plots a bar chart comparing R2 scores for all models across all targets.
    """
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=results_df, x='Target', y='R2', hue='Model')
    plt.title('Out-of-Sample R2 Comparison across Models')
    plt.ylabel('R-Squared Score')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.savefig('plots/model_performance_comparison.png')
    plt.close()

def run_multi_model_pipeline(data_path, dataset_name):
    print(f"\n{'='*50}")
    print(f"Running Multi-Model Pipeline for: {dataset_name}")
    print(f"{'='*50}")

    df = pd.read_csv(data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    features_to_drop = targets + ['Latitude', 'Longitude', 'Sample Date', 'spatial_cluster', 'lat_bin', 'lon_bin']
    available_features = [col for col in df.columns if col not in features_to_drop]

    models_config = {
        'XGBoost': {
            'model': XGBRegressor(
                random_state=42, 
                objective='reg:squarederror',
                n_estimators=1000,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8
            ),
            'grid': None # NO GRID SEARCH
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'grid': {
                'n_estimators': [200, 500],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'LightGBM': {
            'model': LGBMRegressor(random_state=42, verbosity=-1),
            'grid': {
                'n_estimators': [500, 1000],
                'max_depth': [5, 10],
                'learning_rate': [0.01, 0.05],
                'num_leaves': [31, 63]
            }
        }
    }

    overall_results = []

    for target in targets:
        print(f"\n--- Target: {target} ---")
        df_target = df.dropna(subset=[target]).copy()
        X = df_target[available_features].copy()
        y = df_target[target].copy()
        
        # Encoding categorical if any
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        
        # Preprocessing
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        n = X_test_scaled.shape[0]
        p = X_test_scaled.shape[1]

        for model_name, config in models_config.items():
            if config['grid'] is not None:
                print(f"Running GridSearchCV for {model_name}...")
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                model_to_use = GridSearchCV(config['model'], config['grid'], cv=cv, scoring='r2', n_jobs=-1)
                model_to_use.fit(X_train_scaled, y_train)
                best_model = model_to_use.best_estimator_
                best_params_str = str(model_to_use.best_params_)
            else:
                print(f"Training {model_name} with pre-defined parameters...")
                model_to_use = config['model']
                model_to_use.fit(X_train_scaled, y_train)
                best_model = model_to_use
                best_params_str = "Pre-defined"
                
            y_pred = best_model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Save visual for this specific model/target
            save_visuals(None, target, y_test, y_pred, model_name)
            
            print(f"  {model_name} Results -> R2: {r2:.4f}, Adj_R2: {adj_r2:.4f}, RMSE: {rmse:.4f}")
            
            overall_results.append({
                'Target': target,
                'Model': model_name,
                'R2': r2,
                'Adj_R2': adj_r2,
                'RMSE': rmse,
                'BestParams': best_params_str
            })

    print(f"\n{'='*50}\nFINAL MODEL COMPARISON\n{'='*50}")
    results_df = pd.DataFrame(overall_results)
    
    # Generate overall model comparison plot
    plot_model_comparison(results_df, targets)
    
    for target in targets:
        print(f"\n--- {target} Comparison ---")
        display_df = results_df[results_df['Target'] == target]
        print(display_df[['Model', 'R2', 'Adj_R2', 'RMSE']].sort_values(by='R2', ascending=False).to_string(index=False))
    
    print("\nVisuals saved in the 'plots/' directory.")

if __name__ == "__main__":
    run_multi_model_pipeline("shreyasds/training_dataset_enhanced.csv", "Enhanced Dataset (Shreyas)")

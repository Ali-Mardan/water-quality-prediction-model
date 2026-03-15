import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

# Load data
enriched_data = pd.read_csv('final_training_data_enriched.csv')
enriched_data['Sample Date'] = pd.to_datetime(enriched_data['Sample Date'])

# Select features
# Using available features from all sources
features = [
    'pet', 'ppt', 'tmax', 'tmin', 'soil', 'def', 
    'ppt_3mo', 'ppt_6mo', 'tmax_3mo', 'T_range',
    'Month', 'DayOfYear'
]
# Add Landsat features if they are populated (at least some)
landsat_feat = ['red', 'blue', 'green', 'nir', 'swir16', 'swir22', 'NDVI', 'SI2', 'NDTI', 'MNDWI', 'NDMI']
for f in landsat_feat:
    if f in enriched_data.columns:
        features.append(f)

print(f"Features used: {features}")

X = enriched_data[features]
y_TA = enriched_data['Total Alkalinity']
y_EC = enriched_data['Electrical Conductance']
y_DRP = enriched_data['Dissolved Reactive Phosphorus']

# Fill NaNs for the sake of preliminary evaluation
# (Landsat data is partially missing right now)
X = X.fillna(X.median())

models_to_try = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': GradientBoostingRegressor(n_estimators=100, random_state=42), # Simplified XGB as GB
    'Ridge': Ridge(random_state=42)
}

def evaluate(X, y, param_name):
    # Filter rows where target is not NaN
    valid_idx = y.notna()
    X_v = X[valid_idx]
    y_v = y[valid_idx]
    
    print(f"--- {param_name} (Samples: {len(y_v)}) ---")
    X_train, X_test, y_train, y_test = train_test_split(X_v, y_v, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    for name, model in models_to_try.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"{name:20s}: R2 = {r2:.3f}, RMSE = {rmse:.3f}")
    print()

evaluate(X, y_TA, "Total Alkalinity")
evaluate(X, y_EC, "Electrical Conductance")
evaluate(X, y_DRP, "Dissolved Reactive Phosphorus")

import json
import nbformat
import sys

def main():
    with open('Benchmark_Model_Notebook.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Update imports
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'from sklearn.ensemble import RandomForestRegressor' in cell.source:
            if 'GradientBoostingRegressor' not in cell.source:
                cell.source = cell.source.replace(
                    'from sklearn.ensemble import RandomForestRegressor',
                    'from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\nfrom sklearn.linear_model import Ridge'
                )

    # 2. Update model training cell
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'def train_model' in cell.source:
            new_source = """
def train_model(X_train_scaled, y_train, model_type="gb"):
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "ridge":
        model = Ridge(random_state=42)
    else:  # Default to GradientBoosting
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    return model
"""
            cell.source = new_source.strip()
            
    # 3. Update the execution pipeline cell to include tests for all models
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'def run_pipeline' in cell.source:
            new_source = """
def run_pipeline(X, y, param_name="Parameter", model_type="gb"):
    print(f"\\n{'='*60}")
    print(f"Training {model_type.upper()} Model for {param_name}")
    print(f"{'='*60}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Train
    model = train_model(X_train_scaled, y_train, model_type)
    
    # Evaluate (in-sample)
    y_train_pred, r2_train, rmse_train = evaluate_model(model, X_train_scaled, y_train, "Train")
    
    # Evaluate (out-sample)
    y_test_pred, r2_test, rmse_test = evaluate_model(model, X_test_scaled, y_test, "Test")
    
    # Return summary
    results = {
        "Parameter": param_name,
        "Model": model_type.upper(),
        "R2_Train": r2_train,
        "RMSE_Train": rmse_train,
        "R2_Test": r2_test,
        "RMSE_Test": rmse_test
    }
    return model, scaler, pd.DataFrame([results])
"""
            cell.source = new_source.strip()

    # 4. Find the application cell and update to test multiple models
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'y_DRP = wq_data' in cell.source:
            new_source = """
X = wq_data[['swir22','NDMI','MNDWI','pet']]

y_TA = wq_data['Total Alkalinity']
y_EC = wq_data['Electrical Conductance']
y_DRP = wq_data['Dissolved Reactive Phosphorus']

# You can change model_type here: 'gb' limit memory/speed (GradientBoosting), 'rf' (RandomForest), 'ridge'
model_TA, scaler_TA, results_TA = run_pipeline(X, y_TA, "Total Alkalinity", model_type="gb")
model_EC, scaler_EC, results_EC = run_pipeline(X, y_EC, "Electrical Conductance", model_type="gb")
model_DRP, scaler_DRP, results_DRP = run_pipeline(X, y_DRP, "Dissolved Reactive Phosphorus", model_type="gb")
"""
            cell.source = new_source.strip()
            
    with open('Benchmark_Model_Notebook.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    main()

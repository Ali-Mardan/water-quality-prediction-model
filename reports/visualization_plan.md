# Comprehensive Visualization Implementation Plan

This document outlines the detailed plan to programmatically generate a massive suite of visuals supporting the Technical Report for the EY Water Quality Pipeline.

## 1. Exploratory Data Analysis (EDA) Visualizations
*Goal: Prove we understand the underlying heterogeneities and physical distributions within the dataset before modeling.*

1.  **Geospatial Target Heatmap (The Map)**
    *   **Plot Type**: Spatial Scatter Plot (`plt.scatter` using longitude/latitude).
    *   **Data**: `Latitude`, `Longitude`, colored by target values (e.g., `Electrical Conductance` and `Dissolved Reactive Phosphorus`).
    *   **Purpose**: Visually confirms the clustering of sampling stations and highlights regions with acute pollution (e.g., downstream of mining hotspots).
    *   **Filename**: `eda_spatial_map.png`

2.  **Target Distribution & Outlier Analysis**
    *   **Plot Type**: Violin Plot or Boxen Plot with superimposed swarm distributions.
    *   **Data**: The three targets (TA, EC, DRP) plotted on logarithmic scales.
    *   **Purpose**: Demonstrates the massive skewness and physical outliers present in natural water sampling, justifying the use of robust tree-based models over linear regression.
    *   **Filename**: `eda_target_distributions.png`

3.  **Hydro-Informatics Correlation Clustermap**
    *   **Plot Type**: Seaborn `clustermap` (Hierarchical clustered heatmap).
    *   **Data**: Top 15 engineered features (e.g., `perc_barren`, `carbonate_index`, `topographic_slope`, `ppt_6mo`) vs. the targets.
    *   **Purpose**: Proves that the "secondary proxies" extracted from satellites actually contain mathematically significant correlations with the optically invisible targets.
    *   **Filename**: `eda_correlation_clustermap.png`

## 2. Hyperparameter Tuning & Overfitting Analysis
*Goal: Visually document the story of how we arrested spatial leakage via the "Simplicity Paradox".*

4.  **The Overfitting vs. Robustness Curve**
    *   **Plot Type**: Multi-line Validation Curve.
    *   **Data**: Train vs. Spatial Test $R^2$ across varying `max_depth` (2 to 10), plotted for all three models.
    *   **Purpose**: Directly addresses Technical Hurdle 2. It visually demonstrates that while deep trees achieve 0.99 Training $R^2$, they crash on the spatial holdout array. It proves our selection of `max_depth=3` or `4` is optimal.
    *   **Filename**: `model_overfitting_analysis.png`

5.  **Regularization Impact Surface (Optional but powerful)**
    *   **Plot Type**: 2D Contour Plot.
    *   **Data**: `max_depth` vs. `reg_lambda` (L2 regularization), colored by Test $R^2$.
    *   **Purpose**: Shows the distinct "safe zone" of hyperparameters that generalize well.
    *   **Filename**: `model_regularization_surface.png`

## 3. Progress, Results, and Final Evaluation
*Goal: Provide irrefutable visual proof of model competence on the spatial holdout set.*

6.  **Actual vs. Predicted Evaluation Plot**
    *   **Plot Type**: Scatter plot with a $y=x$ (perfect prediction) identity line and a linear regression fit line.
    *   **Data**: Actual holdout values vs. XGBoost predicted values.
    *   **Purpose**: Exposes exactly where the model succeeds and where heteroscedasticity occurs (e.g., underpredicting the extreme 99th percentile of Phosphorus spikes).
    *   **Filename**: `eval_actual_vs_predicted.png`

7.  **Holdout Residuals Spatial Map**
    *   **Plot Type**: Spatial Scatter Plot of prediction errors.
    *   **Data**: `Latitude`, `Longitude`, colored by absolute prediction error ($|y - \hat{y}|$).
    *   **Purpose**: Shows if the model fails in specific geographical regions (e.g., high error only in coastal areas), proving the rigorousness of the Spatial Group Split.
    *   **Filename**: `eval_spatial_residuals.png`

8.  **Global Feature Importance Summary (SHAP style or Gain)**
    *   **Plot Type**: Horizontal Bar Chart of XGBoost native `Gain` importance.
    *   **Data**: Top 15 drivers across all models.
    *   **Purpose**: Scientifically validates the hydro-informatics pipeline by proving that domain-specific features (Topographic Slope, Mining Proxies) hold the highest predictive power, not just generic coordinates.
    *   **Filename**: `eval_feature_importance.png`

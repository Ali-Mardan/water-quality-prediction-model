# EY Datathon: Water Quality Prediction ML Project

This repository contains a comprehensive Machine Learning pipeline developed for the **EY Datathon**, focusing on water quality prediction using satellite imagery (Landsat) and climate data (TerraClimate).

## 🚀 Project Overview

The project leverages spatial and temporal features to predict water quality indicators. It involves complex data extraction, feature engineering (including advanced landscape features), and multi-model training pipelines (XGBoost, LSTM, TimeGPT).

## 📁 Repository Structure

The project is organized into a standard MLOps-ready structure:

- **`data/`**: 
    - `raw/`: Initial datasets from Landsat, TerraClimate, and water quality ground truth.
    - `processed/`: Cleaned, enriched, and encoded features ready for modeling.
- **`notebooks/`**: Jupyter notebooks for data exploration, extraction, and demonstration.
- **`src/`**: Python scripts for data preprocessing, training pipelines, model evaluation, and utilities.
- **`models/`**: Model weights, checkpoints, and final submission artifacts.
- **`reports/`**: Performance visualizations, overfitting analysis, and technical summaries.

## 🛠️ Key Features

- **Multi-Source Data Fusion**: Integrates Landsat spectral bands with TerraClimate meteorological data.
- **Advanced Feature Engineering**: Extraction of spatial landscape features and temporal sequences.
- **Robust Modeling**: Compares standard XGBoost models with sequential models like LSTM and Nilogla's TimeGPT.
- **Spatial Validation**: Implements rigorous spatial cross-validation to ensure model generalization.

## 📊 Results Summary

The best performing model (XGBoost production pipeline) achieved significant predictive accuracy, with detailed performance metrics and tuning curves available in the `reports/` directory.

---
*Created as part of the EY Datathon competition.*

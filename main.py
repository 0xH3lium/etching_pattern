"""
Main script to orchestrate the surface feature extraction and classification pipeline.
"""

import sys
import os
# Add the src directory to the path so we can import from subdirectories
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dataset_creation import create_dataset_from_directory
from features.feature_selection import select_features
from models.models import train_random_forest_kfold, train_xgboost_kfold, train_ann_kfold
from visualization.visualization import visualize_dataset_samples, plot_learning_curves
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def main():
    """Main pipeline function to orchestrate the entire surface feature extraction and classification process."""
    # Configuration
    DATASET_DIRECTORY = './data/augmented_dataset'
    OUTPUT_CSV_FILE = './data/processed/surface_features_dataset.csv'
    SELECTED_CSV_FILE = './data/processed/surface_features_processed.csv'

    # Step 1: Create dataset from images
    print("=== Step 1: Creating Dataset ===")
    if not os.path.exists(OUTPUT_CSV_FILE):
        if not os.path.isdir(DATASET_DIRECTORY):
            print(f"Error: Dataset directory '{DATASET_DIRECTORY}' not found.")
            return
        feature_df = create_dataset_from_directory(DATASET_DIRECTORY, OUTPUT_CSV_FILE)
    else:
        print(f"Dataset file '{OUTPUT_CSV_FILE}' already exists. Skipping dataset creation.")

    # Step 2: Feature selection
    print("\n=== Step 2: Feature Selection ===")
    if not os.path.exists(SELECTED_CSV_FILE):
        full_df = pd.read_csv(OUTPUT_CSV_FILE)
        manual_selected_df = select_features(full_df, method='manual')
        manual_selected_df.to_csv(SELECTED_CSV_FILE, index=False)
        print(f"Selected features saved to '{SELECTED_CSV_FILE}'")
    else:
        print(f"Selected features file '{SELECTED_CSV_FILE}' already exists. Skipping feature selection.")

    # Step 3: Visualize dataset samples (optional)
    print("\n=== Step 3: Visualizing Dataset Samples ===")
    # visualize_dataset_samples(OUTPUT_CSV_FILE, DATASET_DIRECTORY, num_samples=2)

    # Step 4: Train models
    print("\n=== Step 4: Training Models ===")

    # Train Random Forest
    print("\n--- Training Random Forest ---")
    train_random_forest_kfold(SELECTED_CSV_FILE)

    # Train XGBoost
    print("\n--- Training XGBoost ---")
    train_xgboost_kfold(SELECTED_CSV_FILE)

    # Train ANN
    print("\n--- Training ANN ---")
    train_ann_kfold(SELECTED_CSV_FILE)

    # Step 5: Plot learning curves
    print("\n=== Step 5: Plotting Learning Curves ===")
    plot_learning_curves_for_models(SELECTED_CSV_FILE)

    print("\n=== Pipeline Complete ===")


def plot_learning_curves_for_models(dataset_path: str):
    """
    Plot learning curves for Random Forest and XGBoost models.
    
    Args:
        dataset_path: Path to the processed dataset CSV file
    """
    # Load and prepare data
    df = pd.read_csv(dataset_path)
    df = df.drop(columns=['filename'])
    df = df.dropna()
    
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            max_features='sqrt',
            min_samples_leaf=1,
            min_samples_split=2,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.15,
            max_depth=4,
            subsample=1.0,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=42
        )
    }
    
    # Plot learning curves
    plot_learning_curves(models, X_scaled, y_encoded)


if __name__ == "__main__":
    main()
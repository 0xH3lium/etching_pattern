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
import os
import pandas as pd


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
        create_dataset_from_directory(DATASET_DIRECTORY, OUTPUT_CSV_FILE)
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

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
"""
Feature Brute Force Script to find the best feature combinations for Random Forest and XGBoost.

This script systematically tests all possible feature combinations within a given range
to identify which subset gives the best classification performance.
"""
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import os
import time
from tqdm import tqdm


def load_and_prepare_data(dataset_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load and prepare data for feature selection.
    
    Args:
        dataset_path: Path to the dataset CSV file
        
    Returns:
        Tuple of (full_dataframe, X_features, y_labels)
    """
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Remove filename, and any other non-feature columns
    df = df.drop(columns=['filename'], errors='ignore')
    
    # Separate features and labels
    X = df.drop(columns=['label'])
    y = df['label']
    
    print(f"Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, y_encoded


def train_model_cv(
    X: np.ndarray, 
    y: np.ndarray, 
    model_type: str = 'random_forest',
    k_folds: int = 5
) -> float:
    """
    Train a model with cross-validation and return mean accuracy.
    
    Args:
        X: Feature matrix
        y: Labels
        model_type: Type of model ('random_forest' or 'xgboost')
        k_folds: Number of cross-validation folds
        
    Returns:
        Mean cross-validation accuracy
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_accuracies = []
    
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_features='sqrt'
            )
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                eval_metric='mlogloss',
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError("model_type must be 'random_forest' or 'xgboost'")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        fold_accuracies.append(accuracy)
    
    return np.mean(fold_accuracies)


def brute_force_features(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    min_features: int = 2,
    max_features: int = 5,
    model_type: str = 'random_forest',
    k_folds: int = 5,
    max_combinations: int = 1000  # Limit for safety
) -> Dict[str, any]:
    """
    Brute force test all possible feature combinations within the given range.
    
    Args:
        df: Original dataframe to get feature names
        X: Scaled feature matrix
        y: Encoded labels
        min_features: Minimum number of features in combination
        max_features: Maximum number of features in combination
        model_type: Type of model ('random_forest' or 'xgboost')
        k_folds: Number of cross-validation folds
        max_combinations: Maximum number of combinations to test (for computational limits)
        
    Returns:
        Dictionary with best results and statistics
    """
    feature_names = df.drop(columns=['label']).columns.tolist()
    n_features = len(feature_names)
    
    print(f"Starting brute force search for {model_type}...")
    print(f"Total features available: {n_features}")
    print(f"Testing combinations from {min_features} to {max_features} features")
    
    best_accuracy = 0.0
    best_features = []
    best_combination_count = 0
    
    results = []
    
    total_combinations = 0
    for n in range(min_features, min(max_features + 1, n_features + 1)):
        total_combinations += len(list(combinations(range(n_features), n)))
    
    print(f"Total possible combinations to evaluate: {total_combinations}")
    
    # Limit combinations for practicality
    if total_combinations > max_combinations:
        print(f"WARNING: Limiting to first {max_combinations} combinations for efficiency")
    
    combination_count = 0
    
    # Iterate through all possible feature counts
    for n in range(min_features, min(max_features + 1, n_features + 1)):
        print(f"\nTesting {n}-feature combinations...")
        feature_indices = list(combinations(range(n_features), n))
        
        for indices in tqdm(feature_indices, desc=f"{n}-feature combinations"):
            if combination_count >= max_combinations:
                break
                
            # Extract the selected features
            X_subset = X[:, indices]
            
            # Train model with cross-validation
            accuracy = train_model_cv(X_subset, y, model_type, k_folds)
            
            # Store results
            feature_names_subset = [feature_names[i] for i in indices]
            results.append({
                'features': feature_names_subset,
                'accuracy': accuracy,
                'num_features': len(feature_names_subset)
            })
            
            # Update best if current is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = feature_names_subset
                best_combination_count = combination_count
                
            combination_count += 1
            
            if combination_count >= max_combinations:
                break
    
    # Sort results by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    return {
        'results': results,
        'best_accuracy': best_accuracy,
        'best_features': best_features,
        'best_combination_count': best_combination_count,
        'total_evaluated': combination_count
    }


def print_results_summary(result: Dict[str, any], model_type: str):
    """Print a summary of the brute force results."""
    print(f"\n{'='*50}")
    print(f"BRUTE FORCE RESULTS FOR {model_type.upper()}")
    print(f"{'='*50}")
    print(f"Total combinations evaluated: {result['total_evaluated']}")
    print(f"Best accuracy: {result['best_accuracy']:.4f}")
    print(f"Best {len(result['best_features'])}-feature combination: {result['best_features']}")
    
    print(f"\nTop 5 feature combinations:")
    for i, res in enumerate(result['results'][:5]):
        print(f"{i+1}. Accuracy: {res['accuracy']:.4f}, Features: {res['features']}")


def main():
    """Main function to run the feature brute force analysis."""
    # Configuration
    DATASET_PATH = './data/processed/surface_features_processed.csv'
    MIN_FEATURES = 2
    MAX_FEATURES = 5
    K_FOLDS = 5
    MAX_COMBINATIONS = 100  # Set to a smaller number for demo purposes
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset file {DATASET_PATH} not found!")
        print("Please run the main pipeline first to create the dataset.")
        return
    
    # Load and prepare data
    df, X, y = load_and_prepare_data(DATASET_PATH)
    
    # Test Random Forest
    print("Testing Random Forest...")
    start_time = time.time()
    rf_result = brute_force_features(
        df, X, y, 
        min_features=MIN_FEATURES, 
        max_features=MAX_FEATURES, 
        model_type='random_forest',
        k_folds=K_FOLDS,
        max_combinations=MAX_COMBINATIONS
    )
    rf_time = time.time() - start_time
    
    # Test XGBoost
    print("\nTesting XGBoost...")
    start_time = time.time()
    xgb_result = brute_force_features(
        df, X, y, 
        min_features=MIN_FEATURES, 
        max_features=MAX_FEATURES, 
        model_type='xgboost',
        k_folds=K_FOLDS,
        max_combinations=MAX_COMBINATIONS
    )
    xgb_time = time.time() - start_time
    
    # Print results
    print_results_summary(rf_result, 'Random Forest')
    print(f"Random Forest evaluation took {rf_time:.2f} seconds")
    
    print_results_summary(xgb_result, 'XGBoost')
    print(f"XGBoost evaluation took {xgb_time:.2f} seconds")
    
    # Compare best models
    print(f"\n{'='*50}")
    print("FINAL COMPARISON")
    print(f"{'='*50}")
    print(f"Best Random Forest: {rf_result['best_accuracy']:.4f} with {len(rf_result['best_features'])} features")
    print(f"  Features: {rf_result['best_features']}")
    print(f"Best XGBoost: {xgb_result['best_accuracy']:.4f} with {len(xgb_result['best_features'])} features") 
    print(f"  Features: {xgb_result['best_features']}")
    
    if rf_result['best_accuracy'] > xgb_result['best_accuracy']:
        print("\nRandom Forest performed better!")
    else:
        print("\nXGBoost performed better!")


if __name__ == "__main__":
    main()
"""
Simplified Feature Brute Force Script for Random Forest and XGBoost.

This script systematically tests various feature combinations to identify 
the best subset for classification performance, with more efficient computation.
"""
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import os
from tqdm import tqdm


def load_and_prepare_data(dataset_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Load and prepare data for feature selection.
    
    Args:
        dataset_path: Path to the dataset CSV file
        
    Returns:
        Tuple of (full_dataframe, X_features, y_labels, feature_names)
    """
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Remove filename, and any other non-feature columns
    df = df.drop(columns=['filename'], errors='ignore')
    
    # Separate features and labels
    X_df = df.drop(columns=['label'])
    y = df['label']
    
    print(f"Dataset loaded with {X_df.shape[0]} samples and {X_df.shape[1]} features.")
    
    # Get feature names
    feature_names = X_df.columns.tolist()
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    
    return df, X_scaled, y_encoded, feature_names


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
                n_estimators=50,  # Reduced for faster computation
                random_state=42,
                n_jobs=-1,
                max_features='sqrt'
            )
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=50,  # Reduced for faster computation
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


def find_best_feature_combinations(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    min_features: int = 2,
    max_features: int = 4,
    model_type: str = 'random_forest',
    k_folds: int = 3,  # Reduced for faster computation
    max_combinations_per_size: int = 50  # Limit combinations per feature set size
) -> Dict[str, any]:
    """
    Find the best feature combinations for the given model.
    
    Args:
        df: Original dataframe to get feature names
        X: Scaled feature matrix
        y: Encoded labels
        feature_names: List of feature names
        min_features: Minimum number of features in combination
        max_features: Maximum number of features in combination
        model_type: Type of model ('random_forest' or 'xgboost')
        k_folds: Number of cross-validation folds
        max_combinations_per_size: Maximum number of combinations to test per feature count
        
    Returns:
        Dictionary with best results and statistics
    """
    n_features = len(feature_names)
    
    print(f"Starting feature combination search for {model_type}...")
    print(f"Available features ({n_features}): {feature_names}")
    print(f"Testing combinations from {min_features} to {max_features} features")
    print(f"Max {max_combinations_per_size} combinations per feature size for efficiency")
    
    best_accuracy = 0.0
    best_features = []
    all_results = []
    
    # Iterate through all possible feature counts
    for n in range(min_features, min(max_features + 1, n_features + 1)):
        print(f"\nTesting {n}-feature combinations...")
        
        # Get all possible combinations of this size
        all_combinations = list(combinations(range(n_features), n))
        
        # Limit the number of combinations to test for this size
        combinations_to_test = all_combinations[:max_combinations_per_size]
        print(f"Testing {len(combinations_to_test)} out of {len(all_combinations)} possible {n}-feature combinations")
        
        for indices in tqdm(combinations_to_test, desc=f"Testing {n}-feature combinations"):
            # Extract the selected features
            X_subset = X[:, indices]
            
            # Train model with cross-validation
            accuracy = train_model_cv(X_subset, y, model_type, k_folds)
            
            # Store results
            feature_names_subset = [feature_names[i] for i in indices]
            result = {
                'features': feature_names_subset,
                'feature_indices': indices,
                'accuracy': accuracy,
                'num_features': len(feature_names_subset)
            }
            all_results.append(result)
            
            # Update best if current is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = feature_names_subset
        
        print(f"Best accuracy for {n}-feature combinations: {best_accuracy:.4f}")
    
    # Sort results by accuracy
    all_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    return {
        'all_results': all_results,
        'best_accuracy': best_accuracy,
        'best_features': best_features,
        'model_type': model_type
    }


def compare_models_and_features(
    dataset_path: str,
    min_features: int = 2,
    max_features: int = 4
):
    """
    Compare Random Forest and XGBoost with different feature combinations.
    
    Args:
        dataset_path: Path to the dataset CSV file
        min_features: Minimum number of features to test
        max_features: Maximum number of features to test
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset file {dataset_path} not found!")
        print("Please run the main pipeline first to create the dataset.")
        return
    
    # Load and prepare data
    df, X, y, feature_names = load_and_prepare_data(dataset_path)
    
    print(f"\n{'='*60}")
    print("FEATURE COMBINATION OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Features range: {min_features} to {max_features}")
    print(f"Total features: {len(feature_names)}")
    print(f"Samples: {X.shape[0]}")
    
    # Test Random Forest
    print(f"\n{'='*30}")
    print("TESTING RANDOM FOREST")
    print(f"{'='*30}")
    rf_result = find_best_feature_combinations(
        df, X, y, feature_names,
        min_features=min_features, 
        max_features=max_features, 
        model_type='random_forest',
        k_folds=3,
        max_combinations_per_size=200
    )
    
    # Test XGBoost
    print(f"\n{'='*30}")
    print("TESTING XGBOOST")
    print(f"{'='*30}")
    xgb_result = find_best_feature_combinations(
        df, X, y, feature_names,
        min_features=min_features, 
        max_features=max_features, 
        model_type='xgboost',
        k_folds=3,
        max_combinations_per_size=200
    )
    
    # Display results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    print(f"\nRANDOM FOREST BEST RESULT:")
    print(f"Accuracy: {rf_result['best_accuracy']:.4f}")
    print(f"Features: {rf_result['best_features']}")
    
    print(f"\nXGBOOST BEST RESULT:")
    print(f"Accuracy: {xgb_result['best_accuracy']:.4f}")
    print(f"Features: {xgb_result['best_features']}")
    
    # Compare models
    if rf_result['best_accuracy'] > xgb_result['best_accuracy']:
        print(f"\n🎉 Random Forest performed better!")
        print(f"Best combo: {rf_result['best_features']} (Accuracy: {rf_result['best_accuracy']:.4f})")
    else:
        print(f"\n🏆 XGBoost performed better!")
        print(f"Best combo: {xgb_result['best_features']} (Accuracy: {xgb_result['best_accuracy']:.4f})")
    
    # Show top 3 combinations for each model
    print(f"\nTOP 3 COMBINATIONS FOR RANDOM FOREST:")
    for i, res in enumerate(rf_result['all_results'][:3]):
        print(f"{i+1}. {res['accuracy']:.4f}: {res['features']}")
    
    print(f"\nTOP 3 COMBINATIONS FOR XGBOOST:")
    for i, res in enumerate(xgb_result['all_results'][:3]):
        print(f"{i+1}. {res['accuracy']:.4f}: {res['features']}")


def main():
    """Main function to run the feature combination optimization."""
    # Configuration
    DATASET_PATH = './data/processed/surface_features_processed.csv'
    MIN_FEATURES = 5
    MAX_FEATURES = 10  # Limited for computational efficiency
    
    compare_models_and_features(DATASET_PATH, MIN_FEATURES, MAX_FEATURES)


if __name__ == "__main__":
    main()
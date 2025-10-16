

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def perform_rfecv(df: pd.DataFrame, estimator=None, estimator_name: str = 'random_forest', cv: int = 3, scoring: str = 'accuracy', n_jobs: int = -1, max_features: Optional[int] = None) -> List[str]:
    """
    Perform Recursive Feature Elimination with Cross-Validation (RFECV).

    Args:
        df: pandas DataFrame containing the full dataset
        estimator: sklearn estimator object (optional, if provided, estimator_name is ignored)
        estimator_name: name of the estimator to use ('random_forest', 'xgboost', or 'logistic_regression')
        cv: number of cross-validation folds
        scoring: scoring metric for cross-validation
        n_jobs: number of parallel jobs (-1 for all processors)
        max_features: maximum number of features to select (optional)

    Returns:
        List of selected feature names
    """
    # Define features to exclude (non-numeric or metadata)
    exclude_features = ['filename', 'label']
    
    # Separate feature columns from metadata
    feature_columns = [col for col in df.columns if col not in exclude_features]
    X = df[feature_columns]
    y = df['label']
    
    # Encode labels if they are not numeric
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Create the estimator based on the specified name if not provided
    if estimator is None:
        if estimator_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=n_jobs)
        elif estimator_name == 'xgboost':
            from xgboost import XGBClassifier
            estimator = XGBClassifier(n_estimators=50, random_state=42, n_jobs=n_jobs)
        elif estimator_name == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(random_state=42, max_iter=1000, n_jobs=n_jobs)
        else:
            raise ValueError("estimator_name must be 'random_forest', 'xgboost', or 'logistic_regression'")
    # Otherwise, use the provided estimator
    
    # Initialize RFECV with StratifiedKFold to maintain class distribution
    rfecv = RFECV(
        estimator=estimator,
        step=5,  # Remove 5 features at a time for speedup
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        n_jobs=n_jobs,
        min_features_to_select=1,
        verbose=1  # Add logging to show each iteration
    )
    
    # Fit RFECV
    print("Performing Recursive Feature Elimination with Cross-Validation...")
    rfecv.fit(X, y_encoded)
    
    # Get selected features
    selected_features = X.columns[rfecv.support_].tolist()
    
    # If max_features is specified and we have more than that, select top max_features by ranking
    if max_features and len(selected_features) > max_features:
        rankings = rfecv.ranking_
        # Get indices of selected features, sorted by ranking (lower is better)
        selected_indices = [i for i, support in enumerate(rfecv.support_) if support]
        ranked_indices = sorted(selected_indices, key=lambda i: rankings[i])
        top_indices = ranked_indices[:max_features]
        selected_features = X.columns[top_indices].tolist()
    
    # Print results
    print(f"Optimal number of features: {len(selected_features)}")
    
    # Create a plot showing the cross-validation score for different numbers of features
    plt.figure(figsize=(10, 6))
    if hasattr(rfecv, 'cv_results_'):
        # For newer sklearn versions
        scores = rfecv.cv_results_['mean_test_score']
        plt.plot(range(1, len(scores) + 1), scores)
        print(f"Best cross-validation score: {max(scores):.4f}")
    elif hasattr(rfecv, 'grid_scores_'):
        # For older sklearn versions
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        print(f"Best cross-validation score: {max(rfecv.grid_scores_):.4f}")
    else:
        print("Could not plot results: missing cv_scores or grid_scores attribute")
        plt.plot([])  # Empty plot
    plt.title('Recursive Feature Elimination with Cross-Validation')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Cross-Validation Score')
    plt.grid(True)
    plt.show()
    
    return selected_features


def select_features_rfecv(df: pd.DataFrame, 
                         method: str = 'random_forest', 
                         cv: int = 3, 
                         scoring: str = 'accuracy', 
                         n_jobs: int = -1,
                         estimator=None,
                         feature_columns: Optional[List[str]] = None,
                         max_features: Optional[int] = None) -> pd.DataFrame:
    """
    Select features using Recursive Feature Elimination with Cross-Validation (RFECV).
    
    Args:
        df: pandas DataFrame containing the dataset
        method: estimator name ('random_forest', 'xgboost', 'logistic_regression') or 'custom' if estimator is provided
        cv: number of cross-validation folds
        scoring: scoring metric for cross-validation
        n_jobs: number of parallel jobs
        estimator: custom sklearn estimator (optional)
        feature_columns: list of feature column names to use (optional, if None, will auto-detect)
        max_features: maximum number of features to select (optional)

    Returns:
        pandas DataFrame with selected features plus filename and label columns
    """
    # Define features to exclude (non-numeric or metadata)
    exclude_features = ['filename', 'label']

    if feature_columns is None:
        # Separate feature columns from metadata
        feature_columns = [col for col in df.columns if col not in exclude_features]
    
    # Get selected features using RFECV
    selected_features = perform_rfecv(
        df, 
        estimator=estimator, 
        estimator_name=method, 
        cv=cv, 
        scoring=scoring, 
        n_jobs=n_jobs,
        max_features=max_features
    )
    
    # Create new dataframe with selected features plus metadata
    selected_df = df[['filename', 'label'] + selected_features].copy()
    
    print(f"Selected {len(selected_features)} features using RFECV with {method if estimator is None else 'custom estimator'}:")
    print(selected_features)
    
    return selected_df


def select_features(df: pd.DataFrame, method: str = 'manual', n_features: int = 20, exclude_features: Optional[List[str]] = None, max_features: Optional[int] = None) -> pd.DataFrame:
    """
    Select a subset of features from the dataset.

    Args:
        df: pandas DataFrame containing the full dataset
        method: 'manual', 'variance', 'correlation', or 'rfecv'
        n_features: number of features to select (for variance/correlation methods)
        exclude_features: list of feature names to exclude
        max_features: maximum number of features to select for rfecv method

    Returns:
        pandas DataFrame with selected features plus filename and label columns
    """

    # Define features to exclude (non-numeric or metadata)
    default_exclude = ['filename', 'label']
    if exclude_features:
        exclude_features = list(set(default_exclude + exclude_features))
    else:
        exclude_features = default_exclude

    # Separate feature columns from metadata
    feature_columns = [col for col in df.columns if col not in exclude_features]
    feature_data = df[feature_columns]

    if method == 'manual':
        # Manually selected features based on domain knowledge
        """
        selected_features = [
            'mean', 'std_dev', 'skewness', 'kurt',
            'num_blobs', 'mean_blob_size',
            'lbp_entropy',
            'gradient_orientation_entropy', 'mean_gradient_magnitude',
        ]
        """
        selected_features = ['mean', 'std_dev', 'skewness', 'kurt', 'mean_blob_size', 'lbp_hist_0', 'hu_moment_1']

        # Add FFT features (you might want to select specific bins)
        #fft_features = [col for col in feature_columns if col.startswith('fft_bin_')]
        #selected_features.extend(fft_features[1:4])  # Select first 4 FFT bins

        # Add some LBP histogram features
        #lbp_features = [col for col in feature_columns if col.startswith('lbp_hist_')]
        #selected_features.extend(lbp_features[:5])  # Select first 5 LBP histogram bins

        #hu_features = [col for col in feature_columns if col.startswith('hu_')]
        #selected_features.extend(hu_features[:3])

        # Filter to only include features that exist in the dataset
        selected_features = [f for f in selected_features if f in feature_columns]

    elif method == 'variance':
        # Select features based on highest variance
        variances = feature_data.var().sort_values(ascending=False)
        selected_features = variances.head(n_features).index.tolist()

    elif method == 'correlation':
        # Remove highly correlated features (keep first occurrence)
        corr_matrix = feature_data.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [column for column in upper_triangle.columns
                             if any(upper_triangle[column] > 0.95)]
        selected_features = [f for f in feature_columns if f not in high_corr_features]
        # If we still have too many features, select top n by variance
        if len(selected_features) > n_features:
            variances = feature_data[selected_features].var().sort_values(ascending=False)
            selected_features = variances.head(n_features).index.tolist()

    elif method == 'rfecv':
        # Recursive Feature Elimination with Cross-Validation
        selected_features = perform_rfecv(df, estimator_name='random_forest', cv=3, max_features=max_features)
        
    else:
        raise ValueError("Method must be 'manual', 'variance', 'correlation', or 'rfecv'")

    # Create new dataframe with selected features plus metadata
    selected_df = df[['filename', 'label'] + selected_features].copy()

    print(f"Selected {len(selected_features)} features using '{method}' method:")
    print(selected_features)

    return selected_df


def save_processed_features(df: pd.DataFrame, method: str = 'manual', n_features: int = 20, output_path: str = './data/processed/surface_features_processed.csv', max_features: Optional[int] = None):
    """
    Process the dataset using the specified feature selection method and save to the standard processed file name.
    
    Args:
        df: pandas DataFrame containing the full dataset
        method: feature selection method ('manual', 'variance', 'correlation', or 'rfecv')
        n_features: number of features to select (for variance/correlation methods)
        output_path: path to save the processed features
    
    Returns:
        The processed DataFrame
    """
    processed_df = select_features(df, method=method, n_features=n_features, max_features=max_features)
    processed_df.to_csv(output_path, index=False)
    print(f"Processed features saved to '{output_path}' using method '{method}'")
    return processed_df


if __name__ == "__main__":
    # Example usage:
    # Load the full dataset
    try:
        full_df = pd.read_csv('./data/processed/surface_features_dataset.csv')
        print(f"Loaded dataset with {len(full_df)} samples and {len(full_df.columns)-2} features")

        # Process features using the default manual method and save to standard name
        processed_df = save_processed_features(full_df, method='manual')
        
        # If you want to try other methods, you can uncomment and run these:
        # processed_df = save_processed_features(full_df, method='variance', n_features=20)
        # processed_df = save_processed_features(full_df, method='correlation', n_features=20)
        # processed_df = save_processed_features(full_df, method='rfecv', max_features=2)
        
    except FileNotFoundError:
        print("Dataset file not found. Please run the feature extraction first.")
        
    print("\n=== Feature selection completed ===")
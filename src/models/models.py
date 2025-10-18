from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# XGBoost imports
import xgboost as xgb

# PyTorch imports for ANN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def train_random_forest_kfold(
    dataset_path: str = './data/processed/surface_features_processed.csv', 
    k_folds: int = 3, 
    model_dir: str = 'saved_model'
) -> Tuple[float, float]:
    """
    Train Random Forest with K-Fold Cross-Validation.

    Args:
        dataset_path: Path to the dataset CSV file
        k_folds: Number of cross-validation folds
        model_dir: Directory to save the trained model

    Returns:
        Tuple of mean accuracy and standard deviation
    """
    # Configuration
    model_path = os.path.join(model_dir, 'random_forest_model_kfold.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    encoder_path = os.path.join(model_dir, 'label_encoder.joblib')

    os.makedirs(model_dir, exist_ok=True)

    # Load and prepare data
    print("--- Step 1: Loading and Preparing Data ---")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{dataset_path}'.")
        return 0.0, 0.0

    df = df.drop(columns=['filename'])
    df = df.dropna()

    X = df.drop(columns=['label'])
    y = df['label']

    print(f"Dataset loaded successfully with {X.shape[0]} samples and {X.shape[1]} features.")

    # Preprocessing: encode labels and scale features
    print("\n--- Step 2: Preprocessing Full Dataset ---")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Labels encoded and features scaled.")

    # K-Fold cross-validation
    print(f"\n--- Step 3: Performing {k_folds}-Fold Stratified Cross-Validation ---")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    overall_y_true = []
    overall_y_pred = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_scaled, y_encoded)):
        print(f"--- FOLD {fold + 1}/{k_folds} ---")

        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]

        rf_classifier = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            max_features='sqrt',
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )

        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        fold_accuracies.append(accuracy)
        print(f"Validation Accuracy for Fold {fold + 1}: {accuracy:.4f}")

        overall_y_true.extend(y_val)
        overall_y_pred.extend(y_pred)

    # Display overall cross-validation results
    print("\n--- Step 4: Overall Cross-Validation Results ---")
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

    print("\nOverall Classification Report (from out-of-sample predictions):")
    print(classification_report(overall_y_true, overall_y_pred, target_names=le.classes_))

    print("Overall Confusion Matrix:")
    cm = confusion_matrix(overall_y_true, overall_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Overall Confusion Matrix (from K-Fold CV)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # Train final model on all data
    print("\n--- Step 5: Training Final Model on the Entire Dataset ---")
    final_model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2, n_jobs=-1)
    final_model.fit(X_scaled, y_encoded)
    print("Final model training complete.")

    # Analyze feature importance of the final model
    print("\n--- Step 6: Analyzing Feature Importance of the Final Model ---")
    importances = final_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15))
    plt.title('Top 15 Most Important Features (from Final Model)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # Skip saving the model to avoid file system operations
    print("\n--- Step 7: Skipping Model Saving ---")
    print("Model training completed (skipping save to prevent file system operations)")

    return mean_accuracy, std_accuracy


def train_xgboost_kfold(
    dataset_path: str = './data/processed/surface_features_processed.csv', 
    k_folds: int = 3, 
    model_dir: str = 'saved_model'
) -> Tuple[float, float]:
    """
    Train XGBoost with K-Fold Cross-Validation.

    Args:
        dataset_path: Path to the dataset CSV file
        k_folds: Number of cross-validation folds
        model_dir: Directory to save the trained model

    Returns:
        Tuple of mean accuracy and standard deviation
    """
    # Configuration
    model_path = os.path.join(model_dir, 'xgboost_model_kfold.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    encoder_path = os.path.join(model_dir, 'label_encoder.joblib')

    os.makedirs(model_dir, exist_ok=True)

    # Load and prepare data
    print("--- Step 1: Loading and Preparing Data ---")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{dataset_path}'.")
        return 0.0, 0.0

    df = df.drop(columns=['filename'])
    df = df.dropna()

    X = df.drop(columns=['label'])
    y = df['label']

    print(f"Dataset loaded successfully with {X.shape[0]} samples and {X.shape[1]} features.")

    # Preprocessing: encode labels and scale features
    print("\n--- Step 2: Preprocessing Full Dataset ---")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Labels encoded and features scaled.")

    # K-Fold cross-validation
    print(f"\n--- Step 3: Performing {k_folds}-Fold Stratified Cross-Validation with XGBoost ---")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    overall_y_true = []
    overall_y_pred = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_scaled, y_encoded)):
        print(f"--- FOLD {fold + 1}/{k_folds} ---")

        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]

        xgb_classifier = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.15,
            max_depth=4,
            subsample=1.0,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            reg_lambda=1,
            reg_alpha=0.1,
            n_jobs=-1,
            random_state=42
        )

        xgb_classifier.fit(X_train, y_train)

        y_pred = xgb_classifier.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        fold_accuracies.append(accuracy)
        print(f"Validation Accuracy for Fold {fold + 1}: {accuracy:.4f}")

        overall_y_true.extend(y_val)
        overall_y_pred.extend(y_pred)

    # Display overall cross-validation results
    print("\n--- Step 4: Overall Cross-Validation Results ---")
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

    print("\nOverall Classification Report (from out-of-sample predictions):")
    print(classification_report(overall_y_true, overall_y_pred, target_names=le.classes_))

    print("Overall Confusion Matrix:")
    cm = confusion_matrix(overall_y_true, overall_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Overall Confusion Matrix (from K-Fold CV with XGBoost)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # Train final model on all data
    print("\n--- Step 5: Training Final XGBoost Model on the Entire Dataset ---")
    final_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.15,
        max_depth=4,
        subsample=1.0,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        reg_lambda=1,
        reg_alpha=0.1,
        n_jobs=-1,
        random_state=42
    )
    final_model.fit(X_scaled, y_encoded)
    print("Final model training complete.")

    # Analyze feature importance of the final model
    print("\n--- Step 6: Analyzing Feature Importance of the Final Model ---")
    importances = final_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15))
    plt.title('Top 15 Most Important Features (from Final XGBoost Model)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # Skip saving the model to avoid file system operations
    print("\n--- Step 7: Skipping Model Saving ---")
    print("Model training completed (skipping save to prevent file system operations)")

    return mean_accuracy, std_accuracy





class SimpleANN(nn.Module):
    """Simple Artificial Neural Network for classification."""
    def __init__(self, input_size: int, num_classes: int):
        """
        Initialize the neural network.

        Args:
            input_size: Number of input features
            num_classes: Number of output classes
        """
        super(SimpleANN, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


def train_ann_kfold(
    dataset_path: str = './data/processed/surface_features_processed.csv', 
    k_folds: int = 5, 
    model_dir: str = 'saved_model',
    epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 0.005
) -> Tuple[float, float]:
    """
    Train ANN with K-Fold Cross-Validation using PyTorch.

    Args:
        dataset_path: Path to the dataset CSV file
        k_folds: Number of cross-validation folds
        model_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Size of training batches
        learning_rate: Learning rate for optimizer

    Returns:
        Tuple of mean accuracy and standard deviation
    """
    # Configuration
    model_path = os.path.join(model_dir, 'ann_model.pth')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    encoder_path = os.path.join(model_dir, 'label_encoder.joblib')

    os.makedirs(model_dir, exist_ok=True)

    # Load and prepare data
    print("--- Step 1: Loading and Preparing Data ---")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{dataset_path}'.")
        return 0.0, 0.0

    df = df.drop(columns=['filename'])
    df = df.dropna()

    X = df.drop(columns=['label'])
    y = df['label']

    print(f"Dataset loaded successfully with {X.shape[0]} samples and {X.shape[1]} features.")

    # Preprocessing
    print("\n--- Step 2: Preprocessing Full Dataset ---")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Labels encoded and features scaled.")

    # K-Fold cross-validation with PyTorch
    print(f"\n--- Step 3: Performing {k_folds}-Fold Stratified Cross-Validation with PyTorch ANN ---")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    overall_y_true = []
    overall_y_pred = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_scaled, y_encoded)):
        print(f"--- FOLD {fold + 1}/{k_folds} ---")

        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        input_size = X.shape[1]
        num_classes = len(le.classes_)
        model = SimpleANN(input_size, num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, predicted = torch.max(val_outputs.data, 1)

            y_pred_fold = predicted.numpy()
            y_val_fold = y_val_tensor.numpy()

            accuracy = accuracy_score(y_val_fold, y_pred_fold)
            fold_accuracies.append(accuracy)
            print(f"Validation Accuracy for Fold {fold + 1}: {accuracy:.4f}")

            overall_y_true.extend(y_val_fold)
            overall_y_pred.extend(y_pred_fold)

    # Display overall cross-validation results
    print("\n--- Step 4: Overall Cross-Validation Results ---")
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

    print("\nOverall Classification Report:")
    print(classification_report(overall_y_true, overall_y_pred, target_names=le.classes_))

    print("Overall Confusion Matrix:")
    cm = confusion_matrix(overall_y_true, overall_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Overall Confusion Matrix (from K-Fold CV with PyTorch ANN)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # Train final model on all data
    print("\n--- Step 5: Training Final ANN Model on the Entire Dataset ---")

    X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_encoded_tensor = torch.tensor(y_encoded, dtype=torch.long)
    full_dataset = TensorDataset(X_scaled_tensor, y_encoded_tensor)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    final_model = SimpleANN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(epochs):
        final_model.train()
        for inputs, labels in full_loader:
            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Final Model Training: Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("Final model training complete.")

    # Skip saving the model to avoid file system operations
    print("\n--- Step 6: Skipping Model Saving ---")
    print("Model training completed (skipping save to prevent file system operations)")

    return mean_accuracy, std_accuracy


if __name__ == "__main__":
    # Example usage - uncomment to run individual models
    # train_random_forest_kfold()
    # train_xgboost_kfold()
    # train_ann_kfold()
    pass
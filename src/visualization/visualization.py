from typing import List, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import learning_curve
from src.data.image_processing import extract_surface_data

def plot_height_distribution(data_matrix: np.ndarray, title: str = "Height Value Distribution") -> None:
    """
    Plot the distribution of height values from the processed data matrix.

    Args:
        data_matrix: 2D array with NaN for background, values in [0,1] for data
        title: Title for the plot
    """
    # Extract valid (non-NaN) height values
    valid_heights = data_matrix[~np.isnan(data_matrix)].flatten()

    if len(valid_heights) == 0:
        print("No valid height data to plot")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)

    # Histogram of height values
    axes[0].hist(valid_heights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Normalized Height Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Height Values')
    axes[0].grid(True, alpha=0.3)

    # Box plot of height values
    axes[1].boxplot(valid_heights, vert=True)
    axes[1].set_ylabel('Normalized Height Value')
    axes[1].set_title('Box Plot of Height Values')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_surface_matrix(data_matrix: np.ndarray, title: str = "Surface Height Map") -> None:
    """
    Visualize the 2D height matrix as an image.

    Args:
        data_matrix: 2D array with NaN for background, values in [0,1] for data
        title: Title for the plot
    """
    # Create a copy to avoid modifying original data
    display_matrix = data_matrix.copy()

    # Fill NaN values with a distinct color (e.g., -0.1)
    display_matrix[np.isnan(display_matrix)] = -0.1

    plt.figure(figsize=(10, 8))
    # Use a colormap that clearly shows the NaN background
    im = plt.imshow(display_matrix, cmap='viridis', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Normalized Height Value', rotation=270, labelpad=20)

    plt.title(title)
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

    # Add a contour overlay to better visualize height variations
    # Only plot contours for valid data region
    valid_mask = ~np.isnan(data_matrix)
    if np.any(valid_mask):
        # Create coordinate grids
        y, x = np.mgrid[0:data_matrix.shape[0], 0:data_matrix.shape[1]]
        # Plot contours only where data exists
        plt.contour(x, y, np.where(valid_mask, data_matrix, np.nan),
                   levels=15, colors='white', alpha=0.4, linewidths=0.5)

    plt.tight_layout()
    plt.show()


def compare_height_distributions(data_matrices: List[np.ndarray], labels: List[str], title: str = "Comparison of Height Distributions") -> None:
    """
    Compare height value distributions from multiple data matrices.

    Args:
        data_matrices: List of 2D arrays with height data
        labels: List of labels for each matrix
        title: Title for the plot
    """
    # Extract valid height values for each matrix
    height_samples = []
    for matrix in data_matrices:
        valid_heights = matrix[~np.isnan(matrix)].flatten()
        if len(valid_heights) > 0:
            height_samples.append(valid_heights)
        else:
            height_samples.append(np.array([]))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=14)

    # Histogram comparison
    for i, (sample, label) in enumerate(zip(height_samples, labels)):
        if len(sample) > 0:
            axes[0].hist(sample, bins=30, alpha=0.6, label=label, density=True)
    axes[0].set_xlabel('Normalized Height Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Height Distribution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot comparison
    # Filter out empty samples for boxplot
    non_empty_samples = [sample for sample in height_samples if len(sample) > 0]
    non_empty_labels = [label for i, (sample, label) in enumerate(zip(height_samples, labels)) if len(sample) > 0]

    if non_empty_samples:
        axes[1].boxplot(non_empty_samples, labels=non_empty_labels)
        axes[1].set_ylabel('Normalized Height Value')
        axes[1].set_title('Box Plot Comparison')
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()


def visualize_single_image_features(image_path: str) -> None:
    """
    Process a single image and visualize its height features.

    Args:
        image_path: Path to the input image
    """
    try:
        # Extract the data matrix
        data_matrix = extract_surface_data(image_path)

        # Get image name for titles
        image_name = os.path.basename(image_path)

        # Visualize the height matrix
        visualize_surface_matrix(data_matrix, f"Surface Height Map: {image_name}")

        # Plot height distribution
        plot_height_distribution(data_matrix, f"Height Distribution: {image_name}")

        # Print some statistics
        valid_data = data_matrix[~np.isnan(data_matrix)]
        if len(valid_data) > 0:
            print(f"Height Statistics for {image_name}:")
            print(f"  Mean: {np.mean(valid_data):.3f}")
            print(f"  Std Dev: {np.std(valid_data):.3f}")
            print(f"  Min: {np.min(valid_data):.3f}")
            print(f"  Max: {np.max(valid_data):.3f}")
            print(f"  Data Points: {len(valid_data)}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def visualize_dataset_samples(dataset_csv: str, image_dir: str, num_samples: int = 3) -> None:
    """
    Visualize height features for sample images from the dataset.

    Args:
        dataset_csv: Path to the features CSV file
        image_dir: Root directory containing the images
        num_samples: Number of samples to visualize per class
    """
    # Load the dataset
    df = pd.read_csv(dataset_csv)

    # Get unique labels
    labels = df['label'].unique()

    for label in labels:
        # Get samples for this class
        class_samples = df[df['label'] == label].sample(n=min(num_samples, len(df[df['label'] == label])), random_state=42)

        data_matrices = []
        sample_names = []

        for _, row in class_samples.iterrows():
            # Reconstruct image path
            image_path = os.path.join(image_dir, label, row['filename'])

            if os.path.exists(image_path):
                try:
                    # Extract data matrix
                    data_matrix = extract_surface_data(image_path)
                    data_matrices.append(data_matrix)
                    sample_names.append(row['filename'])

                    # Visualize individual sample
                    visualize_surface_matrix(data_matrix, f"{label}: {row['filename']}")
                    plot_height_distribution(data_matrix, f"{label}: {row['filename']}")

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue

        # Compare distributions within class
        if len(data_matrices) > 1:
            compare_height_distributions(
                data_matrices,
                sample_names,
                f"Height Distribution Comparison: {label}"
            )


def plot_learning_curves(models: dict, X: np.ndarray, y: np.ndarray, cv: int = 5, train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10)):
    """
    Plot learning curves for multiple models to check for overfitting.

    Args:
        models: Dictionary of model names to model instances
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        train_sizes: Array of training set sizes to evaluate
    """
    plt.figure(figsize=(12, 8))
    
    for name, model in models.items():
        train_sizes_abs, train_scores, validation_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1
        )
        
        # Calculate mean and std for training and validation scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(validation_scores, axis=1)
        val_std = np.std(validation_scores, axis=1)
        
        color = 'blue' if 'Random' in name else 'green'
        # Plot training scores
        plt.plot(train_sizes_abs, train_mean, 'o-', label=f'{name} Training Score', color=color, alpha=0.7)
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color=color)
        
        # Plot validation scores
        plt.plot(train_sizes_abs, val_mean, 's-', label=f'{name} Validation Score', color='red' if 'Random' in name else 'orange')
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red' if 'Random' in name else 'orange')
    
    plt.title('Learning Curves')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage functions:
    # Uncomment one of the following lines to use:

    # 1. Visualize a single image:
    # visualize_single_image_features('./data/augmented_dataset/Channeling/SharedScreenshot - Copy (2)_bright_contrast.jpg')

    # 2. Visualize samples from your dataset:
    visualize_dataset_samples('./data/processed/surface_features_dataset.csv', './data/augmented_dataset')

    # 3. Quick test with a sample from your existing dataset:
    pass
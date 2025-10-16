from typing import Optional
import os
from tqdm import tqdm
import pandas as pd
from .image_processing import extract_surface_data
from .feature_engineering import extract_features_from_matrix


def create_dataset_from_directory(root_dir: str, output_csv: str) -> pd.DataFrame:
    """
    Walks through a directory, processes each image, extracts features,
    and saves the results to a CSV file.

    Args:
        root_dir: Root directory containing class subdirectories
        output_csv: Output CSV file path

    Returns:
        DataFrame containing extracted features
    """
    all_features_list = []

    # Get a list of all image paths to use with tqdm
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(subdir, file))

    print(f"Found {len(image_paths)} images. Starting feature extraction...")

    # Use tqdm for a nice progress bar
    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            # Extract the class label from the directory name
            label = os.path.basename(os.path.dirname(image_path))

            # Step 1: Extract the 2D data matrix from the image
            data_matrix = extract_surface_data(image_path)

            # Step 2: Compute the feature vector from the matrix
            features = extract_features_from_matrix(data_matrix)

            # Add metadata (filename and label) to the feature dictionary
            features['filename'] = os.path.basename(image_path)
            features['label'] = label

            all_features_list.append(features)

        except Exception as e:
            print(f"\nWARNING: Could not process {image_path}. Error: {e}")
            continue

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(all_features_list)

    # Reorder columns to have filename and label first
    cols = ['filename', 'label'] + [col for col in df.columns if col not in ['filename', 'label']]
    df = df[cols]

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully created dataset with {len(df)} samples.")
    print(f"Dataset saved to: {output_csv}")

    return df


if __name__ == "__main__":
    DATASET_DIRECTORY = '../../data/augmented_dataset'  # The root folder containing class subdirectories
    OUTPUT_CSV_FILE = '../../data/processed/surface_features_dataset.csv'

    if not os.path.isdir(DATASET_DIRECTORY):
        print(f"Error: Dataset directory '{DATASET_DIRECTORY}' not found.")
        print("Please ensure your images are in subdirectories inside a 'dataset' folder.")
    else:
        # Create the dataset
        feature_df = create_dataset_from_directory(DATASET_DIRECTORY, OUTPUT_CSV_FILE)

        # Display the first few rows of the created dataset
        if not feature_df.empty:
            print("\n--- Dataset Preview ---")
            print(feature_df.head())
            print("\n--- Dataset Info ---")
            print(feature_df.info())
            print("\n--- Label Distribution ---")
            print(feature_df['label'].value_counts())
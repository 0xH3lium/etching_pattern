# Surface Feature Extraction and Classification Pipeline

This repository contains a complete pipeline for extracting features from 3D surface plot images and classifying different surface patterns.

## Directory Structure

```
Final/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── image_processing.py          # Extracts 2D data matrix from surface plot images
│   │   ├── dataset_creation.py          # Creates dataset from image directory and extracts features
│   │   └── feature_engineering.py       # Contains functions for extracting various features
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_selection.py         # Selects subset of features from the dataset
│   │   ├── feature_bruteforce.py        # Tests all possible feature combinations to find best performing subset
│   │   └── feature_bruteforce_simple.py # Simpler version of feature optimization
│   ├── models/
│   │   ├── __init__.py
│   │   └── models.py                    # Contains implementations for Random Forest, XGBoost, and ANN models
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualization.py             # Contains functions for visualizing surface data and feature distributions
│   └── main.py                         # Orchestrates the complete pipeline
├── docs/
│   └── feature_engineering.md          # Comprehensive documentation of the feature engineering process
├── data/
│   ├── raw/                           # For original images
│   ├── processed/                     # For processed datasets
│   │   ├── surface_features_dataset.csv
│   │   └── surface_features_processed.csv
│   └── augmented_dataset/             # Directory containing input images organized by class
├── scripts/
│   └── (any utility scripts)
├── tests/
│   └── (for future tests)
└── README.md                         # This file
```

## Usage

### Running the Complete Pipeline
To run the complete pipeline from feature extraction through model training:
```bash
python main.py
```

### Individual Components

#### Data Processing
- `src/data/dataset_creation.py` - Creates dataset from image directory and extracts features
- `src/data/image_processing.py` - Extracts 2D data matrix from surface plot images
- `src/data/feature_engineering.py` - Contains functions for extracting various features

#### Feature Selection & Optimization
- `src/features/feature_selection.py` - Selects subset of features from the dataset
- `src/features/feature_bruteforce.py` - Tests all possible feature combinations to find best performing subset
- `src/features/feature_bruteforce_simple.py` - Simpler version of feature optimization

#### Models
- `src/models/models.py` - Contains implementations for Random Forest, XGBoost, and ANN models with K-Fold cross-validation

#### Visualization
- `src/visualization/visualization.py` - Contains functions for visualizing surface data and feature distributions

## Installation

To install the required dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

- Input images should be placed in `data/augmented_dataset/` in class-specific subdirectories
- Processed datasets are stored in `data/processed/`
- Each image represents a 3D surface plot in PNG, JPG, or JPEG format

## Feature Categories

The pipeline extracts features from seven distinct categories:
1. Statistical Features
2. Texture Features (GLCM)
3. Frequency Domain Features (FFT)
4. Blob Detection Features
5. Local Binary Pattern (LBP) Features
6. Gradient Features
7. Hu Moments

For detailed information about the features, see the documentation in `docs/feature_engineering.md`.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Specify your license here]
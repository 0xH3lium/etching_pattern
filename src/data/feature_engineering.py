from typing import Dict, Any
import numpy as np
from skimage.feature import graycomatrix, graycoprops, blob_log, local_binary_pattern
from scipy.stats import skew, kurtosis
from skimage.filters import sobel_h, sobel_v
import cv2


def prepare_matrix(data_matrix: np.ndarray) -> np.ndarray:
    """
    Prepares the data matrix for feature extraction.
    - Fills NaN with 0.
    - Scales to 0-255 uint8 format for image processing libraries.

    Args:
        data_matrix: Input data matrix

    Returns:
        Prepared matrix in uint8 format
    """
    # Create a copy to avoid modifying the original
    matrix_uint8 = np.nan_to_num(data_matrix, nan=0.0)
    # Scale from 0-1 float to 0-255 uint8
    matrix_uint8 = (matrix_uint8 * 255).astype(np.uint8)
    return matrix_uint8


def calculate_statistical_features(data_matrix: np.ndarray) -> Dict[str, float]:
    """Calculates global statistical features, ignoring NaN values.

    Args:
        data_matrix: Input data matrix

    Returns:
        Dictionary containing statistical features
    """
    # Flatten the array and remove NaNs for calculation
    valid_data = data_matrix[~np.isnan(data_matrix)]
    if valid_data.size == 0:
        return {'mean': 0, 'std_dev': 0, 'skewness': 0, 'kurt': 0}

    features = {
        'mean': np.mean(valid_data),
        'std_dev': np.std(valid_data),
        'skewness': skew(valid_data),
        'kurt': kurtosis(valid_data)
    }
    return features


def calculate_glcm_features(matrix_uint8: np.ndarray) -> Dict[str, float]:
    """Calculates texture features using Gray-Level Co-occurrence Matrix (GLCM).

    Args:
        matrix_uint8: Input matrix in uint8 format

    Returns:
        Dictionary containing GLCM features
    """
    # Define distances and angles for GLCM. Averaging over these makes features more robust.
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0, 45, 90, 135 degrees

    glcm = graycomatrix(matrix_uint8, distances=distances, angles=angles, symmetric=True, normed=True)

    # Calculate properties for each GLCM and average them
    properties = ['contrast', 'homogeneity', 'energy', 'correlation']
    glcm_features = {prop: np.mean(graycoprops(glcm, prop)) for prop in properties}

    return glcm_features


def calculate_fft_features(data_matrix: np.ndarray, num_bins: int = 6) -> Dict[str, float]:
    """Calculates features from the 2D Fourier Transform power spectrum.

    Args:
        data_matrix: Input data matrix
        num_bins: Number of bins for radial profiling

    Returns:
        Dictionary containing FFT features
    """
    matrix_no_nan = np.nan_to_num(data_matrix, nan=0.0)

    # Compute the 2D FFT and shift the zero-frequency component to the center
    f_transform = np.fft.fft2(matrix_no_nan)
    f_transform_shifted = np.fft.fftshift(f_transform)
    power_spectrum = np.abs(f_transform_shifted)**2

    # Calculate radial profile
    center_y, center_x = np.array(power_spectrum.shape) / 2
    y, x = np.indices(power_spectrum.shape)
    radii = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Bin the radii and calculate the mean power in each bin
    max_radius = np.max(radii)
    bin_edges = np.linspace(0, max_radius, num_bins + 1)

    # Using np.histogram is an efficient way to bin the data
    radial_sum, _ = np.histogram(radii, bins=bin_edges, weights=power_spectrum)
    count, _ = np.histogram(radii, bins=bin_edges)

    # Avoid division by zero for empty bins
    radial_mean = np.divide(radial_sum, count, out=np.zeros_like(radial_sum, dtype=float), where=(count!=0))

    fft_features = {f'fft_bin_{i}': val for i, val in enumerate(radial_mean)}
    return fft_features


def calculate_blob_features(matrix_uint8: np.ndarray) -> Dict[str, float]:
    """Detects blobs (for 'pillar' class) using Laplacian of Gaussian.

    Args:
        matrix_uint8: Input matrix in uint8 format

    Returns:
        Dictionary containing blob features
    """
    # blob_log parameters can be tuned.
    # min/max_sigma define the scale of blobs to detect.
    # threshold defines the sensitivity.
    blobs = blob_log(matrix_uint8, min_sigma=3, max_sigma=30, num_sigma=10, threshold=.1)

    num_blobs = len(blobs)
    if num_blobs > 0:
        # The 3rd column of the result is sigma, proportional to blob radius
        mean_blob_size = np.mean(blobs[:, 2])
    else:
        mean_blob_size = 0

    return {'num_blobs': num_blobs, 'mean_blob_size': mean_blob_size}


def calculate_hu_moments(matrix_uint8: np.ndarray) -> Dict[str, float]:
    """Calculates seven Hu Moments which are invariant to scale, rotation, and translation.

    Args:
        matrix_uint8: Input matrix in uint8 format

    Returns:
        Dictionary containing Hu moment features
    """
    moments = cv2.moments(matrix_uint8)
    hu_moments = cv2.HuMoments(moments)

    # Log transform for stability, as the values can be very small
    # Add a small epsilon to avoid log(0)
    log_hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7)

    return {f'hu_moment_{i}': val[0] for i, val in enumerate(log_hu_moments)}


def calculate_lbp_features(matrix_uint8: np.ndarray, P: int = 8, R: int = 1, bins: int = 10) -> Dict[str, float]:
    """
    Calculates features from the Local Binary Pattern histogram.
    
    Args:
        matrix_uint8: Input matrix in uint8 format
        P: Number of circularly symmetric neighbor set points
        R: Radius of circle (spatial resolution of the operator)
        bins: Number of bins for histogram (not used here but kept for consistency)

    Returns:
        Dictionary containing LBP features
    """
    # Calculate LBP
    lbp = local_binary_pattern(matrix_uint8, P, R, method='uniform')

    # Calculate histogram of LBP codes
    # The number of bins for the histogram is P + 2 for the 'uniform' method
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    # You can use the histogram values directly as features
    lbp_features = {f'lbp_hist_{i}': val for i, val in enumerate(hist)}

    # Or calculate statistics on the histogram
    lbp_features['lbp_entropy'] = -np.sum(hist * np.log2(hist + 1e-9))  # Add a small epsilon to avoid log(0)

    return lbp_features


def calculate_gradient_features(matrix_uint8: np.ndarray, bins: int = 8) -> Dict[str, float]:
    """Calculates features based on the distribution of gradient orientations.

    Args:
        matrix_uint8: Input matrix in uint8 format
        bins: Number of orientation bins

    Returns:
        Dictionary containing gradient features
    """
    # Calculate horizontal and vertical gradients
    grad_v = sobel_v(matrix_uint8)
    grad_h = sobel_h(matrix_uint8)

    # Calculate gradient magnitude and orientation
    magnitude = np.sqrt(grad_v**2 + grad_h**2)
    orientation = np.arctan2(grad_v, grad_h)  # in radians from -pi to pi

    # We only care about strong gradients, so let's mask out weak ones
    strong_gradients_mask = magnitude > np.mean(magnitude)
    strong_orientations = orientation[strong_gradients_mask]

    if len(strong_orientations) == 0:
        return {'gradient_orientation_entropy': 0, 'mean_gradient_magnitude': np.mean(magnitude)}

    # Create a histogram of orientations
    hist, _ = np.histogram(strong_orientations, bins=bins, range=(-np.pi, np.pi))

    # Normalize histogram to be a probability distribution
    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist

    # Calculate the entropy of the orientation histogram.
    # Low entropy -> strong directionality. High entropy -> random/isotropic.
    entropy = -np.sum(hist * np.log2(hist + 1e-9))

    return {
        'gradient_orientation_entropy': entropy,
        'mean_gradient_magnitude': np.mean(magnitude)  # Overall "busyness" or "roughness"
    }


def extract_features_from_matrix(data_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Master function to compute and combine all features for a given data matrix.

    Args:
        data_matrix: Input data matrix

    Returns:
        Dictionary containing all extracted features
    """
    # Prepare the matrix for libraries that need uint8 format
    matrix_uint8 = prepare_matrix(data_matrix)

    # --- Original Features ---
    stat_features = calculate_statistical_features(data_matrix)
    glcm_features = calculate_glcm_features(matrix_uint8)
    fft_features = calculate_fft_features(data_matrix)
    blob_features = calculate_blob_features(matrix_uint8)

    # --- NEWLY ADDED FEATURES ---
    lbp_features = calculate_lbp_features(matrix_uint8)
    gradient_features = calculate_gradient_features(matrix_uint8)
    hu_moment_features = calculate_hu_moments(matrix_uint8)

    # Combine all features into a single dictionary
    all_features = {
        **stat_features,
        **glcm_features,
        **fft_features,
        **blob_features,
        **lbp_features,
        **gradient_features,
        **hu_moment_features,
    }

    return all_features
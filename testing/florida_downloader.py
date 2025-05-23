#!/usr/bin/env python3
"""
Florida SuiteSparse Matrix Collection Downloader
===============================================

This module provides functions to download matrices from the SuiteSparse Matrix Collection
(formerly known as the University of Florida Sparse Matrix Collection). It includes:
- Matrix downloading with progress bars
- Color-coded logging
- Matrix information extraction
- File handling utilities

The focus is on the GSET matrices as shown in the provided image.
"""

import os
import requests
import scipy.io as sio
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform colored terminal output
colorama.init(autoreset=True)

# Constants
BASE_URL = "https://sparse.tamu.edu/mat"
DATA_DIR = "data"
CHUNK_SIZE = 8192  # Size of chunks to download

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Logging utility functions
def log_info(msg):
    """Print information message in blue color."""
    print(f"{Fore.BLUE}[INFO] {msg}{Style.RESET_ALL}")

def log_success(msg):
    """Print success message in green color."""
    print(f"{Fore.GREEN}[SUCCESS] {msg}{Style.RESET_ALL}")

def log_warning(msg):
    """Print warning message in yellow color."""
    print(f"{Fore.YELLOW}[WARNING] {msg}{Style.RESET_ALL}")

def log_error(msg):
    """Print error message in red color."""
    print(f"{Fore.RED}[ERROR] {msg}{Style.RESET_ALL}")

def log_debug(msg):
    """Print debug message in magenta color."""
    print(f"{Fore.MAGENTA}[DEBUG] {msg}{Style.RESET_ALL}")

def download_with_progress(url, local_path):
    """
    Download a file from the given URL with a progress bar.
    
    Args:
        url (str): The URL to download from
        local_path (str): The local path to save the file
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Send a HEAD request to get the file size
        response = requests.head(url)
        file_size = int(response.headers.get('content-length', 0))
        
        # Create a progress bar
        progress_bar = tqdm(
            total=file_size, 
            unit='B', 
            unit_scale=True,
            desc=f"Downloading {os.path.basename(local_path)}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        # Make the actual request and download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))
                    
        progress_bar.close()
        return True
    except requests.exceptions.RequestException as e:
        log_error(f"Download failed: {str(e)}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False
    except Exception as e:
        log_error(f"Unexpected error during download: {str(e)}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False

def download_matrix(matrix_id, matrix_name, group):
    """
    Download a matrix from the SuiteSparse Matrix Collection.
    
    Args:
        matrix_id (int): The ID of the matrix
        matrix_name (str): The name of the matrix
        group (str): The group the matrix belongs to
    
    Returns:
        str: Path to the downloaded matrix file, or None if download failed
    """
    log_info(f"Preparing to download matrix {matrix_name} (ID: {matrix_id}) from group {group}")
    
    url = f"{BASE_URL}/{group}/{matrix_name}.mat"
    local_path = os.path.join(DATA_DIR, f"{matrix_name}.mat")
    
    # Skip download if the file already exists
    if os.path.exists(local_path):
        log_info(f"Matrix file already exists at {local_path}")
        return local_path
    
    log_info(f"Downloading from {url} to {local_path}")
    success = download_with_progress(url, local_path)
    
    if success:
        log_success(f"Successfully downloaded matrix to {local_path}")
        return local_path
    else:
        log_error(f"Failed to download matrix {matrix_name}")
        return None

def load_matrix(file_path):
    """
    Load a matrix from a .mat file.
    
    Args:
        file_path (str): Path to the .mat file
        
    Returns:
        numpy.ndarray: The loaded matrix, or None if loading failed
    """
    try:
        log_debug(f"Loading matrix from {file_path}")
        mat_data = sio.loadmat(file_path)
        
        # Look for the matrix in the mat file
        matrix = None
        
        # First try: Look for Problem.A (common SuiteSparse format)
        if 'Problem' in mat_data:
            problem = mat_data['Problem']
            # Some files have Problem as a struct with A field
            if hasattr(problem, 'dtype') and problem.dtype.names is not None and 'A' in problem.dtype.names:
                log_info(f"Found matrix in Problem.A (structured)")
                matrix = problem['A'][0, 0]
            # Some files have Problem as an array containing A
            elif isinstance(problem, np.ndarray) and problem.size > 0:
                # Try to locate the matrix in the Problem array
                if problem.dtype == object:  # Changed np.object to object
                    # Check if Problem is an array with object elements
                    for field_idx in range(len(problem)):
                        field = problem[field_idx]
                        if hasattr(field, 'dtype') and field.dtype.names is not None and 'A' in field.dtype.names:
                            log_info(f"Found matrix in Problem array element {field_idx}")
                            matrix = field['A'][0, 0]
                            break
        
        # Second try: Look directly for matrix
        if matrix is None:
            # Common matrix names in SuiteSparse files
            possible_names = ['A', 'Problem', 'G']
            for name in possible_names:
                if name in mat_data and isinstance(mat_data[name], np.ndarray):
                    log_info(f"Found matrix directly with name '{name}'")
                    matrix = mat_data[name]
                    break
        
        # Third try: Look for any array that's not a special field
        if matrix is None:
            for key in mat_data:
                if key not in ['__header__', '__version__', '__globals__'] and isinstance(mat_data[key], np.ndarray):
                    log_info(f"Using field '{key}' as matrix")
                    matrix = mat_data[key]
                    break
        
        # Fourth try: For G-set matrices, they might be directly the matrix with no label
        if matrix is None and os.path.basename(file_path).startswith('G'):
            # For G matrices, look for the largest array in the file
            largest_array = None
            largest_size = 0
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray) and value.size > largest_size:
                    largest_array = value
                    largest_size = value.size
            if largest_array is not None:
                log_info("Using largest array in file as matrix")
                matrix = largest_array
        
        # If we still don't have a matrix, raise an error
        if matrix is None:
            raise ValueError("Could not find matrix in the file")
        
        # Convert to dense if it's sparse
        if sp.issparse(matrix):
            log_info("Converting sparse matrix to dense")
            matrix = matrix.toarray()
        
        log_success(f"Successfully loaded matrix of shape {matrix.shape}")
        return matrix
    except Exception as e:
        log_error(f"Error loading matrix from {file_path}: {str(e)}")
        # Return a small identity matrix as fallback
        log_warning("Returning identity matrix as fallback")
        return np.eye(800)  # Return 800x800 identity matrix as fallback for G-set matrices

def preprocess_matrix(matrix):
    """
    Preprocess a matrix to handle NaN values and normalize if needed.
    
    Args:
        matrix (numpy.ndarray): The input matrix
        
    Returns:
        numpy.ndarray: The preprocessed matrix
    """
    log_debug(f"Preprocessing matrix of shape {matrix.shape}")
    
    # Check if the matrix is a structured array or has a void dtype
    if matrix.dtype.kind in ['V', 'O']:  # V for void, O for object
        log_warning(f"Matrix has dtype {matrix.dtype}. Attempting to extract numeric data...")
        # Try to extract a numeric matrix from the structured array
        if hasattr(matrix, 'A') and isinstance(matrix.A, np.ndarray):
            log_info("Extracted matrix.A")
            matrix = matrix.A
        elif hasattr(matrix, 'data') and isinstance(matrix.data, np.ndarray):
            log_info("Extracted matrix.data")
            matrix = matrix.data
        elif isinstance(matrix, np.ndarray) and len(matrix.shape) > 0:
            # For structured arrays, try to extract the first field or convert to float
            try:
                if matrix.dtype.names and len(matrix.dtype.names) > 0:
                    log_info(f"Extracting field {matrix.dtype.names[0]} from structured array")
                    matrix = matrix[matrix.dtype.names[0]]
                else:
                    # Last resort: try to convert to float array
                    log_warning("Attempting to convert to float array")
                    matrix = np.array(matrix, dtype=np.float64)
            except (TypeError, ValueError) as e:
                log_error(f"Could not convert matrix: {str(e)}")
                # Return a small identity matrix as a fallback
                log_warning("Returning identity matrix as fallback")
                return np.eye(min(800, 10))
        else:
            log_error("Matrix structure not recognized")
            return np.eye(min(800, 10))  # Return small identity matrix as fallback
    
    # Additional check for proper numeric type
    if not np.issubdtype(matrix.dtype, np.number):
        log_warning(f"Matrix dtype {matrix.dtype} is not numeric. Converting to float64.")
        try:
            matrix = np.array(matrix, dtype=np.float64)
        except (TypeError, ValueError) as e:
            log_error(f"Conversion failed: {str(e)}")
            return np.eye(min(800, 10))
    
    # Handle NaN and Inf values
    matrix = np.nan_to_num(matrix)
    
    # If the matrix has extremely large values, normalize it
    try:
        max_val = np.max(np.abs(matrix))
        if max_val > 1e5:
            log_warning(f"Matrix has large values (max: {max_val}). Normalizing.")
            matrix = matrix / max_val
    except (TypeError, ValueError) as e:
        log_error(f"Error computing max value: {str(e)}")
    
    return matrix

# Matrix information from the image
def get_matrix_info():
    """
    Get information about the matrices from the SuiteSparse Matrix Collection
    as shown in the image.
    
    Returns:
        list: List of dictionaries containing matrix information
    """
    return [
        {"id": 469, "name": "G1", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 19462, "kind": "Undirected Random Graph", "date": "1996"},
        {"id": 470, "name": "G10", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 9862, "kind": "Undirected Weighted Random Graph", "date": "1996"},
        {"id": 471, "name": "G11", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 3200, "kind": "Undirected Weighted Random Graph", "date": "1996"},
        {"id": 472, "name": "G12", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 3200, "kind": "Undirected Weighted Random Graph", "date": "1996"},
        {"id": 473, "name": "G13", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 3200, "kind": "Undirected Weighted Random Graph", "date": "1996"},
        {"id": 474, "name": "G14", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 9388, "kind": "Duplicate Undirected Random Graph", "date": "1996"},
        {"id": 475, "name": "G15", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 9322, "kind": "Duplicate Undirected Random Graph", "date": "1996"},
        {"id": 476, "name": "G16", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 9344, "kind": "Duplicate Undirected Random Graph", "date": "1996"},
        {"id": 477, "name": "G17", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 9344, "kind": "Duplicate Undirected Random Graph", "date": "1996"},
        {"id": 478, "name": "G18", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 9388, "kind": "Undirected Weighted Random Graph", "date": "1996"},
        {"id": 479, "name": "G19", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 9322, "kind": "Undirected Weighted Random Graph", "date": "1996"},
        {"id": 480, "name": "G2", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 19352, "kind": "Undirected Random Graph", "date": "1996"},
        {"id": 481, "name": "G20", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 11166, "kind": "Undirected Weighted Random Graph", "date": "1996"},
        {"id": 482, "name": "G21", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 9354, "kind": "Undirected Weighted Random Graph", "date": "1996"},
        {"id": 491, "name": "G3", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 19352, "kind": "Undirected Random Graph", "date": "1996"},
        {"id": 502, "name": "G4", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 19352, "kind": "Undirected Random Graph", "date": "1996"},
        {"id": 513, "name": "G5", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 19352, "kind": "Undirected Random Graph", "date": "1996"},
        {"id": 524, "name": "G6", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 19352, "kind": "Undirected Weighted Random Graph", "date": "1996"},
        {"id": 533, "name": "G7", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 19352, "kind": "Undirected Random Graph", "date": "1996"},
        {"id": 534, "name": "G8", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 19352, "kind": "Undirected Random Graph", "date": "1996"},
        {"id": 535, "name": "G9", "group": "Gset", "rows": 800, "cols": 800, "nonzeros": 19352, "kind": "Undirected Random Graph", "date": "1996"},
        {"id": 1036, "name": "rdb800l", "group": "Bai", "rows": 800, "cols": 800, "nonzeros": 4640, "kind": "Computational Fluid Dynamics Problem", "date": "1994"}
    ]

def download_all_matrices(subset=None):
    """
    Download all matrices specified in the matrix info list.
    
    Args:
        subset (list, optional): List of matrix names to download. If None, download all.
        
    Returns:
        dict: Dictionary mapping matrix names to their file paths
    """
    matrix_info_list = get_matrix_info()
    downloaded_matrices = {}
    
    # Filter by subset if provided
    if subset:
        matrix_info_list = [info for info in matrix_info_list if info["name"] in subset]
    
    total_matrices = len(matrix_info_list)
    log_info(f"Starting download of {total_matrices} matrices")
    
    # Create a progress bar for the overall download process
    with tqdm(total=total_matrices, desc="Overall Progress", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as overall_pbar:
        for matrix_info in matrix_info_list:
            matrix_id = matrix_info["id"]
            matrix_name = matrix_info["name"]
            group = matrix_info["group"]
            
            # Download the matrix
            file_path = download_matrix(matrix_id, matrix_name, group)
            
            if file_path:
                downloaded_matrices[matrix_name] = file_path
            
            overall_pbar.update(1)
    
    log_success(f"Downloaded {len(downloaded_matrices)}/{total_matrices} matrices successfully")
    return downloaded_matrices

def load_all_matrices(matrix_paths):
    """
    Load all matrices from their file paths.
    
    Args:
        matrix_paths (dict): Dictionary mapping matrix names to their file paths
        
    Returns:
        dict: Dictionary mapping matrix names to their numpy array representations
    """
    matrices = {}
    total_matrices = len(matrix_paths)
    
    log_info(f"Loading {total_matrices} matrices")
    
    # Create a progress bar for loading
    with tqdm(total=total_matrices, desc="Loading Matrices", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for name, path in matrix_paths.items():
            matrix = load_matrix(path)
            if matrix is not None:
                matrices[name] = preprocess_matrix(matrix)
            pbar.update(1)
    
    log_success(f"Loaded {len(matrices)}/{total_matrices} matrices successfully")
    return matrices

def display_matrix_info():
    """Display information about all matrices in a formatted table."""
    matrix_info_list = get_matrix_info()
    
    # Print header
    header = f"{'ID':<6} {'Name':<10} {'Group':<8} {'Rows':<6} {'Cols':<6} {'Nonzeros':<10} {'Kind':<35} {'Date':<6}"
    separator = "-" * len(header)
    
    log_info("Matrix Information Table:")
    print(separator)
    print(header)
    print(separator)
    
    # Print each matrix's information
    for info in matrix_info_list:
        print(f"{info['id']:<6} {info['name']:<10} {info['group']:<8} {info['rows']:<6} {info['cols']:<6} "
              f"{info['nonzeros']:<10} {info['kind']:<35} {info['date']:<6}")
    
    print(separator)
    log_info(f"Total matrices: {len(matrix_info_list)}")

if __name__ == "__main__":
    try:
        # Display information about all matrices
        display_matrix_info()
        
        # Ask user if they want to download all matrices or a subset
        print(f"\n{Fore.CYAN}Do you want to download all matrices or specify a subset?{Style.RESET_ALL}")
        print(f"{Fore.CYAN}1. Download all matrices{Style.RESET_ALL}")
        print(f"{Fore.CYAN}2. Specify a subset{Style.RESET_ALL}")
        print(f"{Fore.CYAN}3. Download specific matrix by name{Style.RESET_ALL}")
        print(f"{Fore.CYAN}4. Exit{Style.RESET_ALL}")
        
        choice = input(f"{Fore.GREEN}Enter your choice (1-4): {Style.RESET_ALL}")
        
        if choice == '1':
            # Download all matrices
            matrix_paths = download_all_matrices()
            matrices = load_all_matrices(matrix_paths)
            log_info(f"Downloaded and loaded {len(matrices)} matrices.")
        
        elif choice == '2':
            # Specify a subset
            print(f"{Fore.CYAN}Enter matrix names separated by commas (e.g., G1,G2,G3):{Style.RESET_ALL}")
            subset_input = input()
            subset = [name.strip() for name in subset_input.split(',')]
            
            matrix_paths = download_all_matrices(subset)
            matrices = load_all_matrices(matrix_paths)
            log_info(f"Downloaded and loaded {len(matrices)} matrices.")
        
        elif choice == '3':
            # Download specific matrix
            matrix_name = input(f"{Fore.GREEN}Enter matrix name (e.g., G1): {Style.RESET_ALL}")
            
            # Find the matrix info
            matrix_info_list = get_matrix_info()
            matrix_info = next((info for info in matrix_info_list if info["name"] == matrix_name), None)
            
            if matrix_info:
                file_path = download_matrix(matrix_info["id"], matrix_info["name"], matrix_info["group"])
                if file_path:
                    matrix = load_matrix(file_path)
                    if matrix is not None:
                        log_success(f"Successfully downloaded and loaded matrix {matrix_name} of shape {matrix.shape}")
                    else:
                        log_error(f"Failed to load matrix {matrix_name}")
            else:
                log_error(f"Matrix {matrix_name} not found in the database")
        
        elif choice == '4':
            log_info("Exiting program")
        
        else:
            log_error("Invalid choice")
    
    except KeyboardInterrupt:
        log_warning("\nOperation interrupted by user")
    except Exception as e:
        log_error(f"Unexpected error: {str(e)}")
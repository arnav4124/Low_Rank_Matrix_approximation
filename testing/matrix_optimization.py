#!/usr/bin/env python3
"""
Higher Rank Matrix Approximation Explorer
=========================================

This script explores how higher rank approximations can potentially outperform SVD.
It implements:
1. Higher rank exploration for ensemble methods
2. Pre/post-processing techniques to improve approximation quality
3. Visualization of rank vs. error relationships
4. Systematic testing of hybrid approaches
"""

import os
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import json

# Import our matrix downloading utilities and analysis functions
from florida_downloader import (
    log_info, log_success, log_warning, log_error, log_debug,
    download_all_matrices, load_all_matrices
)

# Import our matrix analysis and exploration functions
from matrix_analysis import explore_higher_rank_approximations, plot_rank_vs_error, plot_rank_vs_improvement

# =============================================================================
# Pre/Post-Processing Functions
# =============================================================================

def preprocess_matrix(matrix, method='normalize'):
    """
    Apply preprocessing to a matrix to potentially improve approximation quality.
    
    Args:
        matrix (numpy.ndarray): Input matrix
        method (str): Preprocessing method ('normalize', 'center', 'scale', etc.)
        
    Returns:
        tuple: (preprocessed_matrix, preprocess_info) - The preprocessed matrix and info for reversing
    """
    if method == 'normalize':
        # Normalize by Frobenius norm
        frob_norm = la.norm(matrix, 'fro')
        return matrix / frob_norm, {'frob_norm': frob_norm}
    
    elif method == 'center':
        # Center columns
        col_means = matrix.mean(axis=0)
        centered = matrix - col_means
        return centered, {'col_means': col_means}
    
    elif method == 'scale':
        # Scale columns to unit variance
        col_stds = matrix.std(axis=0)
        # Handle zero std
        col_stds[col_stds == 0] = 1.0
        scaled = matrix / col_stds
        return scaled, {'col_stds': col_stds}
    
    elif method == 'center_scale':
        # Center and scale columns (standardize)
        col_means = matrix.mean(axis=0)
        centered = matrix - col_means
        col_stds = centered.std(axis=0)
        # Handle zero std
        col_stds[col_stds == 0] = 1.0
        scaled = centered / col_stds
        return scaled, {'col_means': col_means, 'col_stds': col_stds}
    
    else:
        # No preprocessing
        return matrix, {}

def postprocess_approximation(approx_matrix, preprocess_info, method='normalize'):
    """
    Reverse the preprocessing to get the final approximation.
    
    Args:
        approx_matrix (numpy.ndarray): The approximated matrix
        preprocess_info (dict): Information for reversing preprocessing
        method (str): The preprocessing method that was used
        
    Returns:
        numpy.ndarray: The post-processed approximation
    """
    if method == 'normalize':
        # Denormalize
        return approx_matrix * preprocess_info['frob_norm']
    
    elif method == 'center':
        # Add back column means
        return approx_matrix + preprocess_info['col_means']
    
    elif method == 'scale':
        # Rescale columns
        return approx_matrix * preprocess_info['col_stds']
    
    elif method == 'center_scale':
        # Rescale and then add back means
        rescaled = approx_matrix * preprocess_info['col_stds']
        return rescaled + preprocess_info['col_means']
    
    else:
        # No postprocessing
        return approx_matrix

# =============================================================================
# Enhanced Matrix Approximation Methods
# =============================================================================

def svd_approximation(matrix, rank):
    """
    Standard SVD approximation.
    
    Args:
        matrix (numpy.ndarray): Input matrix
        rank (int): Target rank
        
    Returns:
        numpy.ndarray: Rank-k approximation using SVD
    """
    U, s, Vt = la.svd(matrix, full_matrices=False)
    return U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]

def hybrid_svd_ensemble(matrix, rank, svd_ratio=0.8, mode='combined'):
    """
    Hybrid approach combining SVD with ensemble column selection.
    
    Args:
        matrix (numpy.ndarray): Input matrix
        rank (int): Target rank
        svd_ratio (float): Ratio of rank to use for SVD (0-1)
        mode (str): How to combine SVD and ensemble ('combined', 'sequential')
        
    Returns:
        numpy.ndarray: Hybrid approximation
    """
    log_debug(f"Applying hybrid SVD-ensemble (rank={rank}, svd_ratio={svd_ratio}, mode={mode})")
    
    svd_rank = int(rank * svd_ratio)
    ensemble_rank = rank - svd_rank
    
    if svd_rank <= 0:
        # If SVD rank is 0, use only ensemble
        return weighted_ensemble_column_selection(matrix, rank)
    
    if ensemble_rank <= 0:
        # If ensemble rank is 0, use only SVD
        return svd_approximation(matrix, rank)
    
    # Compute SVD approximation
    U, s, Vt = la.svd(matrix, full_matrices=False)
    svd_approx = U[:, :svd_rank] @ np.diag(s[:svd_rank]) @ Vt[:svd_rank, :]
    
    if mode == 'sequential':
        # Apply ensemble on the residual
        residual = matrix - svd_approx
        ensemble_approx = weighted_ensemble_column_selection(residual, ensemble_rank)
        
        # Combine the approximations
        return svd_approx + ensemble_approx
    
    else:  # mode == 'combined' or other
        # Apply ensemble directly and combine with SVD
        ensemble_approx = weighted_ensemble_column_selection(matrix, ensemble_rank)
        
        # Average the two approximations
        return (svd_approx + ensemble_approx) / 2

def weighted_ensemble_column_selection(matrix, rank, mode='weighted_vote'):
    """
    Select columns using a weighted ensemble approach.
    
    Args:
        matrix (numpy.ndarray): Input matrix
        rank (int): Target rank for approximation
        mode (str): Selection mode ('weighted_vote', 'leverage_score', etc.)
        
    Returns:
        numpy.ndarray: Approximated matrix
    """
    log_debug(f"Applying weighted ensemble column selection (rank={rank}, mode={mode})")
    
    # Different column selection strategies
    selected_cols = []
    
    if mode == 'weighted_vote':
        # Use multiple voting strategies with weights
        # 1. Leverage score sampling
        U, _, _ = la.svd(matrix, full_matrices=False)
        leverage_scores = np.sum(U[:, :min(rank, U.shape[1])]**2, axis=1)
        
        # 2. Column norms
        col_norms = np.linalg.norm(matrix, axis=0)
        normalized_col_norms = col_norms / np.sum(col_norms)
        
        # 3. Row norms (consider columns of the transpose)
        row_norms = np.linalg.norm(matrix, axis=1)
        normalized_row_norms = row_norms / np.sum(row_norms)
        
        # Combine scores with weights
        combined_scores = np.zeros(matrix.shape[1])
        
        # Apply strategies iteratively to select columns
        available_cols = list(range(matrix.shape[1]))
        
        with tqdm(total=rank, desc="Weighted Ensemble Selection") as pbar:
            for i in range(min(rank, matrix.shape[1])):
                # Update combined scores based on available columns
                combined_scores = np.zeros(matrix.shape[1])
                
                # Only consider available columns
                for col in available_cols:
                    # Weight different strategies
                    score1 = normalized_col_norms[col] if col < len(normalized_col_norms) else 0
                    score2 = 0  # Will be computed dynamically
                    
                    # Compute orthogonal projection score
                    if selected_cols:
                        # Compute how orthogonal this column is to already selected columns
                        selected_matrix = matrix[:, selected_cols]
                        current_col = matrix[:, col].reshape(-1, 1)
                        
                        # Project current column onto selected columns
                        projection = selected_matrix @ np.linalg.lstsq(selected_matrix, current_col, rcond=None)[0]
                        
                        # Orthogonality score is 1 - (projection similarity)
                        similarity = np.linalg.norm(projection) / np.linalg.norm(current_col)
                        score2 = 1 - similarity
                    else:
                        score2 = 1.0  # First column gets full orthogonality score
                    
                    # Combine scores with weights
                    combined_scores[col] = 0.3 * score1 + 0.7 * score2
                
                # Select column with highest combined score
                best_col = available_cols[np.argmax([combined_scores[col] for col in available_cols])]
                selected_cols.append(best_col)
                available_cols.remove(best_col)
                
                pbar.update(1)
    
    elif mode == 'leverage_score':
        # Simple leverage score sampling
        U, _, _ = la.svd(matrix, full_matrices=False)
        leverage_scores = np.sum(U[:, :min(rank, U.shape[1])]**2, axis=1)
        selected_cols = np.argsort(leverage_scores)[-rank:]
    
    else:  # Fallback to column norms
        col_norms = np.linalg.norm(matrix, axis=0)
        selected_cols = np.argsort(col_norms)[-rank:]
    
    # Compute approximation using selected columns
    try:
        C = matrix[:, selected_cols]
        U = np.linalg.pinv(C)
        return C @ U @ matrix
    except np.linalg.LinAlgError:
        log_warning("Linear algebra error in column selection. Using SVD fallback.")
        return svd_approximation(matrix, rank)

def matrix_adaptive_approximation(matrix, rank):
    """
    Adaptively choose the approximation method based on matrix properties.
    
    Args:
        matrix (numpy.ndarray): Input matrix
        rank (int): Target rank
        
    Returns:
        numpy.ndarray: Approximated matrix
    """
    log_debug(f"Applying matrix-adaptive approximation (rank={rank})")
    
    # Analyze matrix properties
    sparsity = 1.0 - (np.count_nonzero(matrix) / matrix.size)
    
    # Check if matrix is sparse
    if sparsity > 0.9:
        log_debug("Matrix is very sparse, using CUR decomposition")
        # Use CUR decomposition for sparse matrices
        from matrix_experiments import cur_decomposition
        return cur_decomposition(matrix, rank)[0]
    
    # Check column/row norm distribution
    col_norms = np.linalg.norm(matrix, axis=0)
    col_norm_std = np.std(col_norms) / np.mean(col_norms)
    
    if col_norm_std > 2.0:
        log_debug("Matrix has highly variable column norms, using hybrid method")
        # Use hybrid method for matrices with highly variable column norms
        return hybrid_svd_ensemble(matrix, rank, svd_ratio=0.7, mode='combined')
    
    # Default to SVD for balanced matrices
    log_debug("Using standard SVD for balanced matrix")
    return svd_approximation(matrix, rank)

# =============================================================================
# Rank Optimization Functions
# =============================================================================

def optimize_rank_distribution(matrix, total_rank=40, methods=['svd', 'ensemble']):
    """
    Find the optimal distribution of rank between different methods.
    
    Args:
        matrix (numpy.ndarray): Input matrix
        total_rank (int): Total rank budget
        methods (list): List of methods to combine
        
    Returns:
        dict: Optimal rank distribution and error
    """
    log_info(f"Optimizing rank distribution for total rank {total_rank}")
    
    best_error = float('inf')
    best_distribution = {}
    best_approx = None
    
    # Original matrix frobenius norm for relative error calculation
    orig_norm = la.norm(matrix, 'fro')
    
    # If only one method, use full rank
    if len(methods) == 1:
        if methods[0] == 'svd':
            approx = svd_approximation(matrix, total_rank)
        elif methods[0] == 'ensemble':
            approx = weighted_ensemble_column_selection(matrix, total_rank)
        else:
            approx = matrix_adaptive_approximation(matrix, total_rank)
            
        error = la.norm(matrix - approx, 'fro') / orig_norm
        return {
            'distribution': {methods[0]: total_rank},
            'error': error,
            'approximation': approx
        }
    
    # Try different rank distributions
    for svd_rank in range(0, total_rank + 1):
        ensemble_rank = total_rank - svd_rank
        
        # Create approximation with this distribution
        if 'svd' in methods and 'ensemble' in methods:
            if svd_rank == 0:
                approx = weighted_ensemble_column_selection(matrix, ensemble_rank)
            elif ensemble_rank == 0:
                approx = svd_approximation(matrix, svd_rank)
            else:
                approx = hybrid_svd_ensemble(matrix, total_rank, svd_ratio=svd_rank/total_rank)
        else:
            # Invalid method combination
            continue
            
        # Calculate error
        error = la.norm(matrix - approx, 'fro') / orig_norm
        
        # Update best if improved
        if error < best_error:
            best_error = error
            best_distribution = {'svd': svd_rank, 'ensemble': ensemble_rank}
            best_approx = approx
    
    return {
        'distribution': best_distribution,
        'error': best_error,
        'approximation': best_approx
    }

def optimize_svd_ratio(matrix, rank=40, start=0.2, end=0.9, steps=8):
    """
    Find optimal SVD ratio for hybrid method.
    
    Args:
        matrix (numpy.ndarray): Input matrix
        rank (int): Target rank
        start (float): Starting SVD ratio
        end (float): Ending SVD ratio
        steps (int): Number of steps to try
        
    Returns:
        tuple: (optimal_ratio, best_error, best_approx)
    """
    ratios = np.linspace(start, end, steps)
    best_error = float('inf')
    best_ratio = 0.5
    best_approx = None
    
    # Original matrix frobenius norm for relative error calculation
    orig_norm = la.norm(matrix, 'fro')
    
    # Try different ratios
    with tqdm(total=len(ratios), desc="Testing SVD ratios") as pbar:
        for ratio in ratios:
            approx = hybrid_svd_ensemble(matrix, rank, svd_ratio=ratio)
            error = la.norm(matrix - approx, 'fro') / orig_norm
            
            if error < best_error:
                best_error = error
                best_ratio = ratio
                best_approx = approx
            
            pbar.update(1)
    
    log_success(f"Optimal SVD ratio: {best_ratio:.2f}, best error: {best_error:.6f}")
    return best_ratio, best_error, best_approx

# =============================================================================
# Higher Rank Exploration
# =============================================================================

def run_higher_rank_experiments(matrices, max_rank=120, rank_step=20, 
                               optimize=True, save_dir='results/optimization'):
    """
    Run comprehensive higher rank experiments.
    
    Args:
        matrices (dict): Dictionary of matrices to test
        max_rank (int): Maximum rank to explore
        rank_step (int): Step size for rank exploration
        optimize (bool): Whether to run optimization
        save_dir (str): Directory to save results
        
    Returns:
        DataFrame: Results of experiments
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    optimized_methods = {}
    
    # First, explore higher ranks for each matrix
    for matrix_name, matrix in matrices.items():
        log_info(f"Running higher rank experiments for {matrix_name}")
        
        # Run basic exploration
        rank_results = []
        orig_norm = la.norm(matrix, 'fro')
        
        # Explore ranks
        for rank in range(rank_step, min(max_rank + 1, min(matrix.shape)), rank_step):
            log_info(f"Testing rank {rank} for {matrix_name}")
            
            # 1. Deterministic SVD
            start_time = time.time()
            svd_approx = svd_approximation(matrix, rank)
            svd_time = time.time() - start_time
            svd_error = la.norm(matrix - svd_approx, 'fro') / orig_norm
            
            rank_results.append({
                'matrix': matrix_name,
                'method': 'Deterministic SVD',
                'rank': rank,
                'error': svd_error,
                'time': svd_time
            })
            
            # 2. Ensemble method
            start_time = time.time()
            ensemble_approx = weighted_ensemble_column_selection(matrix, rank)
            ensemble_time = time.time() - start_time
            ensemble_error = la.norm(matrix - ensemble_approx, 'fro') / orig_norm
            
            rank_results.append({
                'matrix': matrix_name,
                'method': 'Weighted Ensemble',
                'rank': rank,
                'error': ensemble_error,
                'time': ensemble_time
            })
            
            # 3. Hybrid method
            start_time = time.time()
            hybrid_approx = hybrid_svd_ensemble(matrix, rank, svd_ratio=0.8)
            hybrid_time = time.time() - start_time
            hybrid_error = la.norm(matrix - hybrid_approx, 'fro') / orig_norm
            
            rank_results.append({
                'matrix': matrix_name,
                'method': 'Hybrid SVD-Ensemble',
                'rank': rank,
                'error': hybrid_error,
                'time': hybrid_time
            })
            
            # 4. Matrix-adaptive method
            start_time = time.time()
            adaptive_approx = matrix_adaptive_approximation(matrix, rank)
            adaptive_time = time.time() - start_time
            adaptive_error = la.norm(matrix - adaptive_approx, 'fro') / orig_norm
            
            rank_results.append({
                'matrix': matrix_name,
                'method': 'Matrix-Adaptive',
                'rank': rank,
                'error': adaptive_error,
                'time': adaptive_time
            })
        
        # Run optimization if requested
        if optimize:
            log_info(f"Optimizing methods for {matrix_name}")
            
            # Find optimal SVD ratio
            best_ratio, best_ratio_error, best_ratio_approx = optimize_svd_ratio(matrix)
            
            # Find optimal rank distribution
            opt_result = optimize_rank_distribution(matrix, total_rank=40, methods=['svd', 'ensemble'])
            
            # Store optimized methods
            optimized_methods[matrix_name] = {
                'svd_ratio': best_ratio,
                'best_ratio_error': best_ratio_error,
                'rank_distribution': opt_result['distribution'],
                'rank_distribution_error': opt_result['error']
            }
            
            # Create plots comparing error vs. rank for this matrix
            rank_df = pd.DataFrame(rank_results)
            
            plt.figure(figsize=(12, 8))
            for method in rank_df['method'].unique():
                method_df = rank_df[rank_df['method'] == method]
                plt.plot(method_df['rank'], method_df['error'], marker='o', label=method)
            
            plt.title(f'Error vs. Rank for Matrix {matrix_name}', fontsize=14)
            plt.xlabel('Rank', fontsize=12)
            plt.ylabel('Relative Error', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(save_dir, f'error_comparison_{matrix_name}.png'), dpi=300)
            plt.close()
        
        # Add results to overall list
        all_results.extend(rank_results)
    
    # Convert all results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(os.path.join(save_dir, 'optimization_comparison.csv'), index=False)
    
    if optimize:
        # Save optimization results
        with open(os.path.join(save_dir, 'optimization_results.json'), 'w') as f:
            json.dump(optimized_methods, f, indent=4)
    
    # Create summary plots
    
    # 1. Improvement heatmap
    pivot_df = results_df.pivot_table(
        index=['matrix', 'rank'],
        columns='method',
        values='error'
    )
    
    # Calculate improvement over SVD
    improvement_df = pivot_df.copy()
    for method in improvement_df.columns:
        if method != 'Deterministic SVD':
            improvement_df[method] = 100 * (pivot_df['Deterministic SVD'] - pivot_df[method]) / pivot_df['Deterministic SVD']
    
    # Drop SVD column
    if 'Deterministic SVD' in improvement_df.columns:
        improvement_df = improvement_df.drop('Deterministic SVD', axis=1)
    
    # Reshape for heatmap
    improvement_long = improvement_df.reset_index().melt(
        id_vars=['matrix', 'rank'],
        var_name='method',
        value_name='improvement'
    )
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    pivot = improvement_long.pivot_table(
        index=['matrix', 'rank'],
        columns='method',
        values='improvement'
    )
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(pivot, cmap=cmap, center=0, annot=True, fmt=".1f",
                linewidths=.5, cbar_kws={'label': 'Improvement Over SVD (%)'})
    
    plt.title('Improvement Over SVD (%) by Method, Matrix, and Rank', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'improvement_heatmap.png'), dpi=300)
    plt.close()
    
    # 2. Average improvement by rank and method
    avg_improvement = improvement_long.groupby(['method', 'rank'])['improvement'].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    for method in avg_improvement['method'].unique():
        method_df = avg_improvement[avg_improvement['method'] == method]
        plt.plot(method_df['rank'], method_df['improvement'], marker='o', label=method)
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.title('Average Improvement Over SVD by Rank', fontsize=14)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Average Improvement (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'optimized_improvement.png'), dpi=300)
    plt.close()
    
    # 3. Time comparison
    time_comparison = results_df.groupby(['method', 'rank'])['time'].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    for method in time_comparison['method'].unique():
        method_df = time_comparison[time_comparison['method'] == method]
        plt.plot(method_df['rank'], method_df['time'], marker='o', label=method)
    
    plt.title('Computation Time by Rank', fontsize=14)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'time_comparison.png'), dpi=300)
    plt.close()
    
    # Run final comparison with optimized methods
    if optimize:
        final_comparison = []
        
        for matrix_name, matrix in matrices.items():
            log_info(f"Comparing optimized methods for {matrix_name}")
            
            orig_norm = la.norm(matrix, 'fro')
            
            # Baseline: Deterministic SVD
            svd_approx = svd_approximation(matrix, 40)
            svd_error = la.norm(matrix - svd_approx, 'fro') / orig_norm
            
            final_comparison.append({
                'matrix': matrix_name,
                'method': 'Deterministic SVD (rank=40)',
                'error': svd_error,
                'improvement': 0.0
            })
            
            # Optimized ensemble
            ensemble_approx = weighted_ensemble_column_selection(matrix, 40)
            ensemble_error = la.norm(matrix - ensemble_approx, 'fro') / orig_norm
            ensemble_improvement = 100 * (svd_error - ensemble_error) / svd_error
            
            final_comparison.append({
                'matrix': matrix_name,
                'method': 'Weighted Ensemble (rank=40)',
                'error': ensemble_error,
                'improvement': ensemble_improvement
            })
            
            # Optimized hybrid with best ratio
            if matrix_name in optimized_methods:
                best_ratio = optimized_methods[matrix_name]['svd_ratio']
                hybrid_approx = hybrid_svd_ensemble(matrix, 40, svd_ratio=best_ratio)
                hybrid_error = la.norm(matrix - hybrid_approx, 'fro') / orig_norm
                hybrid_improvement = 100 * (svd_error - hybrid_error) / svd_error
                
                final_comparison.append({
                    'matrix': matrix_name,
                    'method': f'Optimized Hybrid (ratio={best_ratio:.2f})',
                    'error': hybrid_error,
                    'improvement': hybrid_improvement
                })
            
            # Matrix-adaptive method
            adaptive_approx = matrix_adaptive_approximation(matrix, 40)
            adaptive_error = la.norm(matrix - adaptive_approx, 'fro') / orig_norm
            adaptive_improvement = 100 * (svd_error - adaptive_error) / svd_error
            
            final_comparison.append({
                'matrix': matrix_name,
                'method': 'Matrix-Adaptive (rank=40)',
                'error': adaptive_error,
                'improvement': adaptive_improvement
            })
        
        # Save final comparison
        final_df = pd.DataFrame(final_comparison)
        final_df.to_csv(os.path.join(save_dir, 'optimized_methods_comparison.csv'), index=False)
    
    log_success("Matrix approximation optimization experiments completed!")
    return results_df

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function for higher rank exploration."""
    try:
        log_info("Starting higher rank matrix approximation experiments")
        
        # Define which matrices to test
        target_matrices = [
            "G2", "G3", "G4", "G5", 
            "G12", "G13", "G14", "G15"
        ]
        
        matrix_paths = download_all_matrices(subset=target_matrices)
        
        if not matrix_paths:
            log_error("Failed to download matrices. Exiting.")
            return
        
        # Load matrices
        log_info("Loading matrices...")
        matrices = load_all_matrices(matrix_paths)
        
        if not matrices:
            log_error("Failed to load matrices. Exiting.")
            return
        
        log_success(f"Successfully loaded {len(matrices)} matrices")
        
        # Run optimization experiments
        run_higher_rank_experiments(
            matrices,
            max_rank=100,
            rank_step=10,
            optimize=True,
            save_dir='results/optimization'
        )
        
    except KeyboardInterrupt:
        log_warning("Experiments interrupted by user")
    except Exception as e:
        log_error(f"Error running higher rank experiments: {str(e)}")
        import traceback
        log_error(traceback.format_exc())

if __name__ == "__main__":
    main()
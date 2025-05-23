#!/usr/bin/env python3
"""
Matrix Approximation Analysis Module
===================================

This module provides comprehensive analysis tools for evaluating and comparing
different matrix approximation methods, with a special focus on comparing
ensemble techniques against deterministic SVD.

The module includes:
1. Statistical analysis functions
2. Visualization tools
3. Detailed comparative metrics
4. Strategy recommendations for outperforming SVD
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from matplotlib.ticker import PercentFormatter
import json

# Import logging utilities if available
try:
    from florida_downloader import (
        log_info, log_success, log_warning, log_error, log_debug
    )
except ImportError:
    # Create placeholder logging functions if florida_downloader is not available
    def log_info(msg): print(f"[INFO] {msg}")
    def log_success(msg): print(f"[SUCCESS] {msg}")
    def log_warning(msg): print(f"[WARNING] {msg}")
    def log_error(msg): print(f"[ERROR] {msg}")
    def log_debug(msg): print(f"[DEBUG] {msg}")

# =============================================================================
# Statistical Analysis Functions
# =============================================================================

def calculate_statistical_significance(df, baseline_method="Deterministic SVD", metric="error", alpha=0.05):
    """
    Calculate statistical significance of differences between methods and baseline.
    
    Args:
        df (DataFrame): Results dataframe with columns 'method', 'matrix', and the metric
        baseline_method (str): The baseline method to compare against
        metric (str): The metric column to use for comparison
        alpha (float): Significance level
        
    Returns:
        DataFrame: Statistical test results
    """
    methods = df['method'].unique()
    test_results = []
    
    # Filter out the baseline method results
    baseline_df = df[df['method'] == baseline_method].copy()
    
    for method in methods:
        if method == baseline_method:
            continue
            
        method_df = df[df['method'] == method].copy()
        
        # Merge on matrix to ensure paired comparison
        merged = pd.merge(baseline_df, method_df, on='matrix', suffixes=('_baseline', '_method'))
        
        # Only proceed if we have enough data points
        if len(merged) >= 5:  # Minimum sample size for reasonable analysis
            # Calculate paired differences
            merged['diff'] = merged[f'{metric}_method'] - merged[f'{metric}_baseline']
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(
                merged[f'{metric}_method'], 
                merged[f'{metric}_baseline']
            )
            
            # Difference direction and significance
            mean_diff = merged['diff'].mean()
            is_significant = p_value < alpha
            is_better = mean_diff < 0  # For error metrics, lower is better
            
            test_results.append({
                'method': method,
                'baseline': baseline_method,
                'mean_diff': mean_diff,
                'percent_diff': (mean_diff / merged[f'{metric}_baseline'].mean()) * 100,
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': is_significant,
                'is_better': is_better,
                'is_significantly_better': is_significant and is_better
            })
        else:
            log_warning(f"Not enough paired samples for method {method} vs {baseline_method}")
    
    return pd.DataFrame(test_results)

def analyze_error_distribution(df, method_col='method', error_col='error'):
    """
    Analyze the distribution of errors for each method.
    
    Args:
        df (DataFrame): Results dataframe
        method_col (str): Column name for methods
        error_col (str): Column name for error values
        
    Returns:
        dict: Distribution statistics for each method
    """
    methods = df[method_col].unique()
    distribution_stats = {}
    
    for method in methods:
        method_errors = df[df[method_col] == method][error_col]
        
        if len(method_errors) > 0:
            stats_dict = {
                'mean': method_errors.mean(),
                'median': method_errors.median(),
                'std': method_errors.std(),
                'min': method_errors.min(),
                'max': method_errors.max(),
                'range': method_errors.max() - method_errors.min(),
                'iqr': method_errors.quantile(0.75) - method_errors.quantile(0.25),
                'skewness': method_errors.skew(),
                'kurtosis': method_errors.kurtosis(),
                'count': len(method_errors)
            }
            
            distribution_stats[method] = stats_dict
    
    return distribution_stats

def perform_matrix_specific_analysis(df, matrix_col='matrix', method_col='method', error_col='error'):
    """
    Analyze which methods perform best on specific matrices.
    
    Args:
        df (DataFrame): Results dataframe
        matrix_col (str): Column name for matrix identifiers
        method_col (str): Column name for methods
        error_col (str): Column name for error values
        
    Returns:
        DataFrame: Best methods for each matrix
    """
    matrices = df[matrix_col].unique()
    best_methods = []
    
    for matrix in matrices:
        matrix_df = df[df[matrix_col] == matrix].copy()
        
        # Sort by error (ascending) to find the best method
        matrix_df = matrix_df.sort_values(by=error_col)
        best_method = matrix_df.iloc[0]
        
        # Find the SVD result for this matrix
        svd_result = matrix_df[matrix_df[method_col] == 'Deterministic SVD']
        
        if not svd_result.empty:
            svd_error = svd_result.iloc[0][error_col]
            
            # Calculate improvement over SVD
            improvement = (svd_error - best_method[error_col]) / svd_error * 100
            
            best_methods.append({
                'matrix': matrix,
                'best_method': best_method[method_col],
                'best_error': best_method[error_col],
                'svd_error': svd_error,
                'improvement_percent': improvement,
                'beats_svd': best_method[error_col] < svd_error
            })
        else:
            log_warning(f"No SVD result found for matrix {matrix}")
    
    return pd.DataFrame(best_methods)

def correlation_analysis(df, label='matrix', method_filter=None):
    """
    Analyze correlations between matrix properties and method performance.
    
    Args:
        df (DataFrame): Results dataframe
        label (str): Column to use as index
        method_filter (list): List of methods to include in analysis
        
    Returns:
        DataFrame: Correlation matrix
    """
    if method_filter:
        df = df[df['method'].isin(method_filter)].copy()
    
    # Pivot the data to have methods as columns
    pivot_df = df.pivot(index=label, columns='method', values='error')
    
    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()
    
    return corr_matrix

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_error_improvement_distribution(df, output_path=None):
    """
    Plot the distribution of error improvements over SVD.
    
    Args:
        df (DataFrame): Results DataFrame with 'method' and 'improvement' columns
        output_path (str): Path to save the plot, if None the plot is displayed
    """
    plt.figure(figsize=(14, 8))
    
    # Create violin plot
    ax = sns.violinplot(x='method', y='improvement', data=df, inner='quartile')
    
    # Add individual points
    sns.stripplot(x='method', y='improvement', data=df, color='black', alpha=0.5, size=4, jitter=True)
    
    # Add a horizontal line at zero
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    plt.title('Distribution of Error Improvement Over SVD by Method', fontsize=14)
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Improvement Over SVD (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log_info(f"Saved improvement distribution plot to {output_path}")
    else:
        plt.show()

def plot_matrix_specific_performance(matrix_analysis_df, output_path=None):
    """
    Plot heatmap showing which methods perform best on which matrices.
    
    Args:
        matrix_analysis_df (DataFrame): Result from perform_matrix_specific_analysis
        output_path (str): Path to save the plot, if None the plot is displayed
    """
    plt.figure(figsize=(14, 8))
    
    # Create a colormap for improvement percentage
    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Red (negative) to Blue (positive)
    
    # Plot heatmap
    pivot_table = pd.pivot_table(
        matrix_analysis_df,
        values='improvement_percent',
        index='matrix',
        columns='best_method'
    )
    
    sns.heatmap(pivot_table, cmap=cmap, center=0, annot=True, fmt=".2f",
                linewidths=.5, cbar_kws={'label': 'Improvement Over SVD (%)'})
    
    plt.title('Best Method Performance by Matrix', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log_info(f"Saved matrix-specific performance plot to {output_path}")
    else:
        plt.show()

def plot_correlation_matrix(corr_matrix, output_path=None):
    """
    Plot correlation matrix heatmap.
    
    Args:
        corr_matrix (DataFrame): Correlation matrix
        output_path (str): Path to save the plot, if None the plot is displayed
    """
    plt.figure(figsize=(12, 10))
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f')
    
    plt.title('Correlation Between Method Errors', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log_info(f"Saved correlation matrix plot to {output_path}")
    else:
        plt.show()

def plot_error_time_tradeoff(df, output_path=None):
    """
    Plot trade-off between error and computation time.
    
    Args:
        df (DataFrame): Results DataFrame with 'method', 'error', and 'time' columns
        output_path (str): Path to save the plot, if None the plot is displayed
    """
    plt.figure(figsize=(14, 8))
    
    # Calculate mean error and time for each method
    method_summary = df.groupby('method')[['error', 'time']].mean().reset_index()
    
    # Create scatter plot
    scatter = plt.scatter(
        method_summary['time'], 
        method_summary['error'], 
        s=100, 
        c=range(len(method_summary)), 
        cmap='viridis', 
        alpha=0.7
    )
    
    # Add method labels
    for i, row in method_summary.iterrows():
        plt.annotate(
            row['method'], 
            (row['time'], row['error']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    plt.title('Error vs. Computation Time Trade-off', fontsize=14)
    plt.xlabel('Mean Computation Time (seconds)', fontsize=12)
    plt.ylabel('Mean Error', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add logarithmic scale for time if range is large
    if method_summary['time'].max() / method_summary['time'].min() > 10:
        plt.xscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log_info(f"Saved error-time trade-off plot to {output_path}")
    else:
        plt.show()

def plot_parameter_sensitivity(data, x_param, y_metric, hue_param=None, output_path=None):
    """
    Plot how a parameter affects performance.
    
    Args:
        data (DataFrame): Data with parameter and metric columns
        x_param (str): Parameter to plot on x-axis
        y_metric (str): Metric to plot on y-axis
        hue_param (str): Parameter to use for color grouping (optional)
        output_path (str): Path to save the plot, if None the plot is displayed
    """
    plt.figure(figsize=(12, 8))
    
    if hue_param:
        ax = sns.lineplot(x=x_param, y=y_metric, hue=hue_param, data=data, marker='o')
    else:
        ax = sns.lineplot(x=x_param, y=y_metric, data=data, marker='o')
    
    plt.title(f'Effect of {x_param} on {y_metric}', fontsize=14)
    plt.xlabel(x_param, fontsize=12)
    plt.ylabel(y_metric, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if hue_param:
        plt.legend(title=hue_param)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log_info(f"Saved parameter sensitivity plot to {output_path}")
    else:
        plt.show()

# =============================================================================
# Higher Rank Exploration Functions
# =============================================================================

def explore_higher_rank_approximations(matrices, max_rank=100, rank_step=10, output_dir='results/analysis/rank_exploration'):
    """
    Explore how higher ranks affect approximation quality for both SVD and ensemble methods.
    
    Args:
        matrices (dict): Dictionary of matrices to test
        max_rank (int): Maximum rank to explore
        rank_step (int): Step size for rank exploration
        output_dir (str): Directory to save results
        
    Returns:
        DataFrame: Results of rank exploration
    """
    import scipy.linalg as la
    from tqdm import tqdm
    import time
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    log_info(f"Exploring rank effects from rank {rank_step} to {max_rank} (step: {rank_step})")
    
    results = []
    
    # Iterate through matrices
    for matrix_name, matrix in tqdm(matrices.items(), desc="Processing matrices"):
        log_info(f"Exploring rank effects for matrix {matrix_name}")
        
        # Calculate full SVD for reference
        start_time = time.time()
        U, s, Vt = la.svd(matrix, full_matrices=False)
        svd_time = time.time() - start_time
        
        # Original matrix frobenius norm for relative error calculation
        orig_norm = la.norm(matrix, 'fro')
        
        # Explore different ranks
        for rank in tqdm(range(rank_step, min(max_rank + 1, min(matrix.shape)), rank_step), 
                          desc=f"Testing ranks for {matrix_name}"):
            
            # 1. Standard SVD approximation
            start_time = time.time()
            svd_approx = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
            svd_time_rank = time.time() - start_time
            svd_error = la.norm(matrix - svd_approx, 'fro') / orig_norm
            
            # Record SVD result
            results.append({
                'matrix': matrix_name,
                'method': 'Deterministic SVD',
                'rank': rank,
                'error': svd_error,
                'time': svd_time_rank + svd_time,  # Include SVD computation time
                'relative_to_svd': 0.0  # SVD is the baseline
            })
            
            # 2. Try ensemble-based methods with higher rank
            # Simplified ensemble method for demonstration
            try:
                # Create a higher-rank ensemble approximation by combining columns
                start_time = time.time()
                
                # Simple random column selection as a placeholder for your actual ensemble method
                selected_cols = np.random.choice(matrix.shape[1], size=rank, replace=False)
                col_subset = matrix[:, selected_cols]
                
                # Calculate pseudo-inverse to get approximate factorization
                col_pinv = np.linalg.pinv(col_subset)
                ensemble_approx = col_subset @ col_pinv @ matrix
                
                ensemble_time = time.time() - start_time
                ensemble_error = la.norm(matrix - ensemble_approx, 'fro') / orig_norm
                
                # Calculate improvement over SVD
                improvement = (svd_error - ensemble_error) / svd_error * 100
                
                # Record ensemble result
                results.append({
                    'matrix': matrix_name,
                    'method': 'Higher-Rank Ensemble',
                    'rank': rank,
                    'error': ensemble_error,
                    'time': ensemble_time,
                    'relative_to_svd': improvement
                })
                
                # 3. Try hybrid approach (combine SVD and ensemble)
                start_time = time.time()
                
                # Use SVD for part of the rank, ensemble for the rest
                svd_portion = int(rank * 0.8)  # 80% SVD, 20% ensemble
                ensemble_portion = rank - svd_portion
                
                # SVD part
                svd_part = U[:, :svd_portion] @ np.diag(s[:svd_portion]) @ Vt[:svd_portion, :]
                
                # Ensemble part - operate on the residual
                residual = matrix - svd_part
                selected_cols = np.random.choice(residual.shape[1], size=ensemble_portion, replace=False)
                col_subset = residual[:, selected_cols]
                col_pinv = np.linalg.pinv(col_subset)
                ensemble_part = col_subset @ col_pinv @ residual
                
                # Combined approximation
                hybrid_approx = svd_part + ensemble_part
                hybrid_time = time.time() - start_time
                hybrid_error = la.norm(matrix - hybrid_approx, 'fro') / orig_norm
                
                # Calculate improvement over SVD
                hybrid_improvement = (svd_error - hybrid_error) / svd_error * 100
                
                # Record hybrid result
                results.append({
                    'matrix': matrix_name,
                    'method': 'Hybrid SVD-Ensemble',
                    'rank': rank,
                    'error': hybrid_error,
                    'time': hybrid_time,
                    'relative_to_svd': hybrid_improvement
                })
                
            except Exception as e:
                log_error(f"Error during higher-rank ensemble for {matrix_name}, rank {rank}: {str(e)}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'rank_exploration_results.csv'), index=False)
    
    # Create summary plots
    plot_rank_vs_error(results_df, output_dir)
    plot_rank_vs_improvement(results_df, output_dir)
    
    return results_df

def plot_rank_vs_error(results_df, output_dir):
    """
    Plot how error changes with rank for different methods.
    
    Args:
        results_df (DataFrame): Results from rank exploration
        output_dir (str): Directory to save plots
    """
    # Create a plot for each matrix
    for matrix in results_df['matrix'].unique():
        matrix_df = results_df[results_df['matrix'] == matrix]
        
        plt.figure(figsize=(12, 8))
        
        # Plot error vs rank for each method
        for method in matrix_df['method'].unique():
            method_df = matrix_df[matrix_df['method'] == method]
            plt.plot(method_df['rank'], method_df['error'], marker='o', label=method)
        
        plt.title(f'Error vs. Rank for Matrix {matrix}', fontsize=14)
        plt.xlabel('Rank', fontsize=12)
        plt.ylabel('Relative Error (Frobenius Norm)', fontsize=12)
        plt.yscale('log')  # Log scale often helps visualize error curves
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'rank_vs_error_{matrix}.png'), dpi=300)
        plt.close()

def plot_rank_vs_improvement(results_df, output_dir):
    """
    Plot how improvement over SVD changes with rank.
    
    Args:
        results_df (DataFrame): Results from rank exploration
        output_dir (str): Directory to save plots
    """
    # Filter out SVD results since improvement is always 0
    ensemble_df = results_df[results_df['method'] != 'Deterministic SVD']
    
    # Create a plot for each matrix
    for matrix in ensemble_df['matrix'].unique():
        matrix_df = ensemble_df[ensemble_df['matrix'] == matrix]
        
        plt.figure(figsize=(12, 8))
        
        # Plot improvement vs rank for each method
        for method in matrix_df['method'].unique():
            method_df = matrix_df[matrix_df['method'] == method]
            plt.plot(method_df['rank'], method_df['relative_to_svd'], marker='o', label=method)
        
        # Add horizontal line at y=0 (SVD baseline)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.title(f'Improvement Over SVD vs. Rank for Matrix {matrix}', fontsize=14)
        plt.xlabel('Rank', fontsize=12)
        plt.ylabel('Improvement Over SVD (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'rank_vs_improvement_{matrix}.png'), dpi=300)
        plt.close()
    
    # Create a summary plot showing the best improvement for each method-rank combination
    best_improvements = ensemble_df.groupby(['method', 'rank'])['relative_to_svd'].mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    
    for method in best_improvements['method'].unique():
        method_df = best_improvements[best_improvements['method'] == method]
        plt.plot(method_df['rank'], method_df['relative_to_svd'], marker='o', label=method)
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.title('Average Improvement Over SVD vs. Rank Across All Matrices', fontsize=14)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Average Improvement Over SVD (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'average_rank_vs_improvement.png'), dpi=300)
    plt.close()

# =============================================================================
# Comprehensive Analysis Functions
# =============================================================================

def analyze_ensemble_voting(voting_results_path, output_dir='results/analysis'):
    """
    Analyze ensemble voting results to find strategies for beating SVD.
    
    Args:
        voting_results_path (str): Path to the CSV file with voting results
        output_dir (str): Directory to save analysis results
        
    Returns:
        dict: Analysis results and recommendations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load voting results
    voting_df = pd.read_csv(voting_results_path)
    
    # Quick overview
    method_summary = voting_df.groupby('method')[['error', 'improvement', 'time']].agg(
        ['mean', 'std', 'min', 'max']
    )
    
    # Find methods that beat SVD on any matrix
    methods_beating_svd = voting_df[voting_df['improvement'] > 0].copy()
    
    # Group by matrix to see which matrices are easier to beat
    matrix_summary = voting_df.groupby('matrix')[['improvement']].agg(
        ['mean', 'std', 'min', 'max']
    )
    
    # Check if any method consistently beats SVD
    consistent_winners = []
    for method in voting_df['method'].unique():
        method_df = voting_df[voting_df['method'] == method]
        beat_svd_count = sum(method_df['improvement'] > 0)
        
        if beat_svd_count > 0:
            consistent_winners.append({
                'method': method,
                'matrices_beating_svd': beat_svd_count,
                'total_matrices': len(method_df),
                'success_rate': beat_svd_count / len(method_df),
                'avg_improvement': method_df['improvement'].mean()
            })

    # Convert to DataFrame and sort if not empty
    if consistent_winners:
        consistent_winners_df = pd.DataFrame(consistent_winners).sort_values(by='avg_improvement', ascending=False)
    else:
        consistent_winners_df = pd.DataFrame(columns=['method', 'matrices_beating_svd', 'total_matrices', 'success_rate', 'avg_improvement'])
        log_warning("No methods consistently beat SVD on any matrix.")
    
    # Prepare recommendations based on the analysis
    recommendations = {
        'best_overall_method': voting_df.loc[voting_df['improvement'].idxmax()]['method'] 
                               if not voting_df.empty else None,
        'best_matrix_to_target': voting_df.loc[voting_df['improvement'].idxmax()]['matrix'] 
                                if not voting_df.empty else None,
        'max_improvement_achieved': voting_df['improvement'].max() if not voting_df.empty else None,
        'most_consistent_method': consistent_winners_df.iloc[0]['method'] 
                                 if not consistent_winners_df.empty else None,
    }
    
    # Create comprehensive plots
    
    # 1. Improvement distribution by method
    plot_error_improvement_distribution(
        voting_df, 
        output_path=os.path.join(output_dir, 'ensemble_improvement_distribution.png')
    )
    
    # 2. Error vs Time trade-off
    plot_error_time_tradeoff(
        voting_df, 
        output_path=os.path.join(output_dir, 'ensemble_error_time_tradeoff.png')
    )
    
    # 3. Performance heatmap by matrix and method
    plt.figure(figsize=(14, 8))
    pivot_data = voting_df.pivot(index='matrix', columns='method', values='improvement')
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(pivot_data, cmap=cmap, center=0, annot=True, fmt=".1f",
                linewidths=.5, cbar_kws={'label': 'Improvement Over SVD (%)'})
    
    plt.title('Improvement Over SVD (%) by Method and Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_matrix_method_heatmap.png'), dpi=300)
    
    # Save summary tables
    method_summary.to_csv(os.path.join(output_dir, 'method_summary_stats.csv'))
    matrix_summary.to_csv(os.path.join(output_dir, 'matrix_summary_stats.csv'))
    if not consistent_winners_df.empty:
        consistent_winners_df.to_csv(os.path.join(output_dir, 'consistent_winners.csv'), index=False)
    
    # Prepare complete analysis report
    full_analysis = {
        'summary': {
            'total_methods': len(voting_df['method'].unique()),
            'total_matrices': len(voting_df['matrix'].unique()),
            'methods_beating_svd_count': len(methods_beating_svd['method'].unique()),
            'matrices_beaten_by_ensemble': len(methods_beating_svd['matrix'].unique())
        },
        'best_methods': consistent_winners_df.to_dict('records') if not consistent_winners_df.empty else [],
        'best_matrices_to_target': matrix_summary.sort_values(('improvement', 'max'), ascending=False).to_dict(),
        'recommendations': recommendations,
        'strategies_to_beat_svd': [
            "1. Focus on matrices where ensemble methods show best results",
            "2. Use the best performing ensemble method for each specific matrix",
            "3. Optimize hyperparameters specifically for challenging matrices",
            "4. Consider hybrid approaches combining the best aspects of different methods",
            "5. Try increasing the diversity of ensemble components to capture different matrix structures"
        ]
    }
    
    # Save as JSON
    with open(os.path.join(output_dir, 'ensemble_analysis_report.json'), 'w') as f:
        json.dump(full_analysis, f, indent=4)
    
    log_success(f"Completed ensemble voting analysis. Results saved to {output_dir}")
    return full_analysis

def perform_comprehensive_analysis(voting_results_path, output_dir='results/analysis'):
    """
    Run a comprehensive analysis to determine strategies for beating SVD.
    
    Args:
        voting_results_path (str): Path to the CSV file with voting results
        output_dir (str): Directory to save analysis results
        
    Returns:
        dict: Analysis results and strategy recommendations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load voting results
    voting_df = pd.read_csv(voting_results_path)
    
    # Check current performance vs SVD
    if voting_df['improvement'].max() <= 0:
        log_warning("No method is currently beating SVD. Analyzing for strategies to improve...")
    else:
        log_success(f"Found methods that beat SVD! Max improvement: {voting_df['improvement'].max():.2f}%")
    
    # Run ensemble voting analysis
    ensemble_analysis = analyze_ensemble_voting(voting_results_path, output_dir)
    
    # Analyze method performance by matrix type
    matrix_analysis = perform_matrix_specific_analysis(
        voting_df, matrix_col='matrix', method_col='method', error_col='error'
    )
    matrix_analysis.to_csv(os.path.join(output_dir, 'matrix_method_performance.csv'), index=False)
    
    # Distribution stats
    error_distribution = analyze_error_distribution(voting_df)
    with open(os.path.join(output_dir, 'error_distribution_stats.json'), 'w') as f:
        json.dump(error_distribution, f, indent=4)
    
    # Generate specialized strategies based on patterns observed
    strategies = generate_svd_beating_strategies(voting_df, error_distribution)
    
    # Create a comprehensive result dictionary
    result = {
        'current_status': {
            'any_method_beats_svd': voting_df['improvement'].max() > 0,
            'best_current_improvement': voting_df['improvement'].max(),
            'best_current_method': voting_df.loc[voting_df['improvement'].idxmax()]['method'] 
                                  if not voting_df.empty and voting_df['improvement'].max() > 0 else None
        },
        'ensemble_analysis': ensemble_analysis,
        'detailed_strategies': strategies
    }
    
    # Save the comprehensive analysis
    with open(os.path.join(output_dir, 'comprehensive_analysis.json'), 'w') as f:
        json.dump(result, f, indent=4)
    
    log_success(f"Comprehensive analysis complete. Results and strategies saved to {output_dir}")
    return result

def generate_svd_beating_strategies(df, error_distribution):
    """
    Generate specific strategies to beat SVD based on data patterns.
    
    Args:
        df (DataFrame): Results dataframe
        error_distribution (dict): Distribution statistics for each method
        
    Returns:
        list: Detailed strategies to potentially beat SVD
    """
    strategies = []
    
    # Strategy 1: Identify which matrices are closest to being beaten by ensemble methods
    nearly_beaten = df[df['improvement'] > -1].copy()  # Within 1% of SVD
    if not nearly_beaten.empty:
        best_candidates = nearly_beaten.sort_values('improvement', ascending=False)
        strategies.append({
            'name': 'Focus on nearly-beaten matrices',
            'description': 'Some matrices are very close to being beaten by ensemble methods',
            'matrices_to_target': best_candidates['matrix'].unique().tolist()[:3],
            'recommended_methods': best_candidates['method'].unique().tolist()[:3],
            'expected_improvement_needed': abs(best_candidates['improvement'].iloc[0]) if best_candidates['improvement'].iloc[0] < 0 else 0
        })
    
    # Strategy 2: Method hybridization
    best_methods = df.groupby('method')['improvement'].mean().sort_values(ascending=False)
    if len(best_methods) >= 2:
        top_methods = best_methods.index[:2].tolist()
        strategies.append({
            'name': 'Method hybridization',
            'description': 'Create a hybrid method combining strengths of the top performers',
            'methods_to_combine': top_methods,
            'approach': 'Develop a meta-ensemble that dynamically selects or weights between these methods based on matrix properties'
        })
    
    # Strategy 3: Matrix-specific optimization
    matrix_perf = df.groupby(['matrix', 'method'])['improvement'].max().reset_index()
    matrix_perf = matrix_perf.sort_values('improvement', ascending=False)
    
    if not matrix_perf.empty:
        strategies.append({
            'name': 'Matrix-specific optimization',
            'description': 'Optimize hyperparameters specifically for each matrix type',
            'matrix_method_pairs': [
                {'matrix': row['matrix'], 'method': row['method']} 
                for _, row in matrix_perf.head(3).iterrows()
            ]
        })
    
    # Strategy 4: Ensemble diversity
    strategies.append({
        'name': 'Increase ensemble diversity',
        'description': 'Add more diverse models to the ensemble to capture different patterns',
        'approaches': [
            'Add non-RL methods to the ensemble',
            'Train on different subsets of matrix data',
            'Use different hyperparameters for ensemble members',
            'Explore different state representations for RL agents'
        ]
    })
    
    # Strategy 5: Alternative error metrics
    strategies.append({
        'name': 'Optimize for specific error metrics',
        'description': 'SVD minimizes Frobenius norm. Try optimizing for metrics more relevant to your use case.',
        'alternative_metrics': [
            'Spectral norm (max singular value)',
            'Nuclear norm (sum of singular values)',
            'Element-wise max error (L-infinity norm)',
            'Application-specific error metrics'
        ]
    })
    
    # Strategy 6: Advanced RL techniques
    strategies.append({
        'name': 'Advanced RL techniques',
        'description': 'Implement more sophisticated RL algorithms that may better learn the matrix structure',
        'suggested_techniques': [
            'Proximal Policy Optimization (PPO)',
            'Soft Actor-Critic (SAC)',
            'Model-based RL with matrix structure prediction',
            'Meta-learning for quick adaptation to new matrices'
        ]
    })
    
    return strategies

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Run the analysis on the ensemble results."""
    try:
        log_info("Running comprehensive matrix approximation analysis")
        
        # Path to the ensemble voting results
        voting_results_path = 'results/ensemble_voting_comparison.csv'
        
        # Output directory for analysis results
        output_dir = 'results/analysis'
        
        # Run comprehensive analysis
        analysis_results = perform_comprehensive_analysis(voting_results_path, output_dir)
        
        # Create a summary text file with key findings
        with open(os.path.join(output_dir, 'summary_findings.txt'), 'w') as f:
            f.write("MATRIX APPROXIMATION ANALYSIS SUMMARY\n")
            f.write("=====================================\n\n")
            
            f.write("Current Status:\n")
            f.write(f"- Any method beats SVD: {analysis_results['current_status']['any_method_beats_svd']}\n")
            f.write(f"- Best current improvement: {analysis_results['current_status']['best_current_improvement']:.2f}%\n")
            if analysis_results['current_status']['best_current_method']:
                f.write(f"- Best current method: {analysis_results['current_status']['best_current_method']}\n\n")
            else:
                f.write("- No method currently beats SVD\n\n")
            
            f.write("Key Strategies to Beat SVD:\n")
            for i, strategy in enumerate(analysis_results['detailed_strategies'], 1):
                f.write(f"{i}. {strategy['name']}: {strategy['description']}\n")
            
            f.write("\nFor detailed analysis, please see the JSON files in this directory.\n")
        
        log_success("Analysis complete! Check results in the analysis directory.")
        
        # Return key findings to the caller
        return {
            'any_method_beats_svd': analysis_results['current_status']['any_method_beats_svd'],
            'best_improvement': analysis_results['current_status']['best_current_improvement'],
            'recommended_strategies': [s['name'] for s in analysis_results['detailed_strategies']]
        }
        
    except FileNotFoundError:
        log_error(f"Could not find the ensemble voting results file at {voting_results_path}")
        return {"error": "Results file not found"}
    except Exception as e:
        log_error(f"Error during analysis: {str(e)}")
        import traceback
        log_error(traceback.format_exc())
        return {"error": str(e)}

if __name__ == "__main__":
    main()
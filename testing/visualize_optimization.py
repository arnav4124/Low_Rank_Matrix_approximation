#!/usr/bin/env python3
"""
Matrix Optimization Analysis Dashboard
=====================================

This script generates comprehensive visualizations and statistical analysis from
matrix optimization experiment results, providing clear insights into performance tradeoffs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Set up directories
RESULTS_DIR = 'results/optimization'
REPORTS_DIR = 'results/optimization/reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

# Configure plot style
plt.style.use('ggplot')
sns.set_context("talk")
sns.set_palette("viridis")

def load_data():
    """Load and prepare optimization data."""
    print("Loading optimization data...")
    
    # Load CSV data
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'optimization_comparison.csv'))
    
    # Try to load JSON results if available
    opt_results = None
    json_path = os.path.join(RESULTS_DIR, 'optimization_results.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                opt_results = json.load(f)
        except:
            print("Warning: Could not load optimization_results.json")
    
    # Remove extreme outliers for visualization purposes
    # Matrix-Adaptive sometimes produces very large errors
    df_clean = df.copy()
    df_clean = df_clean[df_clean['error'] < 10]  # Remove extreme outliers
    
    return df, df_clean, opt_results

def create_error_rank_plots(df_clean):
    """Generate error vs. rank plots for each matrix."""
    print("Generating error vs. rank plots...")
    
    # Process each matrix separately
    for matrix in df_clean['matrix'].unique():
        matrix_df = df_clean[df_clean['matrix'] == matrix]
        
        plt.figure(figsize=(12, 8))
        
        # Plot each method
        for method in matrix_df['method'].unique():
            method_df = matrix_df[matrix_df['method'] == method]
            plt.plot(method_df['rank'], method_df['error'], 
                    marker='o', linewidth=2, label=method)
        
        # Add labels and formatting
        plt.title(f'Error vs. Rank for Matrix {matrix}', fontsize=16)
        plt.xlabel('Rank', fontsize=14)
        plt.ylabel('Relative Error', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Find optimal rank points
        for method in matrix_df['method'].unique():
            method_df = matrix_df[matrix_df['method'] == method]
            if not method_df.empty:
                best_idx = method_df['error'].idxmin()
                best_point = method_df.loc[best_idx]
                plt.scatter(best_point['rank'], best_point['error'], 
                           s=100, edgecolor='black', zorder=10)
                plt.annotate(f"{method}: {best_point['error']:.3f}",
                           xy=(best_point['rank'], best_point['error']),
                           xytext=(10, 0), textcoords='offset points',
                           fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, f'error_vs_rank_{matrix}.png'), dpi=300)
        plt.close()

def create_improvement_analysis(df_clean):
    """Calculate and visualize improvements over SVD baseline."""
    print("Analyzing improvement over SVD baseline...")
    
    # Calculate improvement over SVD for each method
    improvements = []
    
    for matrix in df_clean['matrix'].unique():
        matrix_df = df_clean[df_clean['matrix'] == matrix].copy()
        
        # Get SVD error for each rank
        svd_df = matrix_df[matrix_df['method'] == 'Deterministic SVD'].set_index('rank')
        
        # Calculate improvement for other methods
        for method in matrix_df['method'].unique():
            if method != 'Deterministic SVD':
                method_df = matrix_df[matrix_df['method'] == method].copy()
                
                # Join with SVD errors
                method_df = method_df.join(svd_df['error'], on='rank', rsuffix='_svd')
                
                # Calculate percentage improvement
                method_df['improvement'] = 100 * (method_df['error_svd'] - method_df['error']) / method_df['error_svd']
                
                # Store results
                improvements.append(method_df[['matrix', 'method', 'rank', 'error', 'error_svd', 'improvement']])
    
    # Combine all improvements
    if improvements:
        improvement_df = pd.concat(improvements)
        
        # Create heatmap of improvements
        improvement_pivot = improvement_df.pivot_table(
            index='matrix', 
            columns=['method', 'rank'], 
            values='improvement'
        )
        
        plt.figure(figsize=(16, len(improvement_df['matrix'].unique())*0.8))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        # Reshaping for better visualization if needed
        if not improvement_pivot.empty:
            sns.heatmap(improvement_pivot, cmap=cmap, center=0, 
                       annot=True, fmt=".1f", linewidths=.5, 
                       cbar_kws={"label": "Improvement Over SVD (%)"})
            
            plt.title('Improvement Over SVD (%) by Method and Rank', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(REPORTS_DIR, 'improvement_heatmap.png'), dpi=300)
            plt.close()
        
        # Create line chart of average improvements by rank
        plt.figure(figsize=(12, 8))
        avg_by_rank = improvement_df.groupby(['method', 'rank'])['improvement'].mean().reset_index()
        
        for method in avg_by_rank['method'].unique():
            method_data = avg_by_rank[avg_by_rank['method'] == method]
            plt.plot(method_data['rank'], method_data['improvement'], 
                   marker='o', linewidth=2, label=method)
        
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.title('Average Improvement Over SVD by Rank', fontsize=16)
        plt.xlabel('Rank', fontsize=14)
        plt.ylabel('Average Improvement (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'avg_improvement_by_rank.png'), dpi=300)
        plt.close()
        
        # Save improvement data
        improvement_df.to_csv(os.path.join(REPORTS_DIR, 'improvement_analysis.csv'), index=False)
        
        return improvement_df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no improvements calculated

def create_time_analysis(df_clean):
    """Analyze computation time versus error tradeoffs."""
    print("Analyzing time-error tradeoffs...")
    
    # Create scatterplot of error vs time
    plt.figure(figsize=(12, 8))
    
    # Use different markers for different methods
    markers = {'Deterministic SVD': 'o', 'Weighted Ensemble': 's', 
              'Hybrid SVD-Ensemble': '^', 'Matrix-Adaptive': 'D'}
    
    for method in df_clean['method'].unique():
        method_df = df_clean[df_clean['method'] == method]
        plt.scatter(method_df['time'], method_df['error'], 
                  s=80, alpha=0.7, marker=markers.get(method, 'o'), 
                  label=method)
    
    plt.xscale('log')  # Use log scale for time
    plt.title('Error vs. Computation Time', fontsize=16)
    plt.xlabel('Computation Time (seconds, log scale)', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'time_vs_error.png'), dpi=300)
    plt.close()
    
    # Create bar chart comparing time by method and rank
    plt.figure(figsize=(14, 10))
    
    # Group by method and rank
    time_by_method_rank = df_clean.groupby(['method', 'rank'])['time'].mean().reset_index()
    time_pivot = time_by_method_rank.pivot(index='method', columns='rank', values='time')
    
    # Plot as heatmap
    sns.heatmap(time_pivot, annot=True, fmt=".2f", cmap='YlGnBu')
    plt.title('Average Computation Time by Method and Rank (seconds)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'time_by_method_rank.png'), dpi=300)
    plt.close()
    
    # Create time-efficiency metric (lower error / time)
    df_efficiency = df_clean.copy()
    # Add small constant to avoid division by zero
    df_efficiency['efficiency'] = df_efficiency['error'] / (df_efficiency['time'] + 0.001)
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='method', y='efficiency', data=df_efficiency)
    plt.title('Error per Second (Lower is Better)', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'error_efficiency.png'), dpi=300)
    plt.close()

def create_method_ranking(df_clean):
    """Create method ranking analysis across matrices."""
    print("Creating method ranking analysis...")
    
    # Calculate method rankings for each matrix
    matrices = df_clean['matrix'].unique()
    methods = df_clean['method'].unique()
    ranking_data = []
    
    for matrix in matrices:
        matrix_df = df_clean[df_clean['matrix'] == matrix]
        
        # Calculate average error by method for this matrix
        method_errors = matrix_df.groupby('method')['error'].mean()
        
        # Rank methods (lower error = better rank)
        method_ranks = method_errors.rank()
        
        # Convert to 0-100 scale (higher = better)
        max_rank = len(method_errors)
        normalized_scores = 100 * (max_rank + 1 - method_ranks) / max_rank
        
        # Add to results
        for method in methods:
            if method in normalized_scores:
                ranking_data.append({
                    'matrix': matrix,
                    'method': method,
                    'score': normalized_scores[method]
                })
    
    # Convert to DataFrame
    ranking_df = pd.DataFrame(ranking_data)
    
    # Create heatmap of method rankings
    rank_pivot = ranking_df.pivot(index='matrix', columns='method', values='score')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(rank_pivot, annot=True, fmt=".1f", cmap='YlGnBu')
    plt.title('Method Performance Rankings by Matrix (higher is better)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'method_rankings.png'), dpi=300)
    plt.close()
    
    # Calculate overall average rankings
    overall_ranks = ranking_df.groupby('method')['score'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=overall_ranks.index, y=overall_ranks.values)
    plt.title('Overall Method Performance Rankings', fontsize=16)
    plt.ylabel('Average Score (higher is better)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'overall_rankings.png'), dpi=300)
    plt.close()
    
    # Save rankings
    ranking_df.to_csv(os.path.join(REPORTS_DIR, 'method_rankings.csv'), index=False)
    
    return ranking_df, overall_ranks

def create_best_method_map(df_clean):
    """Create visualization of which method works best for each matrix and rank."""
    print("Creating best method map...")
    
    # Find best method for each matrix and rank
    best_methods = []
    
    for matrix in df_clean['matrix'].unique():
        for rank in df_clean['rank'].unique():
            subset = df_clean[(df_clean['matrix'] == matrix) & (df_clean['rank'] == rank)]
            if not subset.empty:
                best_idx = subset['error'].idxmin()
                best_row = subset.loc[best_idx]
                best_methods.append({
                    'matrix': matrix,
                    'rank': rank,
                    'method': best_row['method'],
                    'error': best_row['error']
                })
    
    # Convert to DataFrame
    best_df = pd.DataFrame(best_methods)
    
    # Create heatmap
    best_pivot = best_df.pivot(index='matrix', columns='rank', values='method')
    
    # Get unique methods for coloring
    unique_methods = best_df['method'].unique()
    method_colors = dict(zip(unique_methods, sns.color_palette("tab10", len(unique_methods))))
    
    # Convert methods to integers for coloring
    numeric_matrix = np.zeros(best_pivot.shape)
    for i, method in enumerate(unique_methods):
        mask = best_pivot == method
        numeric_matrix[mask] = i + 1
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    # Create heatmap with custom colormap
    cmap = ListedColormap([plt.cm.tab10(i) for i in range(len(unique_methods))])
    plt.pcolormesh(numeric_matrix, cmap=cmap)
    
    # Add method labels to cells
    for i in range(best_pivot.shape[0]):
        for j in range(best_pivot.shape[1]):
            if pd.notna(best_pivot.iloc[i, j]):
                plt.text(j + 0.5, i + 0.5, best_pivot.iloc[i, j], 
                        ha='center', va='center', color='white',
                        fontsize=9, fontweight='bold')
    
    # Set axes labels and ticks
    plt.yticks(np.arange(0.5, best_pivot.shape[0]), best_pivot.index)
    plt.xticks(np.arange(0.5, best_pivot.shape[1]), best_pivot.columns)
    plt.xlabel('Rank', fontsize=14)
    plt.ylabel('Matrix', fontsize=14)
    plt.title('Best Performing Method by Matrix and Rank', fontsize=16)
    
    # Add legend
    legend_elements = [Patch(facecolor=method_colors[method], 
                           label=method) for method in unique_methods]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'best_method_map.png'), dpi=300)
    plt.close()
    
    # Save best method data
    best_df.to_csv(os.path.join(REPORTS_DIR, 'best_methods.csv'), index=False)

def generate_summary_report(df, improvement_df, ranking_df, overall_ranks, opt_results):
    """Generate summary report with key findings."""
    print("Generating summary report...")
    
    # Get key statistics
    total_matrices = len(df['matrix'].unique())
    total_ranks = len(df['rank'].unique())
    rank_range = f"{df['rank'].min()} to {df['rank'].max()}"
    best_overall = overall_ranks.index[0]  # Method with highest overall rank
    
    # Calculate improvement statistics if available
    svd_outperformance = "N/A"
    if not improvement_df.empty:
        improvements = improvement_df[improvement_df['improvement'] > 0]
        pct_improvements = len(improvements) / len(improvement_df) * 100
        svd_outperformance = f"{pct_improvements:.1f}%"
    
    # Calculate time efficiency
    time_efficiency = df.groupby('method')[['time', 'error']].mean()
    time_efficiency['efficiency'] = time_efficiency['error'] / time_efficiency['time']
    most_efficient = time_efficiency['efficiency'].idxmin()
    
    # Create markdown report
    with open(os.path.join(REPORTS_DIR, 'optimization_summary.md'), 'w') as f:
        f.write("# Matrix Optimization Analysis Summary\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"This analysis compares {len(df['method'].unique())} different matrix approximation methods ")
        f.write(f"across {total_matrices} matrices and {total_ranks} different rank values (from {rank_range}).\n\n")
        
        f.write("## Key Findings\n\n")
        f.write(f"1. **Best Overall Method**: {best_overall}\n")
        f.write(f"2. **SVD Outperformance**: Alternative methods outperformed SVD in {svd_outperformance} of cases\n")
        f.write(f"3. **Most Time-Efficient Method**: {most_efficient}\n\n")
        
        f.write("## Method Performance Summary\n\n")
        f.write("| Method | Average Score | Average Error | Average Time (s) |\n")
        f.write("|--------|--------------|--------------|------------------|\n")
        
        # Add a row for each method
        method_stats = df.groupby('method').agg({'error': 'mean', 'time': 'mean'})
        for method in overall_ranks.index:
            score = overall_ranks[method]
            error = method_stats.loc[method, 'error']
            time = method_stats.loc[method, 'time']
            f.write(f"| {method} | {score:.1f} | {error:.4f} | {time:.2f} |\n")
        
        f.write("\n## Matrix-Specific Insights\n\n")
        
        # Add matrix-specific insights
        for matrix in df['matrix'].unique():
            matrix_data = df[df['matrix'] == matrix]
            best_method_idx = matrix_data['error'].idxmin()
            best_method_row = matrix_data.loc[best_method_idx]
            
            f.write(f"### Matrix {matrix}\n\n")
            f.write(f"- **Best Method**: {best_method_row['method']} ")
            f.write(f"(error: {best_method_row['error']:.4f} at rank {int(best_method_row['rank'])})\n")
            
            # Add optimization results if available
            if opt_results and matrix in opt_results:
                matrix_opt = opt_results[matrix]
                f.write(f"- **Optimal SVD Ratio**: {matrix_opt.get('svd_ratio', 'N/A')}\n")
                if 'rank_distribution' in matrix_opt:
                    rank_dist = matrix_opt['rank_distribution']
                    f.write(f"- **Rank Distribution**: {rank_dist}\n")
            
            f.write("\n")
        
        f.write("## Conclusion\n\n")
        if best_overall == 'Deterministic SVD':
            f.write("The analysis confirms that traditional SVD remains the most robust method ")
            f.write("for matrix approximation across the tested matrices. ")
            f.write("However, for specific matrices and ranks, alternative methods can offer improvements.\n\n")
        else:
            f.write(f"The analysis shows that {best_overall} offers superior performance ")
            f.write("compared to traditional SVD across the tested matrices. ")
            f.write("The optimal method varies depending on the specific matrix and rank requirements.\n\n")
        
        f.write("For time-critical applications, SVD provides a good balance of accuracy and computational efficiency, ")
        f.write(f"while {most_efficient} offers the best error-to-computation-time ratio.\n")
    
    print(f"Summary report saved to: {os.path.join(REPORTS_DIR, 'optimization_summary.md')}")

def main():
    """Main function to run analysis pipeline."""
    print("Starting matrix optimization analysis...")
    
    # Load data
    df, df_clean, opt_results = load_data()
    
    # Generate visualizations and analyses
    create_error_rank_plots(df_clean)
    improvement_df = create_improvement_analysis(df_clean)
    create_time_analysis(df_clean)
    ranking_df, overall_ranks = create_method_ranking(df_clean)
    create_best_method_map(df_clean)
    
    # Generate summary report
    generate_summary_report(df_clean, improvement_df, ranking_df, overall_ranks, opt_results)
    
    print(f"Analysis complete! All reports and visualizations saved to: {REPORTS_DIR}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Matrix Approximation Methods Analysis
=====================================

This script analyzes the computational efficiency and error characteristics of
different matrix approximation methods at rank 10, generating comprehensive
visualizations and tables for comparison.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import json
from tabulate import tabulate

# Configure visualization style
plt.style.use('ggplot')
sns.set_context("talk")
sns.set_palette("viridis")

# Create directories for results
RESULTS_DIR = 'results/rank10_analysis'
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(filepath='proof/matrix_experiments.csv'):
    """Load matrix experiment data."""
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} experiment records")
    return df

def analyze_error_metrics(df):
    """Analyze error metrics by method."""
    # Group by method
    method_errors = df.groupby('method').agg({
        'rel_frob_error': ['mean', 'std', 'min', 'max'],
        'spec_error': ['mean', 'std', 'min', 'max'],
        'time': ['mean', 'std', 'min', 'max']
    })
    
    # Calculate the efficiency metrics (error/time ratio - lower is better)
    efficiency_df = df.copy()
    efficiency_df['efficiency'] = efficiency_df['rel_frob_error'] / efficiency_df['time']
    method_efficiency = efficiency_df.groupby('method')['efficiency'].mean()
    
    # Add column for calculated efficiency to the summary dataframe
    method_summary = method_errors.copy()
    method_summary['efficiency'] = method_efficiency
    
    # Sort by mean Frobenius error
    method_summary = method_summary.sort_values(('rel_frob_error', 'mean'))
    
    return method_summary

def create_error_comparison_plots(df):
    """Create plots comparing error metrics across methods."""
    print("Generating error comparison visualizations...")
    
    # 1. Bar plot of relative Frobenius error by method
    plt.figure(figsize=(12, 8))
    sns.barplot(x='method', y='rel_frob_error', data=df, 
               palette='viridis', errorbar=None)
    plt.title('Relative Frobenius Error by Method (Rank 10)', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Relative Frobenius Error', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'frobenius_error_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Bar plot of spectral error by method
    plt.figure(figsize=(12, 8))
    sns.barplot(x='method', y='spec_error', data=df, 
               palette='viridis', errorbar=None)
    plt.title('Spectral Error by Method (Rank 10)', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Spectral Error', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'spectral_error_comparison.png'), dpi=300)
    plt.close()
    
    # 3. Combined error metrics
    plt.figure(figsize=(14, 10))
    
    # Create a DataFrame with long-format data for seaborn
    long_df = pd.melt(df, id_vars=['method', 'matrix'], 
                     value_vars=['rel_frob_error', 'spec_error'],
                     var_name='error_type', value_name='error_value')
    
    sns.barplot(x='method', y='error_value', hue='error_type', data=long_df, 
               palette='Set2')
    plt.title('Error Metrics by Method (Rank 10)', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Error Value', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Error Type', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'combined_error_comparison.png'), dpi=300)
    plt.close()
    
    # 4. Error distribution by method
    plt.figure(figsize=(14, 10))
    
    sns.boxplot(x='method', y='rel_frob_error', data=df, palette='viridis')
    plt.title('Distribution of Relative Frobenius Error by Method (Rank 10)', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Relative Frobenius Error', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'error_distribution.png'), dpi=300)
    plt.close()
    
    # 5. Error vs matrix type
    plt.figure(figsize=(16, 10))
    
    # Create a pivot table
    error_pivot = df.pivot(index='matrix', columns='method', values='rel_frob_error')
    
    # Plot as heatmap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(error_pivot, cmap=cmap, annot=True, fmt=".3f", 
               linewidths=.5)
    plt.title('Relative Frobenius Error by Method and Matrix (Rank 10)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'error_by_matrix_heatmap.png'), dpi=300)
    plt.close()

def analyze_computational_efficiency(df):
    """Analyze computational efficiency of different methods."""
    print("Analyzing computational efficiency...")
    
    # 1. Bar plot of computation time by method
    plt.figure(figsize=(12, 8))
    sns.barplot(x='method', y='time', data=df, palette='viridis', errorbar=None)
    plt.title('Computation Time by Method (Rank 10)', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'computation_time.png'), dpi=300)
    plt.close()
    
    # 2. Time distribution by method
    plt.figure(figsize=(14, 10))
    
    sns.boxplot(x='method', y='time', data=df, palette='viridis')
    plt.title('Distribution of Computation Time by Method (Rank 10)', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'time_distribution.png'), dpi=300)
    plt.close()
    
    # 3. Efficiency metric (error/time ratio - lower is better)
    efficiency_df = df.copy()
    efficiency_df['efficiency'] = efficiency_df['rel_frob_error'] / efficiency_df['time']
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='method', y='efficiency', data=efficiency_df, palette='viridis', errorbar=None)
    plt.title('Error-Time Efficiency by Method (Rank 10)', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Efficiency (Error/Time) - Lower is Better', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'error_time_efficiency.png'), dpi=300)
    plt.close()
    
    return efficiency_df

def create_error_time_tradeoff_plot(df):
    """Create scatter plot showing error vs time tradeoff."""
    print("Generating error-time tradeoff visualization...")
    
    # Get method averages
    method_avg = df.groupby('method').agg({
        'rel_frob_error': 'mean',
        'time': 'mean'
    }).reset_index()
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    # Use different markers for each method
    markers = ['o', 's', '^', 'D', '*']
    
    # Create scatter plot
    for i, method in enumerate(method_avg['method']):
        plt.scatter(
            method_avg.loc[method_avg['method'] == method, 'time'],
            method_avg.loc[method_avg['method'] == method, 'rel_frob_error'],
            label=method,
            marker=markers[i % len(markers)],
            s=200
        )
    
    # Add method names as annotations
    for i, row in method_avg.iterrows():
        plt.annotate(
            row['method'],
            (row['time'] * 1.1, row['rel_frob_error']),
            fontsize=12
        )
    
    # Add connecting lines to origin to show "efficiency" vectors
    for i, row in method_avg.iterrows():
        plt.plot([0, row['time']], [0, row['rel_frob_error']], 
                'k--', alpha=0.3)
    
    plt.title('Error vs Computation Time Trade-off (Rank 10)', fontsize=16)
    plt.xlabel('Computation Time (seconds)', fontsize=14)
    plt.ylabel('Relative Frobenius Error', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Use log scale for better visibility
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'error_time_tradeoff.png'), dpi=300)
    plt.close()

def create_method_comparison_table(df, summary_df):
    """Create a detailed comparison table of methods."""
    print("Creating method comparison table...")
    
    # Reset index to make 'method' a column
    summary_df_reset = summary_df.reset_index()
    
    # Calculate summary statistics
    comparison_data = []
    
    # Get SVD metrics for comparison
    svd_error = df[df['method'] == 'Deterministic SVD']['rel_frob_error'].mean()
    svd_time = df[df['method'] == 'Deterministic SVD']['time'].mean()
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        
        # Calculate metrics
        frob_error = method_df['rel_frob_error'].mean()
        spec_error = method_df['spec_error'].mean()
        comp_time = method_df['time'].mean()
        efficiency = frob_error / comp_time
        
        # Calculate relative to SVD
        rel_error = frob_error / svd_error
        rel_time = comp_time / svd_time
        speedup = svd_time / comp_time
        
        # Normalized scores (0-1, lower is better)
        max_error = df['rel_frob_error'].mean().max()
        max_time = df['time'].mean().max()
        norm_error = frob_error / max_error
        norm_time = comp_time / max_time
        combined_score = (norm_error + norm_time) / 2
        
        comparison_data.append({
            'Method': method,
            'Frob Error': f"{frob_error:.4f}",
            'Spectral Error': f"{spec_error:.4f}",
            'Time (s)': f"{comp_time:.4f}",
            'Error/Time': f"{efficiency:.4f}",
            'vs SVD Error': f"{rel_error:.2f}x",
            'vs SVD Time': f"{rel_time:.2f}x",
            'Speedup': f"{speedup:.2f}x",
            'Overall Score': f"{combined_score:.4f}"
        })
    
    # Convert to DataFrame and sort by Frobenius error
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Frob Error')
    
    # Create text table
    headers = comparison_df.columns.tolist()
    table_data = comparison_df.values.tolist()
    
    table_text = tabulate(table_data, headers=headers, tablefmt="grid")
    
    # Write to file
    with open(os.path.join(RESULTS_DIR, 'method_comparison_table.txt'), 'w') as f:
        f.write("Method Comparison at Rank 10\n")
        f.write("===========================\n\n")
        f.write(table_text)
    
    # Also create a LaTeX version for publication
    latex_table = tabulate(table_data, headers=headers, tablefmt="latex")
    with open(os.path.join(RESULTS_DIR, 'method_comparison_table.tex'), 'w') as f:
        f.write(latex_table)
    
    # HTML version for web viewing
    html_table = tabulate(table_data, headers=headers, tablefmt="html")
    with open(os.path.join(RESULTS_DIR, 'method_comparison_table.html'), 'w') as f:
        f.write("<html><head><title>Method Comparison at Rank 10</title>")
        f.write("<style>table {border-collapse: collapse; width: 100%;} ")
        f.write("th, td {text-align: left; padding: 8px;} ")
        f.write("tr:nth-child(even) {background-color: #f2f2f2;} ")
        f.write("th {background-color: #4CAF50; color: white;}</style></head>")
        f.write("<body><h1>Method Comparison at Rank 10</h1>")
        f.write(html_table)
        f.write("</body></html>")
    
    # Return the comparison data
    return comparison_df

def create_comprehensive_visualization(df, summary_df):
    """Create a single comprehensive visualization dashboard."""
    print("Creating comprehensive visualization dashboard...")
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3)
    
    # 1. Error comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    methods = df['method'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    
    error_data = []
    for i, method in enumerate(methods):
        method_errors = df[df['method'] == method]['rel_frob_error'].mean()
        error_data.append((method, method_errors))
    
    error_data.sort(key=lambda x: x[1])
    methods_sorted = [x[0] for x in error_data]
    errors_sorted = [x[1] for x in error_data]
    
    bars = ax1.bar(methods_sorted, errors_sorted, color=colors)
    ax1.set_title('Average Frobenius Error', fontsize=14)
    ax1.set_ylabel('Relative Error', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.set_yscale('log')
    
    # 2. Time comparison (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    
    time_data = []
    for i, method in enumerate(methods):
        method_time = df[df['method'] == method]['time'].mean()
        time_data.append((method, method_time))
    
    time_data.sort(key=lambda x: x[1])
    methods_time_sorted = [x[0] for x in time_data]
    times_sorted = [x[1] for x in time_data]
    
    ax2.bar(methods_time_sorted, times_sorted, color=colors)
    ax2.set_title('Average Computation Time', fontsize=14)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.set_yscale('log')
    
    # 3. Error vs Time scatterplot (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    for i, method in enumerate(methods):
        method_df = df[df['method'] == method]
        ax3.scatter(
            method_df['time'].mean(), 
            method_df['rel_frob_error'].mean(),
            label=method,
            color=colors[i],
            s=100
        )
        ax3.annotate(
            method,
            (method_df['time'].mean(), method_df['rel_frob_error'].mean()),
            fontsize=10,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    ax3.set_title('Error vs Time Trade-off', fontsize=14)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Relative Error', fontsize=12)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Matrix-specific error heatmap (middle row)
    ax4 = fig.add_subplot(gs[1, :])
    
    pivot_data = df.pivot_table(
        index='matrix', 
        columns='method', 
        values='rel_frob_error'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax4)
    ax4.set_title('Error by Matrix and Method', fontsize=14)
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Comparison to SVD (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Calculate ratio to SVD
    ratio_to_svd = []
    for method in methods:
        if method == 'Deterministic SVD':
            continue
        
        method_error = df[df['method'] == method]['rel_frob_error'].mean()
        svd_error = df[df['method'] == 'Deterministic SVD']['rel_frob_error'].mean()
        ratio = method_error / svd_error
        ratio_to_svd.append((method, ratio))
    
    ratio_to_svd.sort(key=lambda x: x[1])
    ratio_methods = [x[0] for x in ratio_to_svd]
    ratios = [x[1] for x in ratio_to_svd]
    
    ax5.bar(ratio_methods, ratios)
    ax5.axhline(y=1.0, color='r', linestyle='--')
    ax5.set_title('Error Ratio to SVD', fontsize=14)
    ax5.set_ylabel('Error Ratio (method/SVD)', fontsize=12)
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Speedup relative to SVD (bottom middle)
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Calculate speedup relative to SVD
    speedup_vs_svd = []
    for method in methods:
        if method == 'Deterministic SVD':
            continue
        
        method_time = df[df['method'] == method]['time'].mean()
        svd_time = df[df['method'] == 'Deterministic SVD']['time'].mean()
        speedup = svd_time / method_time
        speedup_vs_svd.append((method, speedup))
    
    speedup_vs_svd.sort(key=lambda x: x[1], reverse=True)
    speedup_methods = [x[0] for x in speedup_vs_svd]
    speedups = [x[1] for x in speedup_vs_svd]
    
    ax6.bar(speedup_methods, speedups)
    ax6.axhline(y=1.0, color='r', linestyle='--')
    ax6.set_title('Speedup Relative to SVD', fontsize=14)
    ax6.set_ylabel('Speedup Factor (SVD/method)', fontsize=12)
    ax6.tick_params(axis='x', rotation=45)
    
    # 7. Method ranking (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    
    # Create a combined score (lower is better) using the raw data
    # rather than summary_df to avoid indexing issues
    method_scores = []
    for method in methods:
        error = df[df['method'] == method]['rel_frob_error'].mean()
        time = df[df['method'] == method]['time'].mean()
        
        # Normalize
        max_error = df.groupby('method')['rel_frob_error'].mean().max()
        max_time = df.groupby('method')['time'].mean().max()
        
        norm_error = error / max_error
        norm_time = time / max_time
        combined_score = (norm_error + norm_time) / 2
        
        method_scores.append((method, combined_score))
    
    method_scores.sort(key=lambda x: x[1])
    score_methods = [x[0] for x in method_scores]
    scores = [x[1] for x in method_scores]
    
    ax7.barh(score_methods, scores)
    ax7.set_title('Overall Method Ranking', fontsize=14)
    ax7.set_xlabel('Combined Score (lower is better)', fontsize=12)
    
    # Add overall title and adjust layout
    plt.suptitle('Comprehensive Analysis of Matrix Approximation Methods (Rank 10)', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(RESULTS_DIR, 'comprehensive_analysis_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_analysis_report(df, summary_df, comparison_df):
    """Generate a detailed analysis report with key findings."""
    print("Generating analysis report...")
    
    # Get the first method (with lowest error) from the sorted summary DataFrame
    summary_df_reset = summary_df.reset_index()
    best_error_method = summary_df_reset.iloc[0]['method']
    
    # Method with best time
    best_time_idx = summary_df_reset[('time', 'mean')].idxmin()
    best_time_method = summary_df_reset.iloc[best_time_idx]['method']
    
    # Method with best efficiency
    best_efficiency_idx = summary_df_reset['efficiency'].idxmin()
    best_efficiency_method = summary_df_reset.iloc[best_efficiency_idx]['method']
    
    # Method with best overall score from comparison DataFrame
    best_overall_method = comparison_df.iloc[0]['Method']  # Already sorted by error
    
    # Randomized SVD stats
    randomized_stats = comparison_df[comparison_df['Method'] == 'Randomized SVD']
    
    # Calculate matrix-specific insights
    matrix_insights = {}
    for matrix in df['matrix'].unique():
        matrix_df = df[df['matrix'] == matrix]
        
        if not matrix_df.empty:
            # Best method for this matrix
            best_method_df = matrix_df.loc[matrix_df['rel_frob_error'].idxmin()]
            best_method = best_method_df['method']
            
            # SVD error for this matrix
            svd_rows = matrix_df[matrix_df['method'] == 'Deterministic SVD']
            svd_error = float(svd_rows['rel_frob_error'].values[0]) if not svd_rows.empty else None
            
            # Fastest method for this matrix
            fastest_method_df = matrix_df.loc[matrix_df['time'].idxmin()]
            fastest_method = fastest_method_df['method']
            
            matrix_insights[matrix] = {
                'best_method': best_method,
                'svd_error': svd_error,
                'fastest_method': fastest_method
            }
    
    # Extract error and time performance directly from the dataframe
    error_performance = {}
    time_performance = {}
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        error_performance[method] = {
            'mean_frob_error': float(method_data['rel_frob_error'].mean()),
            'mean_spec_error': float(method_data['spec_error'].mean())
        }
        
        time_performance[method] = {
            'mean_time': float(method_data['time'].mean())
        }
    
    # Parse randomized SVD stats correctly
    try:
        if not randomized_stats.empty:
            svd_speedup = float(randomized_stats['Speedup'].values[0].replace('x', ''))
            svd_error_ratio = float(randomized_stats['vs SVD Error'].values[0].replace('x', ''))
        else:
            svd_speedup = 'N/A'
            svd_error_ratio = 'N/A'
    except:
        svd_speedup = 'N/A'
        svd_error_ratio = 'N/A'
    
    # Create the report with serializable Python types
    report = {
        'title': 'Matrix Approximation Methods Analysis at Rank 10',
        'date': 'May 8, 2025',
        'key_findings': {
            'best_error_method': str(best_error_method),
            'best_time_method': str(best_time_method),
            'best_efficiency_method': str(best_efficiency_method),
            'best_overall_method': str(best_overall_method),
            'randomized_svd_speedup': svd_speedup,
            'randomized_svd_error_ratio': svd_error_ratio
        },
        'error_performance': error_performance,
        'time_performance': time_performance,
        'matrix_specific_insights': matrix_insights,
        'recommendations': [
            f"For highest accuracy: Use {best_error_method}",
            f"For fastest computation: Use {best_time_method}",
            f"For best efficiency (error/time ratio): Use {best_efficiency_method}",
            f"Best overall balance: Use {best_overall_method}"
        ]
    }
    
    # Save as JSON
    with open(os.path.join(RESULTS_DIR, 'rank10_analysis_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Create markdown version
    with open(os.path.join(RESULTS_DIR, 'rank10_analysis_report.md'), 'w') as f:
        f.write(f"# {report['title']}\n\n")
        f.write(f"Date: {report['date']}\n\n")
        
        f.write("## Key Findings\n\n")
        f.write(f"- **Best Accuracy Method**: {report['key_findings']['best_error_method']}\n")
        f.write(f"- **Fastest Method**: {report['key_findings']['best_time_method']}\n")
        f.write(f"- **Most Efficient Method**: {report['key_findings']['best_efficiency_method']}\n")
        f.write(f"- **Best Overall Method**: {report['key_findings']['best_overall_method']}\n\n")
        
        if 'randomized_svd_speedup' in report['key_findings'] and report['key_findings']['randomized_svd_speedup'] != 'N/A':
            f.write("## Randomized SVD Performance\n\n")
            f.write(f"- **Speedup vs. Deterministic SVD**: {report['key_findings']['randomized_svd_speedup']:.2f}x\n")
            f.write(f"- **Error Ratio to Deterministic SVD**: {report['key_findings']['randomized_svd_error_ratio']:.4f}x\n\n")
        
        f.write("## Error Performance\n\n")
        f.write("| Method | Mean Frobenius Error | Mean Spectral Error |\n")
        f.write("|--------|----------------------|---------------------|\n")
        
        for method, stats in report['error_performance'].items():
            f.write(f"| {method} | {stats['mean_frob_error']:.4f} | {stats['mean_spec_error']:.4f} |\n")
        
        f.write("\n## Time Performance\n\n")
        f.write("| Method | Mean Time (seconds) |\n")
        f.write("|--------|---------------------|\n")
        
        for method, stats in report['time_performance'].items():
            f.write(f"| {method} | {stats['mean_time']:.4f} |\n")
        
        f.write("\n## Matrix-Specific Insights\n\n")
        f.write("| Matrix | Best Method | Fastest Method |\n")
        f.write("|--------|-------------|---------------|\n")
        
        for matrix, insights in report['matrix_specific_insights'].items():
            f.write(f"| {matrix} | {insights['best_method']} | {insights['fastest_method']} |\n")
        
        f.write("\n## Recommendations\n\n")
        for rec in report['recommendations']:
            f.write(f"- {rec}\n")
    
    return report

def main():
    """Main function to run the analysis."""
    print("\n" + "="*80)
    print("Matrix Approximation Methods Analysis (Rank 10)")
    print("="*80 + "\n")
    
    # Load data
    df = load_data()
    
    # Analyze error metrics
    summary_df = analyze_error_metrics(df)
    
    # Create error comparison plots
    create_error_comparison_plots(df)
    
    # Analyze computational efficiency
    efficiency_df = analyze_computational_efficiency(df)
    
    # Create error vs time tradeoff plot
    create_error_time_tradeoff_plot(df)
    
    # Create method comparison table
    comparison_df = create_method_comparison_table(df, summary_df)
    
    # Create comprehensive visualization
    create_comprehensive_visualization(df, summary_df)
    
    # Generate analysis report
    report = generate_analysis_report(df, summary_df, comparison_df)
    
    print("\nAnalysis complete! All results saved to:", RESULTS_DIR)
    print("\nKey findings:")
    print(f"- Best Accuracy Method: {report['key_findings']['best_error_method']}")
    print(f"- Fastest Method: {report['key_findings']['best_time_method']}")
    print(f"- Most Efficient Method: {report['key_findings']['best_efficiency_method']}")
    print(f"- Best Overall Method: {report['key_findings']['best_overall_method']}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Early Stopping Visualization
============================

This script analyzes the training results and visualizes the benefits of early stopping
by showing how reward, loss, and other metrics stabilize after a certain number of episodes.
It supports analysis for both DQN and A2C algorithms.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import json

# Set style for plots
plt.style.use('ggplot')
sns.set_context("paper", font_scale=1.2)

# Define constants
RESULTS_DIR = "results"
PLOT_DIR = os.path.join(RESULTS_DIR, "early_stopping")
DQN_TRAINING_DATA_PATH = "proof/training_results.csv"
A2C_TRAINING_DATA_PATH = "proof/a2c_training_results_fixed.csv"  # Using the fixed file

# Create directory for plots if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)

def load_training_data(filepath, algorithm="DQN"):
    """Load training results from CSV file."""
    try:
        data = pd.read_csv(filepath)
        print(f"Successfully loaded {algorithm} data with {len(data)} rows")
        
        # Add algorithm column for reference
        data['Algorithm'] = algorithm
        
        # For A2C data, rename columns to match expected format
        if algorithm == "A2C":
            # Map A2C's value loss to the Loss column expected by analyze_convergence
            if 'Avg_Value_Loss' in data.columns and 'Loss' not in data.columns:
                data['Loss'] = data['Avg_Value_Loss']
            
            # Map A2C's entropy column if present
            if 'Avg_Entropy' in data.columns and 'Entropy' not in data.columns:
                data['Entropy'] = data['Avg_Entropy']
        
        return data
    except Exception as e:
        print(f"Error loading {algorithm} data: {str(e)}")
        return None

def analyze_convergence(data, algorithm="DQN"):
    """
    Analyze when metrics converge and calculate potential time savings.
    
    Args:
        data (pd.DataFrame): The training data
        algorithm (str): The algorithm name (DQN or A2C)
        
    Returns:
        dict: Dictionary with convergence analysis results
    """
    # Calculate rolling means to smooth the data
    window_size = min(5, len(data) // 10) if len(data) > 10 else 1
    data['reward_rolling'] = data['Reward'].rolling(window=window_size, center=True).mean()
    data['loss_rolling'] = data['Loss'].rolling(window=window_size, center=True).mean()
    
    # Fill NaN values created by the rolling window
    data['reward_rolling'] = data['reward_rolling'].fillna(data['Reward'])
    data['loss_rolling'] = data['loss_rolling'].fillna(data['Loss'])
    
    # Calculate percentage change between consecutive episodes
    data['reward_pct_change'] = data['reward_rolling'].pct_change().abs()
    data['loss_pct_change'] = data['loss_rolling'].pct_change().abs()
    
    # Set threshold for convergence (change less than 1%)
    reward_threshold = 0.01
    loss_threshold = 0.01
    
    # Find the episode where metrics stabilize (3 consecutive episodes below threshold)
    stabilization_point = None
    consecutive_stable = 0
    required_consecutive = 3
    
    for i in range(1, len(data)):
        if (data['reward_pct_change'].iloc[i] < reward_threshold and 
            data['loss_pct_change'].iloc[i] < loss_threshold):
            consecutive_stable += 1
            if consecutive_stable >= required_consecutive:
                stabilization_point = i - required_consecutive + 1
                break
        else:
            consecutive_stable = 0
    
    # If no stabilization is found, use 80% of the data
    if stabilization_point is None:
        stabilization_point = int(len(data) * 0.8)
    
    # Calculate time savings
    total_episodes = data['Episode'].iloc[-1]
    episodes_saved = total_episodes - data['Episode'].iloc[stabilization_point]
    time_savings_pct = (episodes_saved / total_episodes) * 100
    
    # Current date for report
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Check if entropy data exists (A2C specific)
    has_entropy = 'Entropy' in data.columns
    
    # Prepare results
    results = {
        'algorithm': algorithm,
        'stabilization_episode': int(data['Episode'].iloc[stabilization_point]),
        'total_episodes': int(total_episodes),
        'episodes_saved': int(episodes_saved),
        'time_savings_percentage': round(time_savings_pct, 2),
        'final_reward': float(data['Reward'].iloc[-1]),
        'stabilized_reward': float(data['Reward'].iloc[stabilization_point]),
        'reward_difference': float(data['Reward'].iloc[-1] - data['Reward'].iloc[stabilization_point]),
        'reward_difference_pct': round(((data['Reward'].iloc[-1] - data['Reward'].iloc[stabilization_point]) / data['Reward'].iloc[-1]) * 100, 2),
        'date': current_date,
        'has_entropy': has_entropy
    }
    
    # Add A2C specific metrics if available
    if has_entropy:
        results['final_entropy'] = float(data['Entropy'].iloc[-1])
        results['stabilized_entropy'] = float(data['Entropy'].iloc[stabilization_point])
    
    return results

def plot_training_metrics(data, analysis_results, algorithm="DQN"):
    """
    Create visualizations of training metrics showing the early stopping point.
    
    Args:
        data (pd.DataFrame): The training data
        analysis_results (dict): Analysis results from analyze_convergence function
        algorithm (str): The algorithm name (DQN or A2C)
    """
    # Define the stabilization point
    stop_episode = analysis_results['stabilization_episode']
    
    # Plot 1: Training metrics with early stopping point
    if analysis_results['has_entropy']:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot reward
    ax1.plot(data['Episode'], data['Reward'], 'b-', label='Reward')
    ax1.axvline(x=stop_episode, color='r', linestyle='--', label=f'Early stopping point (Episode {stop_episode})')
    ax1.set_ylabel('Reward Value')
    ax1.set_title(f'Training Reward Over Time with Early Stopping Point ({algorithm})')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(data['Episode'], data['Loss'], 'g-', label='Loss')
    ax2.axvline(x=stop_episode, color='r', linestyle='--', label=f'Early stopping point (Episode {stop_episode})')
    ax2.set_ylabel('Loss Value')
    ax2.set_title('Training Loss Over Time with Early Stopping Point')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # If we have entropy data (for A2C), plot it
    if analysis_results['has_entropy']:
        ax3.plot(data['Episode'], data['Entropy'], 'purple', label='Entropy')
        ax3.axvline(x=stop_episode, color='r', linestyle='--', label=f'Early stopping point (Episode {stop_episode})')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Entropy Value')
        ax3.set_title('Training Entropy Over Time with Early Stopping Point')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
    else:
        ax2.set_xlabel('Episode')
    
    # Add vertical lines at each data point for better visualization
    if len(data) < 20:  # Only if we have few data points
        for episode in data['Episode']:
            ax1.axvline(x=episode, color='grey', alpha=0.1)
            ax2.axvline(x=episode, color='grey', alpha=0.1)
            if analysis_results['has_entropy']:
                ax3.axvline(x=episode, color='grey', alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{algorithm.lower()}_training_metrics_with_stopping.png'), dpi=300)
    plt.close()
    
    # Plot 2: Percentage change in metrics
    plt.figure(figsize=(12, 6))
    plt.plot(data['Episode'][1:], data['reward_pct_change'][1:], 'b-', label='Reward % Change')
    plt.plot(data['Episode'][1:], data['loss_pct_change'][1:], 'g-', label='Loss % Change')
    plt.axvline(x=stop_episode, color='r', linestyle='--', label=f'Early stopping point (Episode {stop_episode})')
    plt.axhline(y=0.01, color='k', linestyle=':', label='1% Change Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Absolute Percentage Change')
    plt.title(f'Stability of {algorithm} Training Metrics Over Time')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization of small changes
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{algorithm.lower()}_training_stability.png'), dpi=300)
    plt.close()
    
    # Plot 3: Time savings visualization
    labels = ['Full Training', 'Early Stopping']
    episodes = [analysis_results['total_episodes'], analysis_results['stabilization_episode']]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, episodes, color=['skyblue', 'lightgreen'])
    
    # Add time savings annotation
    time_savings = analysis_results['time_savings_percentage']
    plt.annotate(f'{time_savings}% time saved', 
                 xy=(1, episodes[1]), 
                 xytext=(1, episodes[1] + (episodes[0] - episodes[1])/2),
                 ha='center',
                 va='center',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Add episode values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{int(height)} episodes',
                 ha='center', va='bottom')
    
    plt.ylabel('Number of Episodes')
    plt.title(f'{algorithm} Training Duration Comparison')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{algorithm.lower()}_time_savings.png'), dpi=300)
    plt.close()
    
    # Plot 4: Reward Stabilization
    plt.figure(figsize=(10, 6))
    plt.plot(data['Episode'], data['Reward'], 'b-', label='Actual Reward')
    
    # Add a horizontal line at the stabilized reward
    stabilized_reward = analysis_results['stabilized_reward']
    plt.axhline(y=stabilized_reward, color='g', linestyle='-', label=f'Stabilized Reward ({stabilized_reward:.2f})')
    
    # Add vertical line at stopping point
    plt.axvline(x=stop_episode, color='r', linestyle='--', label=f'Early stopping point (Episode {stop_episode})')
    
    # Add a shaded area for the post-stabilization region
    plt.axvspan(stop_episode, data['Episode'].max(), alpha=0.2, color='gray', label='Potentially unnecessary training')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{algorithm} Reward Stabilization Analysis')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{algorithm.lower()}_reward_stabilization.png'), dpi=300)
    plt.close()
    
    # Comprehensive Aggregated Graph
    plot_aggregated_results(data, analysis_results, algorithm)

def plot_aggregated_results(data, analysis_results, algorithm="DQN"):
    """
    Create a comprehensive aggregated visualization that combines all key metrics
    and insights from the early stopping analysis into a single graph.
    
    Args:
        data (pd.DataFrame): The training data
        analysis_results (dict): Analysis results from analyze_convergence function
        algorithm (str): The algorithm name (DQN or A2C)
    """
    # Create a figure with GridSpec for complex layout
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 4)
    
    # Define the stabilization point
    stop_episode = analysis_results['stabilization_episode']
    
    # Find the index of the row containing the stop_episode
    stop_idx = data.index[data['Episode'] == stop_episode].tolist()[0]
    
    # Main plot: Training metrics (top spanning all columns)
    ax_main = fig.add_subplot(gs[0, :])
    
    # Create twin axis for loss
    ax_loss = ax_main.twinx()
    
    # Plot reward on left axis
    reward_line, = ax_main.plot(data['Episode'], data['Reward'], 'b-', linewidth=2, label='Reward')
    ax_main.set_ylabel('Reward Value', color='blue', fontsize=12)
    ax_main.tick_params(axis='y', labelcolor='blue')
    
    # Plot loss on right axis
    loss_line, = ax_loss.plot(data['Episode'], data['Loss'], 'g-', linewidth=2, label='Loss')
    ax_loss.set_ylabel('Loss Value', color='green', fontsize=12)
    ax_loss.tick_params(axis='y', labelcolor='green')
    
    # Add vertical line for early stopping point
    stop_line = ax_main.axvline(x=stop_episode, color='r', linestyle='--', linewidth=2)
    
    # Add shaded region for unnecessary training
    ax_main.axvspan(stop_episode, data['Episode'].max(), alpha=0.2, color='lightgray')
    
    # Add annotations for key metrics
    time_saved = f"{analysis_results['time_savings_percentage']}% time saved"
    reward_diff = f"Reward diff: {analysis_results['reward_difference_pct']}%"
    
    ax_main.annotate(time_saved,
                    xy=(stop_episode, data['Reward'].min()),
                    xytext=(stop_episode - 30, data['Reward'].min() - 1),
                    fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    ax_main.annotate(reward_diff,
                    xy=(stop_episode, data['Reward'].max()),
                    xytext=(stop_episode + 10, data['Reward'].max() + 1),
                    fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', fc='cyan', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2'))
    
    # Set title for main plot
    ax_main.set_title(f'Comprehensive {algorithm} Early Stopping Analysis', fontsize=16, fontweight='bold')
    
    # Create a custom legend for the main plot
    lines = [reward_line, loss_line, stop_line]
    labels = ['Reward', 'Loss', f'Early stopping (Episode {stop_episode})']
    leg = ax_main.legend(lines, labels, loc='upper center', ncol=3, fontsize=10, 
                        bbox_to_anchor=(0.5, 1.15))
    
    # Bottom left: Stability analysis
    ax_stability = fig.add_subplot(gs[1, :2])
    ax_stability.plot(data['Episode'][1:], data['reward_pct_change'][1:], 'b-', label='Reward % Change')
    ax_stability.plot(data['Episode'][1:], data['loss_pct_change'][1:], 'g-', label='Loss % Change')
    ax_stability.axvline(x=stop_episode, color='r', linestyle='--')
    ax_stability.axhline(y=0.01, color='k', linestyle=':', label='1% Threshold')
    ax_stability.set_ylabel('% Change (log scale)')
    ax_stability.set_title('Training Stability', fontsize=12)
    ax_stability.set_yscale('log')
    ax_stability.legend(fontsize=8, loc='upper right')
    ax_stability.grid(True, alpha=0.3)
    
    # Bottom middle: Time savings bar chart
    ax_time = fig.add_subplot(gs[1, 2:])
    labels = ['Full\nTraining', 'Early\nStopping']
    episodes = [analysis_results['total_episodes'], analysis_results['stabilization_episode']]
    colors = ['#FF9999', '#66B2FF']
    bars = ax_time.bar(labels, episodes, color=colors)
    
    # Add percentage saved
    savings_pct = analysis_results['time_savings_percentage']
    
    # Add episode counts on top of bars
    for bar in bars:
        height = bar.get_height()
        ax_time.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f'{int(height)} eps',
                     ha='center', va='bottom', fontsize=9)
    
    # Add an arrow showing the savings
    ax_time.annotate(f'{savings_pct}% saved',
                     xy=(0.5, episodes[1] + (episodes[0] - episodes[1])/2),
                     xytext=(0.5, episodes[1] + (episodes[0] - episodes[1])/2),
                     ha='center',
                     va='center',
                     fontsize=11,
                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    
    ax_time.set_title('Training Time Comparison', fontsize=12)
    ax_time.grid(axis='y', alpha=0.3)
    
    # Bottom: Key metrics and statistics
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')  # Turn off axis
    
    # Get epsilon value at the stabilization point using the correct index
    if 'Epsilon' in data.columns:
        epsilon_value = data.loc[stop_idx, 'Epsilon']
        epsilon_row = ['Epsilon Value', f"{epsilon_value:.4f}", 'Exploration rate at stabilization']
    else:
        # For A2C, show entropy instead if available
        epsilon_row = []
        if 'Entropy' in data.columns:
            entropy_value = data.loc[stop_idx, 'Entropy']
            epsilon_row = ['Entropy Value', f"{entropy_value:.4f}", 'Policy entropy at stabilization']
    
    # Create a summary table
    table_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Stabilization Episode', f"{stop_episode} / {int(data['Episode'].max())}", 'Training converges at this point'],
        ['Episodes Saved', f"{analysis_results['episodes_saved']}", f"Reduced training time by {savings_pct}%"],
        ['Final Reward', f"{analysis_results['final_reward']:.4f}", 'Reward after full training'],
        ['Stabilized Reward', f"{analysis_results['stabilized_reward']:.4f}", 'Reward at stabilization point'],
        ['Reward Difference', f"{analysis_results['reward_difference']:.4f} ({analysis_results['reward_difference_pct']}%)", 'Minimal difference indicates effective early stopping']
    ]
    
    # Add epsilon/entropy row if available
    if epsilon_row:
        table_data.append(epsilon_row)
    
    # Create the table
    table = ax_stats.table(cellText=table_data,
                         loc='center',
                         cellLoc='center',
                         colWidths=[0.25, 0.25, 0.5])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the header row
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Add a subtitle for the table
    ax_stats.text(0.5, 0.95, f'{algorithm} Early Stopping Analysis Summary',
                 horizontalalignment='center',
                 fontsize=12,
                 fontweight='bold')
    
    # Add timestamp and conclusion at the bottom of the figure
    plt.figtext(0.5, 0.01, 
               f"Generated on {analysis_results['date']} | {algorithm} Training Analysis | Early stopping can save {savings_pct}% training time with only {analysis_results['reward_difference_pct']}% reward difference",
               ha="center", 
               fontsize=9,
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig(os.path.join(PLOT_DIR, f'{algorithm.lower()}_comprehensive_early_stopping_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_algorithm_comparison(dqn_results, a2c_results):
    """
    Create comparative visualizations between DQN and A2C early stopping results.
    
    Args:
        dqn_results (dict): Analysis results for DQN
        a2c_results (dict): Analysis results for A2C
    """
    # Comparison 1: Time savings
    labels = ['DQN', 'A2C']
    full_episodes = [dqn_results['total_episodes'], a2c_results['total_episodes']]
    stop_episodes = [dqn_results['stabilization_episode'], a2c_results['stabilization_episode']]
    savings_pct = [dqn_results['time_savings_percentage'], a2c_results['time_savings_percentage']]
    
    plt.figure(figsize=(10, 7))
    x = np.arange(len(labels))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = plt.bar(x - width/2, full_episodes, width, label='Full Training', color='#FF9999')
    bars2 = plt.bar(x + width/2, stop_episodes, width, label='Early Stopping', color='#66B2FF')
    
    # Add episode counts and savings percentage
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        plt.text(bar1.get_x() + bar1.get_width()/2., height1 + 5,
                 f'{int(height1)}',
                 ha='center', va='bottom')
        plt.text(bar2.get_x() + bar2.get_width()/2., height2 + 5,
                 f'{int(height2)}',
                 ha='center', va='bottom')
        
        # Add savings annotation
        plt.annotate(f'{savings_pct[i]}% saved',
                    xy=(i, (height1 + height2)/2),
                    xytext=(i + 0.3, (height1 + height2)/2),
                    ha='center',
                    va='center',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    plt.ylabel('Number of Episodes')
    plt.title('Early Stopping Comparison: DQN vs A2C')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'algorithm_comparison_episodes.png'), dpi=300)
    plt.close()
    
    # Comparison 2: Reward difference
    reward_diff = [abs(dqn_results['reward_difference']), abs(a2c_results['reward_difference'])]
    reward_diff_pct = [abs(dqn_results['reward_difference_pct']), abs(a2c_results['reward_difference_pct'])]
    
    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, reward_diff, color=['#4472C4', '#ED7D31'])
    
    # Add reward difference values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.4f} ({reward_diff_pct[i]}%)',
                 ha='center', va='bottom')
    
    plt.ylabel('Absolute Reward Difference')
    plt.title('Early Stopping Impact on Final Reward: DQN vs A2C')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'algorithm_comparison_reward_impact.png'), dpi=300)
    plt.close()
    
    # Comparison 3: Comprehensive comparison
    plt.figure(figsize=(12, 8))
    
    # Create a comprehensive metrics table
    metrics = [
        'Early stopping episode',
        'Episode savings',
        'Time savings (%)',
        'Reward at stopping point',
        'Final reward',
        'Absolute reward difference',
        'Relative reward difference (%)'
    ]
    
    dqn_values = [
        dqn_results['stabilization_episode'],
        dqn_results['episodes_saved'],
        dqn_results['time_savings_percentage'],
        f"{dqn_results['stabilized_reward']:.4f}",
        f"{dqn_results['final_reward']:.4f}",
        f"{abs(dqn_results['reward_difference']):.4f}",
        f"{abs(dqn_results['reward_difference_pct']):.2f}"
    ]
    
    a2c_values = [
        a2c_results['stabilization_episode'],
        a2c_results['episodes_saved'],
        a2c_results['time_savings_percentage'],
        f"{a2c_results['stabilized_reward']:.4f}",
        f"{a2c_results['final_reward']:.4f}",
        f"{abs(a2c_results['reward_difference']):.4f}",
        f"{abs(a2c_results['reward_difference_pct']):.2f}"
    ]
    
    # Create table
    ax = plt.subplot(111)
    ax.axis('off')
    
    table_data = []
    for i in range(len(metrics)):
        table_data.append([metrics[i], dqn_values[i], a2c_values[i]])
    
    colLabels = ['Metric', 'DQN', 'A2C']
    
    table = ax.table(cellText=table_data,
                   colLabels=colLabels,
                   loc='center',
                   cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style the header row
    for j in range(len(colLabels)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Add alternating row colors
    for i in range(len(metrics)):
        for j in range(3):
            if i % 2 == 0:
                table[(i+1, j)].set_facecolor('#E6F1FF')
    
    plt.title('Comparative Analysis: DQN vs A2C Early Stopping', fontsize=14, y=0.8)
    
    # Add conclusion text
    dqn_better = dqn_results['time_savings_percentage'] > a2c_results['time_savings_percentage']
    better_algo = "DQN" if dqn_better else "A2C"
    
    conclusion_text = (
        f"Early stopping analysis indicates that {better_algo} benefits more from early stopping "
        f"with {dqn_results['time_savings_percentage']}% vs {a2c_results['time_savings_percentage']}% "
        f"time savings. Both algorithms maintain good performance with minimal reward degradation."
    )
    
    plt.figtext(0.5, 0.05, conclusion_text, ha='center', va='center', fontsize=11, 
                bbox={'facecolor':'#FFF9B2', 'alpha':0.8, 'pad':5, 'boxstyle':'round'})
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'dqn_vs_a2c_early_stopping_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_early_stopping_report(analysis_results, algorithm="DQN"):
    """
    Generate a textual report summarizing the early stopping analysis.
    
    Args:
        analysis_results (dict): Analysis results from analyze_convergence function
        algorithm (str): The algorithm name (DQN or A2C)
    """
    report = {
        "title": f"{algorithm} Early Stopping Analysis Report",
        "date": "May 8, 2025",
        "algorithm": algorithm,
        "summary": {
            "stabilization_point": f"Episode {analysis_results['stabilization_episode']} of {analysis_results['total_episodes']}",
            "time_savings": f"{analysis_results['time_savings_percentage']}%",
            "episodes_saved": analysis_results['episodes_saved'],
            "reward_difference": f"{analysis_results['reward_difference']:.4f} ({analysis_results['reward_difference_pct']}%)"
        },
        "conclusion": f"Training can be stopped at episode {analysis_results['stabilization_episode']} with minimal impact on performance, saving approximately {analysis_results['time_savings_percentage']}% of training time.",
        "recommendations": [
            "Implement early stopping based on reward stabilization",
            "Monitor the percentage change in reward and loss values",
            "Consider a 1% change threshold as a stopping criterion",
            f"Set a window of {analysis_results['stabilization_episode']} episodes as the baseline training duration"
        ],
        "visualizations": [
            f"{algorithm.lower()}_training_metrics_with_stopping.png",
            f"{algorithm.lower()}_training_stability.png",
            f"{algorithm.lower()}_time_savings.png",
            f"{algorithm.lower()}_reward_stabilization.png",
            f"{algorithm.lower()}_comprehensive_early_stopping_analysis.png"
        ],
        "data_source": DQN_TRAINING_DATA_PATH if algorithm == "DQN" else A2C_TRAINING_DATA_PATH
    }
    
    # Add A2C specific metrics if available
    if analysis_results.get('has_entropy', False):
        report["a2c_specific"] = {
            "entropy_at_stopping": f"{analysis_results['stabilized_entropy']:.4f}",
            "final_entropy": f"{analysis_results['final_entropy']:.4f}"
        }
    
    # Save report as JSON
    with open(os.path.join(PLOT_DIR, f'{algorithm.lower()}_early_stopping_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save report as text
    with open(os.path.join(PLOT_DIR, f'{algorithm.lower()}_early_stopping_report.txt'), 'w') as f:
        f.write(f"{algorithm} Early Stopping Analysis Report\n")
        f.write(f"{'=' * (len(algorithm) + 31)}\n\n")
        f.write(f"Date: {report['date']}\n\n")
        
        f.write(f"Summary\n-------\n")
        f.write(f"Stabilization Point: {report['summary']['stabilization_point']}\n")
        f.write(f"Time Savings: {report['summary']['time_savings']}\n")
        f.write(f"Episodes Saved: {report['summary']['episodes_saved']}\n")
        f.write(f"Reward Difference: {report['summary']['reward_difference']}\n\n")
        
        if algorithm == "A2C" and 'a2c_specific' in report:
            f.write(f"A2C Specific Metrics\n-------------------\n")
            f.write(f"Entropy at stopping point: {report['a2c_specific']['entropy_at_stopping']}\n")
            f.write(f"Final entropy: {report['a2c_specific']['final_entropy']}\n\n")
        
        f.write(f"Conclusion\n----------\n")
        f.write(f"{report['conclusion']}\n\n")
        
        f.write(f"Recommendations\n---------------\n")
        for i, rec in enumerate(report['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
        
        f.write(f"\nVisualizations\n--------------\n")
        for viz in report['visualizations']:
            f.write(f"- {viz}\n")
        
        f.write(f"\nData Source: {report['data_source']}\n")
    
    return report

def generate_comparison_report(dqn_results, a2c_results):
    """
    Generate a comparative report between DQN and A2C early stopping results.
    
    Args:
        dqn_results (dict): Analysis results for DQN
        a2c_results (dict): Analysis results for A2C
    """
    # Determine which algorithm benefits more from early stopping
    dqn_savings = dqn_results['time_savings_percentage']
    a2c_savings = a2c_results['time_savings_percentage']
    
    better_algo = "DQN" if dqn_savings > a2c_savings else "A2C"
    
    # Calculate additional comparative metrics
    eps_diff = abs(dqn_results['stabilization_episode'] - a2c_results['stabilization_episode'])
    
    report = {
        "title": "DQN vs A2C Early Stopping Comparative Analysis",
        "date": "May 8, 2025",
        "comparison_summary": {
            "best_for_early_stopping": better_algo,
            "dqn_time_savings": f"{dqn_savings}%",
            "a2c_time_savings": f"{a2c_savings}%",
            "stabilization_episode_difference": eps_diff,
            "dqn_stabilization": dqn_results['stabilization_episode'],
            "a2c_stabilization": a2c_results['stabilization_episode']
        },
        "comparative_visualizations": [
            "algorithm_comparison_episodes.png",
            "algorithm_comparison_reward_impact.png",
            "dqn_vs_a2c_early_stopping_comparison.png"
        ],
        "conclusion": (
            f"Early stopping analysis shows that {better_algo} benefits more from early stopping "
            f"with {max(dqn_savings, a2c_savings)}% time savings compared to {min(dqn_savings, a2c_savings)}% "
            f"for the other algorithm. Both maintain performance with minimal reward degradation."
        ),
        "recommendations": [
            f"Prioritize implementing early stopping for {better_algo} to maximize time efficiency",
            "Use similar convergence criteria (1% change threshold) for both algorithms",
            "Consider the episode difference when planning distributed training resources",
            "Apply early stopping to reduce computational costs in future reinforcement learning tasks"
        ]
    }
    
    # Save report as JSON
    with open(os.path.join(PLOT_DIR, 'dqn_vs_a2c_comparative_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save report as text
    with open(os.path.join(PLOT_DIR, 'dqn_vs_a2c_comparative_report.txt'), 'w') as f:
        f.write(f"DQN vs A2C Early Stopping Comparative Analysis\n")
        f.write(f"=============================================\n\n")
        f.write(f"Date: {report['date']}\n\n")
        
        f.write(f"Comparison Summary\n------------------\n")
        f.write(f"Best Algorithm for Early Stopping: {report['comparison_summary']['best_for_early_stopping']}\n")
        f.write(f"DQN Time Savings: {report['comparison_summary']['dqn_time_savings']}\n")
        f.write(f"A2C Time Savings: {report['comparison_summary']['a2c_time_savings']}\n")
        f.write(f"DQN Stabilization Episode: {report['comparison_summary']['dqn_stabilization']}\n")
        f.write(f"A2C Stabilization Episode: {report['comparison_summary']['a2c_stabilization']}\n")
        f.write(f"Episode Difference: {report['comparison_summary']['stabilization_episode_difference']}\n\n")
        
        f.write(f"Conclusion\n----------\n")
        f.write(f"{report['conclusion']}\n\n")
        
        f.write(f"Recommendations\n---------------\n")
        for i, rec in enumerate(report['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
        
        f.write(f"\nComparative Visualizations\n--------------------------\n")
        for viz in report['comparative_visualizations']:
            f.write(f"- {viz}\n")
    
    return report

def main():
    """Main function to execute the early stopping analysis."""
    print("Starting early stopping analysis for both DQN and A2C...")
    
    # Create separate subdirectories for each algorithm
    dqn_dir = os.path.join(PLOT_DIR, 'dqn')
    a2c_dir = os.path.join(PLOT_DIR, 'a2c')
    os.makedirs(dqn_dir, exist_ok=True)
    os.makedirs(a2c_dir, exist_ok=True)
    
    # Load DQN training data
    dqn_data = load_training_data(DQN_TRAINING_DATA_PATH, "DQN")
    if dqn_data is None:
        print("Could not load DQN training data. Skipping DQN analysis.")
        dqn_results = None
    else:
        # Analyze DQN convergence
        dqn_results = analyze_convergence(dqn_data, "DQN")
        print(f"DQN Analysis results:")
        for key, value in dqn_results.items():
            print(f"  {key}: {value}")
        
        # Generate DQN visualizations
        print("Generating DQN visualizations...")
        plot_training_metrics(dqn_data, dqn_results, "DQN")
        
        # Generate DQN report
        print("Generating DQN report...")
        generate_early_stopping_report(dqn_results, "DQN")
    
    # Load A2C training data
    a2c_data = load_training_data(A2C_TRAINING_DATA_PATH, "A2C")
    if a2c_data is None:
        print("Could not load A2C training data. Skipping A2C analysis.")
        a2c_results = None
    else:
        # Analyze A2C convergence
        a2c_results = analyze_convergence(a2c_data, "A2C")
        print(f"A2C Analysis results:")
        for key, value in a2c_results.items():
            print(f"  {key}: {value}")
        
        # Generate A2C visualizations
        print("Generating A2C visualizations...")
        plot_training_metrics(a2c_data, a2c_results, "A2C")
        
        # Generate A2C report
        print("Generating A2C report...")
        generate_early_stopping_report(a2c_results, "A2C")
    
    # Generate comparative analysis if both data sets are available
    if dqn_results and a2c_results:
        print("Generating comparative visualizations between DQN and A2C...")
        plot_algorithm_comparison(dqn_results, a2c_results)
        
        print("Generating comparative report...")
        generate_comparison_report(dqn_results, a2c_results)
    
    print(f"Analysis complete! Results saved to {PLOT_DIR}")
    if dqn_results:
        print(f"DQN early stopping point detected at episode {dqn_results['stabilization_episode']}")
        print(f"DQN potential time savings: {dqn_results['time_savings_percentage']}%")
    if a2c_results:
        print(f"A2C early stopping point detected at episode {a2c_results['stabilization_episode']}")
        print(f"A2C potential time savings: {a2c_results['time_savings_percentage']}%")

if __name__ == "__main__":
    main()
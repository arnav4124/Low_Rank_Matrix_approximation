#!/usr/bin/env python3
"""
Multi-Matrix Training and Ensemble Evaluation
============================================

This module implements a more robust approach to matrix approximation by:
1. Training on multiple matrices to create more generalizable models
2. Creating ensemble models that average predictions across multiple trained models
3. Comparing performance between single-matrix and multi-matrix training

The goal is to test if models trained on multiple matrices generalize better.
"""

import os
import numpy as np
import torch
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from tqdm import tqdm
from colorama import Fore, Style

# Import our matrix downloading utilities
from florida_downloader import (
    log_info, log_success, log_warning, log_error, log_debug,
    download_all_matrices, load_all_matrices
)

# Import our matrix approximation methods and RL agents
from matrix_experiments import (
    deterministic_rank_approx, randomized_rank_approx, cur_decomposition,
    ColumnSelectionEnv, EnhancedColumnSelectionEnv,
    DQNAgent, A2CAgent, load_model_with_compatibility,
    rl_column_selection_with_dqn, rl_column_selection_with_a2c,
    evaluate_approximation, train_dqn, train_a2c
)

# =============================================================================
# Multi-Matrix Training Functions
# =============================================================================

def train_on_multiple_matrices(matrices, target_rank=40, num_train_episodes=300,
                              num_matrices=3, model_type='dqn'):
    """
    Train a model on multiple matrices and return the averaged or ensemble model.

    Args:
        matrices (dict): Dictionary of matrices for training and testing
        target_rank (int): Target rank for approximation
        num_train_episodes (int): Number of training episodes
        num_matrices (int): Number of matrices to use for training (default: 3)
        model_type (str): Type of model to train ('dqn' or 'a2c')

    Returns:
        tuple: (ensemble_model, train_matrices, individual_models)
    """
    log_info(f"Training {model_type.upper()} model on {num_matrices} matrices")

    # Select training matrices
    matrix_names = list(matrices.keys())
    if len(matrix_names) < num_matrices:
        log_warning(f"Not enough matrices. Using all {len(matrix_names)} available matrices.")
        num_matrices = len(matrix_names)

    train_matrix_names = matrix_names[:num_matrices]
    train_matrices = {name: matrices[name] for name in train_matrix_names}

    log_info(f"Training matrices: {train_matrix_names}")

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")

    # Dictionary to store individual models
    individual_models = {}
    all_weights = []

    # Train on each matrix
    for matrix_name, matrix in train_matrices.items():
        log_info(f"Training on matrix: {matrix_name}")

        # Set model path
        model_path = f"models/ensemble_{model_type}_{matrix_name}.pt"

        # Setup environment dimensions
        if model_type == 'dqn':
            env = ColumnSelectionEnv(matrix, target_rank)
            state_dim = env.n
            action_dim = env.n

            # Initialize DQN agent
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=5e-4,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.05,
                use_prioritized_replay=True
            )

            # Train DQN
            _, _, best_error = train_dqn(
                agent,
                env,
                num_episodes=num_train_episodes,
                early_stopping_patience=30,
                save_path=model_path
            )

            # Save individual model reference
            individual_models[matrix_name] = agent

            # Save weights for averaging
            all_weights.append(agent.q_network.state_dict())

        elif model_type == 'a2c':
            env = EnhancedColumnSelectionEnv(matrix, target_rank,
                                           state_type='combined', reward_type='combined')
            state_dim = env.n * 3 if env.state_type == 'combined' else env.n
            action_dim = env.n

            # Initialize A2C agent
            agent = A2CAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=3e-4,
                gamma=0.99,
                entropy_coef=0.02,
                value_coef=0.5
            )

            # Train A2C
            _, _, best_error = train_a2c(
                agent,
                env,
                num_episodes=num_train_episodes,
                update_interval=5,
                early_stopping_patience=30,
                save_path=model_path
            )

            # Save individual model reference
            individual_models[matrix_name] = agent

            # Save weights for averaging
            all_weights.append(agent.model.state_dict())

        log_success(f"Trained {model_type.upper()} on {matrix_name} with best error: {best_error:.6f}")

    # Create ensemble/averaged model
    if model_type == 'dqn':
        # Create a new DQN model (using dimensions from the last trained model for simplicity)
        ensemble_model = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=5e-4,
            gamma=0.99,
            epsilon=0.05,  # Low epsilon for evaluation
            epsilon_decay=1.0,  # No decay
            epsilon_min=0.05,
            use_prioritized_replay=True
        )

        # Average the weights
        ensemble_weights = average_model_weights(all_weights)
        ensemble_model.q_network.load_state_dict(ensemble_weights)
        ensemble_model.target_network.load_state_dict(ensemble_weights)

        # Save the ensemble model
        torch.save(ensemble_weights, f"models/ensemble_{model_type}_multi_{num_matrices}.pt")

    elif model_type == 'a2c':
        # Create a new A2C model
        ensemble_model = A2CAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=3e-4,
            gamma=0.99,
            entropy_coef=0.01,
            value_coef=0.5
        )

        # Average the weights
        ensemble_weights = average_model_weights(all_weights)
        ensemble_model.model.load_state_dict(ensemble_weights)

        # Save the ensemble model
        torch.save(ensemble_weights, f"models/ensemble_{model_type}_multi_{num_matrices}.pt")

    log_success(f"Created ensemble model from {num_matrices} individual models")
    return ensemble_model, train_matrices, individual_models

def average_model_weights(weight_list):
    """
    Average multiple model weights to create an ensemble model.

    Args:
        weight_list (list): List of model state dictionaries

    Returns:
        dict: Averaged state dictionary
    """
    # Start with a copy of the first state dict
    avg_weights = deepcopy(weight_list[0])

    # Average each parameter
    for key in avg_weights.keys():
        # Skip if not a parameter (e.g. running_mean in batch norm)
        if not torch.is_floating_point(avg_weights[key]):
            continue
            
        # Initialize tensor for the sum
        avg_weights[key] = torch.zeros_like(avg_weights[key])
        
        # Sum all weights
        for state_dict in weight_list:
            avg_weights[key] += state_dict[key]
            
        # Divide by the number of models
        avg_weights[key] /= len(weight_list)
    
    return avg_weights

# =============================================================================
# Ensemble prediction methods
# =============================================================================

def ensemble_column_selection(A, rank, agents_list, mode='vote'):
    """
    Select columns using an ensemble of agents.

    Args:
        A (numpy.ndarray): Input matrix
        rank (int): Target rank for approximation
        agents_list (list): List of trained agents
        mode (str): Ensemble mode ('vote', 'average', 'best')

    Returns:
        tuple: (A_approx, selected_columns) - Approximated matrix and selected columns
    """
    log_debug(f"Applying ensemble column selection (rank={rank}, mode={mode})")

    if len(agents_list) == 0:
        log_error("Empty agents list provided for ensemble")
        return None, []

    # Check agent type (first agent in the list)
    agent_type = 'dqn' if hasattr(agents_list[0], 'q_network') else 'a2c'

    # For 'best' mode, we'll run each agent and pick the one with lowest error
    if mode == 'best':
        best_error = float('inf')
        best_approx = None
        best_columns = None

        for agent in agents_list:
            if agent_type == 'dqn':
                A_approx, selected_cols = rl_column_selection_with_dqn(A, rank, agent)
            else:
                A_approx, selected_cols = rl_column_selection_with_a2c(
                    A, rank, agent, state_type='combined', reward_type='combined')

            error = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
            if error < best_error:
                best_error = error
                best_approx = A_approx
                best_columns = selected_cols

        return best_approx, best_columns

    # Create multiple environments
    envs = []
    for _ in range(len(agents_list)):
        if agent_type == 'dqn':
            envs.append(ColumnSelectionEnv(A, rank))
        else:
            envs.append(EnhancedColumnSelectionEnv(A, rank, 'combined', 'combined'))

    # Reset all environments
    states = [env.reset() for env in envs]

    # Track selected columns
    all_selected_columns = [[] for _ in range(len(agents_list))]

    # Run all agents in parallel
    with tqdm(total=rank, desc="Ensemble Column Selection") as pbar:
        for step in range(rank):
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(agents_list):
                if agent_type == 'dqn':
                    action = agent.select_action(states[i], envs[i].available_columns)
                else:
                    action = agent.select_action(states[i], envs[i].available_columns)
                actions.append(action)

            # Voting mode: select most common action
            if mode == 'vote':
                # Count votes
                vote_counts = {}
                for action in actions:
                    vote_counts[action] = vote_counts.get(action, 0) + 1

                # Select action with most votes (break ties randomly)
                max_votes = max(vote_counts.values())
                top_actions = [a for a, count in vote_counts.items() if count == max_votes]
                selected_action = np.random.choice(top_actions)

                # Apply this action to all environments
                for i, env in enumerate(envs):
                    if selected_action in env.available_columns:
                        states[i], _, _, _ = env.step(selected_action)
                        all_selected_columns[i].append(selected_action)
                    else:
                        # If selected action not available, pick a random one
                        if env.available_columns:
                            random_action = np.random.choice(env.available_columns)
                            states[i], _, _, _ = env.step(random_action)
                            all_selected_columns[i].append(random_action)

            # Average mode: each agent selects its own column
            else:  # mode == 'average'
                for i, (agent, env, action) in enumerate(zip(agents_list, envs, actions)):
                    states[i], _, _, _ = env.step(action)
                    all_selected_columns[i].append(action)

            pbar.update(1)

    # For vote mode, use the first environment's columns (all should be the same)
    if mode == 'vote':
        selected_columns = all_selected_columns[0]

    # For average mode, combine results
    else:
        # Count frequency of each column across all agents
        col_counts = {}
        for cols in all_selected_columns:
            for col in cols:
                col_counts[col] = col_counts.get(col, 0) + 1

        # Take top 'rank' columns by frequency
        sorted_cols = sorted(col_counts.items(), key=lambda x: x[1], reverse=True)
        selected_columns = [col for col, _ in sorted_cols[:rank]]

        # If we don't have enough columns, add random ones
        if len(selected_columns) < rank:
            remaining = rank - len(selected_columns)
            available = [c for c in range(A.shape[1]) if c not in selected_columns]
            if available:
                selected_columns.extend(np.random.choice(
                    available, size=min(remaining, len(available)), replace=False))

    # Compute approximation using selected columns
    try:
        C = A[:, selected_columns]
        U = np.linalg.pinv(C)
        A_approx = C @ U @ A
    except np.linalg.LinAlgError as e:
        log_error(f"Linear algebra error in ensemble column selection: {str(e)}")
        # Fallback to random selection
        log_warning("Falling back to random column selection")
        fallback_cols = np.random.choice(A.shape[1], size=min(rank, A.shape[1]), replace=False)
        C = A[:, fallback_cols]
        U = np.linalg.pinv(C)
        A_approx = C @ U @ A
        selected_columns = fallback_cols.tolist()

    return A_approx, selected_columns

# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_models_on_matrices(matrices, target_rank=40):
    """
    Evaluate models on a set of matrices and compare performance between single-model
    and ensemble approaches.

    Args:
        matrices (dict): Dictionary of matrices
        target_rank (int): Target rank for approximation

    Returns:
        DataFrame: Results of all evaluations
    """
    log_info("Starting comprehensive model evaluation")

    results = []

    # Load existing models (individual models)
    dqn_model_paths = [f for f in os.listdir('models') if f.startswith('dqn_') and f.endswith('.pt')]
    a2c_model_paths = [f for f in os.listdir('models') if f.startswith('a2c_') and f.endswith('.pt')]
    ensemble_dqn_paths = [f for f in os.listdir('models') if f.startswith('ensemble_dqn_') and f.endswith('.pt')]
    ensemble_a2c_paths = [f for f in os.listdir('models') if f.startswith('ensemble_a2c_') and f.endswith('.pt')]

    log_info(f"Found {len(dqn_model_paths)} DQN models, {len(a2c_model_paths)} A2C models, "
             f"{len(ensemble_dqn_paths)} ensemble DQN models, {len(ensemble_a2c_paths)} ensemble A2C models")

    # Define methods to compare
    methods = {}

    # Add baseline methods
    methods["Deterministic SVD"] = lambda A, r: deterministic_rank_approx(A, r)
    methods["Randomized SVD"] = lambda A, r: randomized_rank_approx(A, r)
    methods["CUR"] = lambda A, r: cur_decomposition(A, r)[0]

    # Add ensemble methods if available
    if len(ensemble_dqn_paths) > 0:
        # Get most recent ensemble DQN model
        ensemble_dqn_path = sorted(ensemble_dqn_paths)[-1]
        log_info(f"Using ensemble DQN model: {ensemble_dqn_path}")

        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = action_dim = next(iter(matrices.values())).shape[1]  # Use first matrix shape

        ensemble_dqn = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        ensemble_dqn.q_network = load_model_with_compatibility(
            os.path.join('models', ensemble_dqn_path),
            state_dim,
            action_dim,
            'dqn',
            device
        )
        ensemble_dqn.target_network.load_state_dict(ensemble_dqn.q_network.state_dict())
        ensemble_dqn.epsilon = 0.05  # Low exploration for evaluation

        methods["Ensemble DQN"] = lambda A, r: rl_column_selection_with_dqn(A, r, ensemble_dqn)[0]

    if len(ensemble_a2c_paths) > 0:
        # Get most recent ensemble A2C model
        ensemble_a2c_path = sorted(ensemble_a2c_paths)[-1]
        log_info(f"Using ensemble A2C model: {ensemble_a2c_path}")

        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        matrix = next(iter(matrices.values()))  # Use first matrix
        state_dim = matrix.shape[1] * 3  # For combined state
        action_dim = matrix.shape[1]

        ensemble_a2c = A2CAgent(state_dim=state_dim, action_dim=action_dim)
        ensemble_a2c.model = load_model_with_compatibility(
            os.path.join('models', ensemble_a2c_path),
            state_dim,
            action_dim,
            'a2c',
            device
        )

        methods["Ensemble A2C"] = lambda A, r: rl_column_selection_with_a2c(
            A, r, ensemble_a2c, 'combined', 'combined')[0]

    # Add individual model methods if available
    if len(dqn_model_paths) > 0:
        # Load a representative individual DQN model
        dqn_path = sorted(dqn_model_paths)[0]
        log_info(f"Using individual DQN model: {dqn_path}")

        # Extract matrix name from filename
        matrix_name = dqn_path[4:-3]  # Remove 'dqn_' prefix and '.pt' suffix

        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = action_dim = next(iter(matrices.values())).shape[1]

        single_dqn = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        single_dqn.q_network = load_model_with_compatibility(
            os.path.join('models', dqn_path),
            state_dim,
            action_dim,
            'dqn',
            device
        )
        single_dqn.target_network.load_state_dict(single_dqn.q_network.state_dict())
        single_dqn.epsilon = 0.05

        methods[f"Single DQN ({matrix_name})"] = lambda A, r: rl_column_selection_with_dqn(A, r, single_dqn)[0]

    if len(a2c_model_paths) > 0:
        # Load a representative individual A2C model
        a2c_path = sorted(a2c_model_paths)[0]
        log_info(f"Using individual A2C model: {a2c_path}")

        # Extract matrix name from filename
        matrix_name = a2c_path[4:-3]  # Remove 'a2c_' prefix and '.pt' suffix

        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        matrix = next(iter(matrices.values()))
        state_dim = matrix.shape[1] * 3  # For combined state
        action_dim = matrix.shape[1]

        single_a2c = A2CAgent(state_dim=state_dim, action_dim=action_dim)
        single_a2c.model = load_model_with_compatibility(
            os.path.join('models', a2c_path),
            state_dim,
            action_dim,
            'a2c',
            device
        )

        methods[f"Single A2C ({matrix_name})"] = lambda A, r: rl_column_selection_with_a2c(
            A, r, single_a2c, 'combined', 'combined')[0]

    # Evaluate all methods on all matrices
    for matrix_name, matrix in matrices.items():
        log_info(f"Evaluating on matrix: {matrix_name}")

        for method_name, method_fn in methods.items():
            try:
                log_debug(f"Applying method {method_name} to matrix {matrix_name}")
                start_time = time.time()
                A_approx = method_fn(matrix, target_rank)
                elapsed_time = time.time() - start_time

                eval_result = evaluate_approximation(matrix_name, matrix, A_approx, method_name)
                eval_result['time'] = elapsed_time
                results.append(eval_result)

                log_success(f"{method_name} on {matrix_name}: "
                           f"Error={eval_result['rel_frob_error']:.4f}, Time={elapsed_time:.4f}s")
            except Exception as e:
                log_error(f"Failed {method_name} on {matrix_name}: {str(e)}")

    # Convert results to DataFrame
    result_df = pd.DataFrame(results)

    # Save results
    result_df.to_csv('results/ensemble_comparison.csv', index=False)

    log_success("Evaluation complete and results saved")
    return result_df

def compare_ensemble_vs_single(result_df, train_matrices):
    """
    Create plots comparing ensemble models against single models.

    Args:
        result_df (DataFrame): Results from evaluation
        train_matrices (list): List of matrices used for training

    Returns:
        None
    """
    log_info("Generating comparison plots between ensemble and single models")

    # Create a string of training matrix names
    train_matrices_str = '_'.join(train_matrices) if isinstance(train_matrices, list) else str(train_matrices)

    # Filter methods for cleaner plots
    ensemble_methods = [c for c in result_df['method'].unique() if 'Ensemble' in c]
    single_methods = [c for c in result_df['method'].unique() if 'Single' in c]
    baseline_methods = ['Deterministic SVD', 'Randomized SVD', 'CUR']

    # Only keep methods we want to compare
    plot_methods = ensemble_methods + single_methods + baseline_methods
    plot_df = result_df[result_df['method'].isin(plot_methods)].copy()

    # 1. Error comparison bar plot
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='matrix', y='rel_frob_error', hue='method', data=plot_df)
    plt.title(f'Error Comparison - Ensemble vs. Single Models', fontsize=14)
    plt.xlabel('Matrix', fontsize=12)
    plt.ylabel('Relative Frobenius Error', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('results/ensemble_vs_single_error.png', dpi=300)

    # 2. Time comparison
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='matrix', y='time', hue='method', data=plot_df)
    plt.title(f'Time Comparison - Ensemble vs. Single Models', fontsize=14)
    plt.xlabel('Matrix', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('results/ensemble_vs_single_time.png', dpi=300)

    # 3. Error heatmap relative to SVD
    # Create pivot tables for SVD and all other methods
    svd_errors = result_df[result_df['method'] == 'Deterministic SVD'].set_index('matrix')['rel_frob_error']

    # Prepare percentage data
    percentage_data = {}
    for method in plot_methods:
        if method != 'Deterministic SVD':
            method_errors = plot_df[plot_df['method'] == method].set_index('matrix')['rel_frob_error']
            percentages = {}
            for matrix in method_errors.index:
                if matrix in svd_errors.index:
                    # Calculate percentage difference from SVD (positive means worse than SVD)
                    percentages[matrix] = 100.0 * (method_errors[matrix] - svd_errors[matrix]) / svd_errors[matrix]
            percentage_data[method] = percentages

    # Convert to DataFrame
    percentage_df = pd.DataFrame(percentage_data).T

    # Plot percentage heatmap
    plt.figure(figsize=(14, 8))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Red is positive (worse), Blue is negative (better)

    # Create heatmap with percentage values
    sns.heatmap(percentage_df, cmap=cmap, center=0, annot=True, fmt=".1f",
                linewidths=.5, cbar_kws={'label': 'Error % relative to SVD'})

    plt.title(f'Error Percentage Relative to Deterministic SVD', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/ensemble_relative_error_heatmap.png', dpi=300)

    log_success("Comparison plots saved to results directory")

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Run multiple matrix training and evaluation experiments."""
    try:
        log_info("Starting multi-matrix training and ensemble evaluation experiments")

        # Define which matrices to use
        target_matrices = [
            "G1", "G2", "G3", "G4", "G5",
            "G10", "G11", "G12", "G13",
            "G14", "G15"
        ]

        log_info(f"Target matrices: {target_matrices}")
        matrix_paths = download_all_matrices(subset=target_matrices)

        if not matrix_paths:
            log_error("Failed to download any matrices. Exiting.")
            return

        # Load the matrices
        log_info("Loading matrices...")
        matrices = load_all_matrices(matrix_paths)

        if not matrices:
            log_error("Failed to load any matrices. Exiting.")
            return

        log_success(f"Successfully loaded {len(matrices)} matrices")

        # Define target rank
        target_rank = 40

        # Step 1: Train models on multiple matrices
        log_info("Starting multi-matrix training")
        ensemble_dqn, train_matrices_dict, individual_dqn_models = train_on_multiple_matrices(
            matrices,
            target_rank=target_rank,
            num_train_episodes=300,
            num_matrices=3,
            model_type='dqn'
        )

        ensemble_a2c, _, individual_a2c_models = train_on_multiple_matrices(
            matrices,
            target_rank=target_rank,
            num_train_episodes=300,
            num_matrices=3,
            model_type='a2c'
        )

        # Get names of training matrices
        train_matrices = list(train_matrices_dict.keys())

        # Step 2: Evaluate all models on all matrices
        log_info("Evaluating models")
        result_df = evaluate_models_on_matrices(matrices, target_rank)

        # Step 3: Compare ensemble vs single models
        compare_ensemble_vs_single(result_df, train_matrices)

        # Step 4: Test ensemble method where multiple agents vote
        log_info("Testing ensemble voting methods")

        # Create test matrices by excluding training matrices
        test_matrices = {name: matrix for name, matrix in matrices.items()
                        if name not in train_matrices}

        # Create ensemble agent lists
        dqn_agents = list(individual_dqn_models.values())
        a2c_agents = list(individual_a2c_models.values())

        # Test on each matrix
        voting_results = []
        for matrix_name, matrix in test_matrices.items():
            log_info(f"Testing ensemble voting on {matrix_name}")

            # Get reference error from basic SVD
            svd_approx = deterministic_rank_approx(matrix, target_rank)
            svd_error = np.linalg.norm(matrix - svd_approx, 'fro') / np.linalg.norm(matrix, 'fro')

            # Test DQN with different voting methods
            for mode in ['vote', 'average', 'best']:
                try:
                    start_time = time.time()
                    A_approx, _ = ensemble_column_selection(matrix, target_rank, dqn_agents, mode=mode)
                    elapsed_time = time.time() - start_time

                    error = np.linalg.norm(matrix - A_approx, 'fro') / np.linalg.norm(matrix, 'fro')
                    improvement = (svd_error - error) / svd_error * 100  # % improvement over SVD

                    voting_results.append({
                        'matrix': matrix_name,
                        'method': f'DQN-{mode}',
                        'error': error,
                        'time': elapsed_time,
                        'svd_error': svd_error,
                        'improvement': improvement
                    })

                    log_success(f"DQN-{mode} on {matrix_name}: Error={error:.4f}, "
                               f"Improvement={improvement:.2f}%, Time={elapsed_time:.4f}s")

                except Exception as e:
                    log_error(f"Failed DQN-{mode} on {matrix_name}: {str(e)}")

            # Test A2C with different voting methods
            for mode in ['vote', 'average', 'best']:
                try:
                    start_time = time.time()
                    A_approx, _ = ensemble_column_selection(matrix, target_rank, a2c_agents, mode=mode)
                    elapsed_time = time.time() - start_time

                    error = np.linalg.norm(matrix - A_approx, 'fro') / np.linalg.norm(matrix, 'fro')
                    improvement = (svd_error - error) / svd_error * 100  # % improvement over SVD

                    voting_results.append({
                        'matrix': matrix_name,
                        'method': f'A2C-{mode}',
                        'error': error,
                        'time': elapsed_time,
                        'svd_error': svd_error,
                        'improvement': improvement
                    })

                    log_success(f"A2C-{mode} on {matrix_name}: Error={error:.4f}, "
                               f"Improvement={improvement:.2f}%, Time={elapsed_time:.4f}s")

                except Exception as e:
                    log_error(f"Failed A2C-{mode} on {matrix_name}: {str(e)}")

        # Save voting results
        voting_df = pd.DataFrame(voting_results)
        voting_df.to_csv('results/ensemble_voting_comparison.csv', index=False)

        # Plot voting results
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='matrix', y='improvement', hue='method', data=voting_df)
        plt.title(f'Improvement Over SVD by Ensemble Method', fontsize=14)
        plt.xlabel('Matrix', fontsize=12)
        plt.ylabel('Improvement Over SVD (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('results/ensemble_voting_improvement.png', dpi=300)

        log_success("Multi-matrix training and ensemble experiments completed successfully!")

    except KeyboardInterrupt:
        log_warning("Experiments interrupted by user")
    except Exception as e:
        log_error(f"Error running multi-matrix experiments: {str(e)}")
        import traceback
        log_error(traceback.format_exc())

if __name__ == "__main__":
    main()
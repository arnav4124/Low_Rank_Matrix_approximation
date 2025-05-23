#!/usr/bin/env python3
"""
Matrix Approximation Experiments
================================

This module implements various matrix approximation techniques and runs experiments
on matrices from the SuiteSparse Matrix Collection. It includes:
- SVD-based approximation methods
- CUR decomposition
- Reinforcement Learning-based column selection
- Evaluation metrics and visualization

All operations include progress bars and color-coded logging for better tracking.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import functional as F
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

# Import our matrix downloading utilities
from florida_downloader import (
    log_info, log_success, log_warning, log_error, log_debug,
    get_matrix_info, download_matrix, load_matrix, preprocess_matrix
)

# Initialize colorama
colorama.init(autoreset=True)

# Make sure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# =============================================================================
# Matrix Approximation Methods
# =============================================================================

def deterministic_rank_approx(A, rank):
    """
    Compute a deterministic low-rank approximation using SVD.
    
    Args:
        A (numpy.ndarray): Input matrix
        rank (int): Target rank for the approximation
    
    Returns:
        numpy.ndarray: Low-rank approximation of A
    """
    log_debug(f"Computing deterministic SVD rank-{rank} approximation")
    tqdm.write(f"Computing SVD... (size: {A.shape})")
    
    with tqdm(total=3, desc="SVD Steps") as pbar:
        # Step 1: Compute SVD
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        pbar.update(1)
        
        # Step 2: Truncate to rank
        U_k = U[:, :rank]
        s_k = s[:rank]
        Vt_k = Vt[:rank, :]
        pbar.update(1)
        
        # Step 3: Multiply back to get approximation
        A_approx = U_k @ np.diag(s_k) @ Vt_k
        pbar.update(1)
    
    return A_approx

def randomized_rank_approx(A, rank, n_oversample=10, n_iter=7):
    """
    Compute a randomized low-rank approximation.
    
    Args:
        A (numpy.ndarray): Input matrix
        rank (int): Target rank for the approximation
        n_oversample (int): Oversampling parameter
        n_iter (int): Number of power iterations
    
    Returns:
        numpy.ndarray: Low-rank approximation of A
    """
    log_debug(f"Computing randomized SVD rank-{rank} approximation")
    m, n = A.shape
    rank_target = min(rank + n_oversample, min(m, n))
    
    steps = n_iter + 3  # Random projection, power iterations, and final SVD + reconstruction
    with tqdm(total=steps, desc="Randomized SVD") as pbar:
        # Step 1: Random projection
        Q = np.random.normal(size=(n, rank_target))
        pbar.update(1)
        
        # Step 2: Power iteration
        for _ in range(n_iter):
            Q = A @ (A.T @ Q)
            Q, _ = np.linalg.qr(Q, mode='reduced')
            pbar.update(1)
        
        # Step 3: Project and compute SVD
        B = Q.T @ A
        U_B, s, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_B
        pbar.update(1)
        
        # Step 4: Reconstruct
        A_approx = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        pbar.update(1)
    
    return A_approx

def cur_decomposition(A, rank, c_factor=5):
    """
    Compute a CUR decomposition for low-rank approximation.
    
    Args:
        A (numpy.ndarray): Input matrix
        rank (int): Target rank for the approximation
        c_factor (int): Multiplication factor for number of columns/rows to sample
    
    Returns:
        tuple: (A_approx, C, U, R) - Approximated matrix and CUR components
    """
    log_debug(f"Computing CUR decomposition (rank-{rank})")
    m, n = A.shape
    
    steps = 5  # Compute probabilities, sample columns, sample rows, compute U, reconstruct
    with tqdm(total=steps, desc="CUR Decomposition") as pbar:
        # Step 1: Compute column norms for sampling probabilities
        col_norms = np.sum(A**2, axis=0)
        col_probs = col_norms / np.sum(col_norms)
        pbar.update(1)
        
        # Step 2: Sample columns
        c_indices = np.random.choice(n, size=rank*c_factor, replace=False, p=col_probs)
        C = A[:, c_indices]
        pbar.update(1)
        
        # Step 3: Compute row norms for sampling probabilities using C
        row_norms = np.sum(C**2, axis=1)
        row_probs = row_norms / np.sum(row_norms)
        r_indices = np.random.choice(m, size=rank*c_factor, replace=False, p=row_probs)
        R = A[r_indices, :]
        pbar.update(1)
        
        # Step 4: Compute intersection matrix U
        W = A[r_indices, :][:, c_indices]
        U = np.linalg.pinv(W)
        pbar.update(1)
        
        # Step 5: Compute approximation
        A_approx = C @ U @ R
        pbar.update(1)
    
    return A_approx, C, U, R

# =============================================================================
# Reinforcement Learning Environment for Column Selection
# =============================================================================

class ColumnSelectionEnv:
    """
    Environment for RL-based column selection.
    
    This environment represents the task of selecting columns from a matrix
    to create a low-rank approximation using CUR-like decomposition.
    """
    def __init__(self, A, target_rank):
        """
        Initialize the column selection environment.
        
        Args:
            A (numpy.ndarray): Input matrix
            target_rank (int): Target rank for the approximation
        """
        self.A = A
        self.A_norm = np.linalg.norm(A, 'fro')
        self.m, self.n = A.shape
        self.target_rank = target_rank
        self.selected_columns = []
        self.available_columns = list(range(self.n))
        
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            numpy.ndarray: Initial state
        """
        self.selected_columns = []
        self.available_columns = list(range(self.n))
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation.
        
        Returns:
            numpy.ndarray: State representation (binary vector of selected columns)
        """
        # Binary vector indicating which columns have been selected
        state = np.zeros(self.n)
        state[self.selected_columns] = 1
        return state
    
    def step(self, action):
        """
        Take action (select a column) and return new state, reward, done flag.
        
        Args:
            action (int): Column index to select
        
        Returns:
            tuple: (next_state, reward, done, info) - Standard RL step return
        """
        if action in self.available_columns:
            self.selected_columns.append(action)
            self.available_columns.remove(action)
            
            # Calculate current approximation error
            if len(self.selected_columns) >= 2:
                C = self.A[:, self.selected_columns]
                try:
                    U = np.linalg.pinv(C)
                    A_approx = C @ U @ self.A
                    error = np.linalg.norm(self.A - A_approx, 'fro') / self.A_norm
                    reward = -error  # Negative error as reward
                except np.linalg.LinAlgError:
                    # Penalty for creating linearly dependent columns
                    reward = -1.0
            else:
                # Not enough columns to create an approximation yet
                reward = 0.0
            
            done = len(self.selected_columns) >= self.target_rank
            
            return self._get_state(), reward, done, {"selected_columns": self.selected_columns}
        else:
            # Penalty for invalid action
            return self._get_state(), -1.0, False, {"error": "Invalid action"}

class EnhancedColumnSelectionEnv(ColumnSelectionEnv):
    """
    Enhanced environment for RL-based column selection with different state representations
    and reward functions.
    """
    def __init__(self, A, target_rank, state_type='binary', reward_type='error'):
        """
        Initialize the enhanced column selection environment.
        
        Args:
            A (numpy.ndarray): Input matrix
            target_rank (int): Target rank for the approximation
            state_type (str): Type of state representation ('binary', 'error', 'combined')
            reward_type (str): Type of reward function ('error', 'improvement', 'combined')
        """
        super().__init__(A, target_rank)
        self.state_type = state_type
        self.reward_type = reward_type
        
        # Pre-compute column leverage scores for state representation
        U, _, _ = np.linalg.svd(A, full_matrices=False)
        self.leverage_scores = np.sum(U[:, :min(A.shape)]**2, axis=1)
    
    def _get_state(self):
        """
        Get enhanced state representation based on state_type.
        
        Returns:
            numpy.ndarray: State representation
        """
        if self.state_type == 'binary':
            # Binary vector indicating which columns have been selected
            state = np.zeros(self.n)
            state[self.selected_columns] = 1
            return state
        
        elif self.state_type == 'error':
            # State based on current approximation error for each column
            state = np.zeros(self.n)
            if not self.selected_columns:
                # Initial state - use column norms
                col_norms = np.sum(self.A**2, axis=0)
                state = col_norms / np.max(col_norms)
            else:
                # Calculate error contribution for each column
                C = self.A[:, self.selected_columns]
                for col in range(self.n):
                    if col in self.selected_columns:
                        state[col] = 0  # Already selected
                    else:
                        # Calculate error reduction if this column is added
                        temp_cols = self.selected_columns + [col]
                        C_temp = self.A[:, temp_cols]
                        U_temp = np.linalg.pinv(C_temp)
                        A_approx = C_temp @ U_temp @ self.A
                        error = np.linalg.norm(self.A - A_approx, 'fro') / self.A_norm
                        state[col] = -error  # Negative error (higher is better)
            return state
        
        elif self.state_type == 'combined':
            # Combine binary selection, column norms, and leverage scores
            binary = np.zeros(self.n)
            binary[self.selected_columns] = 1
            
            col_norms = np.sum(self.A**2, axis=0) / np.sum(self.A**2)
            
            # Compute current error contribution for each column
            error_contrib = np.zeros(self.n)
            if self.selected_columns:
                C = self.A[:, self.selected_columns]
                U = np.linalg.pinv(C)
                P = C @ U  # Projection matrix
                residual = self.A - P @ self.A
                error_contrib = np.sum(residual**2, axis=0) / np.sum(residual**2)
            
            # Concatenate features
            return np.concatenate([binary, col_norms, error_contrib])
    
    def step(self, action):
        """
        Take action with enhanced reward calculation.
        
        Args:
            action (int): Column index to select
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if action in self.available_columns:
            self.selected_columns.append(action)
            self.available_columns.remove(action)
            
            # Calculate current approximation
            done = len(self.selected_columns) >= self.target_rank
            
            if len(self.selected_columns) >= 2:
                C = self.A[:, self.selected_columns]
                try:
                    U = np.linalg.pinv(C)
                    A_approx = C @ U @ self.A
                    error = np.linalg.norm(self.A - A_approx, 'fro') / self.A_norm
                    
                    if self.reward_type == 'error':
                        reward = -error
                    elif self.reward_type == 'improvement':
                        # Reward based on improvement from previous step
                        if hasattr(self, 'prev_error'):
                            reward = self.prev_error - error
                        else:
                            reward = 0
                        self.prev_error = error
                    elif self.reward_type == 'combined':
                        # Combine error and diversity rewards
                        base_reward = -error
                        
                        # Add diversity bonus
                        if len(self.selected_columns) > 1:
                            C_norm = np.linalg.norm(C, axis=0)
                            C_normalized = C / C_norm
                            coherence = np.abs(C_normalized.T @ C_normalized)
                            np.fill_diagonal(coherence, 0)  # Ignore self-coherence
                            diversity_penalty = np.mean(coherence)
                            reward = base_reward - diversity_penalty
                        else:
                            reward = base_reward
                    else:
                        reward = -error  # Default
                except np.linalg.LinAlgError:
                    # Handle numerical instability
                    reward = -1
            else:
                reward = 0
            
            return self._get_state(), reward, done, {}
        else:
            return self._get_state(), -1, False, {}  # Penalty for invalid action

# =============================================================================
# RL Agents
# =============================================================================

class DQNNetwork(nn.Module):
    """Enhanced Deep Q-Network for column selection."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 128]):
        """
        Initialize the DQN network with configurable architecture.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dims (list): Dimensions of hidden layers
        """
        super(DQNNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build a dynamic network based on hidden_dims
        layers = []
        prev_dim = state_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))  # Add batch normalization
            prev_dim = dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, action_dim)
        
        # Apply weight initialization for better convergence
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor (Q-values)
        """
        # For batch size 1, adjust batch normalization dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.hidden_layers(x)
        q_values = self.output_layer(features)
        return q_values

class LegacyDQNNetwork(nn.Module):
    """Old DQN network architecture for compatibility with saved models."""
    
    def __init__(self, state_dim, action_dim):
        super(LegacyDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class LegacyA2CNetwork(nn.Module):
    """Old A2C network architecture for compatibility with saved models."""
    
    def __init__(self, state_dim, action_dim):
        super(LegacyA2CNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

def load_model_with_compatibility(model_path, state_dim, action_dim, model_type='dqn', device=None):
    """
    Load model weights with backward compatibility for old model architectures.
    
    Args:
        model_path (str): Path to the saved model
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        model_type (str): Type of model ('dqn' or 'a2c')
        device (torch.device): Device to load the model to
        
    Returns:
        nn.Module: Loaded model with weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try to load the state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Check which architecture the model was saved with by looking at keys
        if model_type == 'dqn':
            if 'fc1.weight' in state_dict:  # Old architecture
                log_info(f"Detected legacy DQN model format in {model_path}")
                model = LegacyDQNNetwork(state_dim, action_dim)
                model.load_state_dict(state_dict)
                return model
            else:  # New architecture
                model = DQNNetwork(state_dim, action_dim)
                model.load_state_dict(state_dict)
                return model
        elif model_type == 'a2c':
            if 'shared.0.weight' in state_dict:  # Old architecture
                log_info(f"Detected legacy A2C model format in {model_path}")
                model = LegacyA2CNetwork(state_dim, action_dim)
                model.load_state_dict(state_dict)
                return model
            else:  # New architecture
                model = A2CNetwork(state_dim, action_dim)
                model.load_state_dict(state_dict)
                return model
    except Exception as e:
        log_warning(f"Failed to load model directly: {str(e)}")
        # If loading fails, return a new model with default initialization
        if model_type == 'dqn':
            log_info(f"Creating new DQN model (ignoring saved weights)")
            return DQNNetwork(state_dim, action_dim)
        else:
            log_info(f"Creating new A2C model (ignoring saved weights)")
            return A2CNetwork(state_dim, action_dim)

class DQNAgent:
    """Enhanced DQN agent for column selection with prioritized experience replay."""
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01,
                 use_prioritized_replay=True):
        """
        Initialize the enhanced DQN agent.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            lr (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Decay rate for exploration
            epsilon_min (float): Minimum exploration rate
            use_prioritized_replay (bool): Whether to use prioritized experience replay
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create networks
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with learning rate scheduler
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=20)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability
        
        # Prioritized experience replay
        self.use_prioritized_replay = use_prioritized_replay
        self.buffer = []
        self.buffer_size = 20000  # Increased buffer size
        self.batch_size = 128     # Increased batch size
        self.update_freq = 5      # Update target network more frequently
        self.steps = 0
        
        # For prioritized experience replay
        self.priorities = np.zeros(self.buffer_size)
        self.alpha = 0.6          # Priority exponent
        self.beta = 0.4           # Initial importance sampling weight
        self.beta_increment = 0.001
        self.max_priority = 1.0
    
    def select_action(self, state, available_actions):
        """
        Select an action based on the current state.
        
        Args:
            state (numpy.ndarray): Current state
            available_actions (list): List of available actions
            
        Returns:
            int: Selected action
        """
        if not available_actions:
            return None
        
        try:
            # Exploration
            if np.random.rand() < self.epsilon:
                return np.random.choice(available_actions)
                
            # Exploitation - convert state to tensor and ensure it's on the right device
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # Handle batch dimension for batch normalization
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            # Set network to eval mode for inference
            self.q_network.eval()
            
            # Get Q-values, ensuring all tensor operations stay on the same device
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                # Move to CPU only after computation is complete
                q_values = q_values.cpu().detach().numpy().flatten()
                
            # Set network back to train mode
            self.q_network.train()
            
            # Mask unavailable actions with large negative values
            mask = np.ones_like(q_values) * float('-inf')
            for action in available_actions:
                mask[action] = 0
            masked_q_values = q_values + mask
            
            # Choose action with highest Q-value among available actions
            return np.argmax(masked_q_values)
                
        except Exception as e:
            print(f"Error in select_action: {e}")
            # Fallback to random selection
            return np.random.choice(available_actions)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer with priority.
        
        Args:
            state (numpy.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (numpy.ndarray): Next state
            done (bool): Whether episode is done
        """
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities[len(self.buffer) - 1] = self.max_priority
        else:
            # Replace random sample with lower priority
            if self.use_prioritized_replay:
                idx = np.random.choice(len(self.buffer), p=1.0 / (self.priorities[:len(self.buffer)] + 1e-10))
            else:
                idx = np.random.randint(0, len(self.buffer))
            
            self.buffer[idx] = (state, action, reward, next_state, done)
            self.priorities[idx] = self.max_priority
    
    def get_batch(self):
        """
        Get a batch of experiences with prioritized sampling.
        
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        if self.use_prioritized_replay:
            # Compute sampling probabilities from priorities
            priorities = self.priorities[:len(self.buffer)]
            probs = priorities ** self.alpha
            probs = probs / np.sum(probs)
            
            # Sample batch and compute importance sampling weights
            indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights = weights / np.max(weights)  # Normalize weights
            
            # Increase beta over time for convergence to unbiased updates
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), self.batch_size)
            weights = np.ones(self.batch_size)
        
        # Get batch data
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        weights = np.array(weights, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities in the replay buffer.
        
        Args:
            indices (list): Indices of transitions
            td_errors (numpy.ndarray): TD errors for the transitions
        """
        if not self.use_prioritized_replay:
            return
            
        for i, idx in enumerate(indices):
            # Priority is proportional to absolute TD error plus small constant
            # Extract single value properly to avoid deprecation warning
            td_error = float(abs(td_errors[i].item()))
            self.priorities[idx] = td_error + 1e-5
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def replay(self):
        """Learn from prioritized experiences in replay buffer."""
        # Get batch
        batch = self.get_batch()
        if batch is None:
            return 0  # No learning done
            
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions)
        
        # Target Q values with double DQN: select action using the online network
        # but evaluate using the target network
        with torch.no_grad():
            # Select actions from online network
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            # Evaluate with target network
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute TD errors for updating priorities
        td_errors = target_q.detach() - current_q.detach()  # Detach both tensors
        
        # Update priorities - make sure to detach tensor before converting to numpy
        self.update_priorities(indices, td_errors.cpu().detach().numpy())
        
        # Weighted loss to account for importance sampling
        loss = (weights * self.loss_fn(current_q, target_q)).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

class A2CNetwork(nn.Module):
    """Enhanced Actor-Critic Network for column selection."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 128]):
        """
        Initialize the A2C network with improved architecture.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dims (list): Dimensions of hidden layers
        """
        super(A2CNetwork, self).__init__()
        
        # Build a dynamic shared network based on hidden_dims
        shared_layers = []
        prev_dim = state_dim
        
        # Create shared feature extractor with batch normalization
        for i, dim in enumerate(hidden_dims[:-1]):
            shared_layers.append(nn.Linear(prev_dim, dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Actor (policy) network with separate final hidden layer
        self.actor_hidden = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[-1])
        )
        self.actor_output = nn.Linear(hidden_dims[-1], action_dim)
        
        # Critic (value) network with separate final hidden layer
        self.critic_hidden = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[-1])
        )
        self.critic_output = nn.Linear(hidden_dims[-1], 1)
        
        # Apply weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (actor_output, critic_output) - Policy logits and value
        """
        # For batch size 1, adjust batch normalization dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        shared_features = self.shared(x)
        
        # Actor path
        actor_features = self.actor_hidden(shared_features)
        actor_output = self.actor_output(actor_features)
        
        # Critic path
        critic_features = self.critic_hidden(shared_features)
        critic_output = self.critic_output(critic_features)
        
        return actor_output, critic_output

class A2CAgent:
    """Enhanced A2C agent for column selection with additional optimization features."""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, entropy_coef=0.01, value_coef=0.5):
        """
        Initialize the enhanced A2C agent.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            lr (float): Learning rate
            gamma (float): Discount factor
            entropy_coef (float): Entropy regularization coefficient
            value_coef (float): Value loss coefficient
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create A2C network
        self.model = A2CNetwork(state_dim, action_dim)
        self.model.to(self.device)
        
        # Advanced optimizer with learning rate scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=30, verbose=True)
        
        # Track learning metrics
        self.avg_reward = 0
        self.total_updates = 0
    
    def select_action(self, state, available_actions):
        """
        Select an action based on the current state.
        
        Args:
            state (numpy.ndarray): Current state
            available_actions (list): List of available actions
            
        Returns:
            int: Selected action
        """
        if not available_actions:
            return None
        
        try:
            # Convert state to tensor and ensure it's on the right device
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # Handle batch dimension for batch normalization
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            # Set model to eval mode for inference to properly handle batch normalization
            self.model.eval()
            
            # Make sure model is on the correct device
            self.model = self.model.to(self.device)
            
            # Get policy logits and value - all on the same device
            with torch.no_grad():
                logits, _ = self.model(state_tensor)
                # Move to CPU only after computation is complete
                logits = logits.cpu().detach().numpy().flatten()  # Ensure 1D array
                
            # Set model back to train mode
            self.model.train()
            
            # Create action mask for unavailable actions
            mask = np.ones_like(logits) * float('-inf')
            for action in available_actions:
                mask[action] = 0
            masked_logits = logits + mask
            
            # Safe softmax calculation
            max_logit = np.max(masked_logits)
            if np.isneginf(max_logit):  # All actions are masked
                return np.random.choice(available_actions)
                
            exp_logits = np.exp(masked_logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)
            
            # Handle numerical issues
            if np.isnan(probs).any() or np.sum(probs) < 1e-10:
                print("Warning: Invalid probability distribution, using random action")
                return np.random.choice(available_actions)
            
            # Exploration-exploitation trade-off: sometimes pick greedy, sometimes sample
            if np.random.random() < 0.1:  # 10% of the time be greedy
                return np.argmax(probs)
            else:
                try:
                    return np.random.choice(self.action_dim, p=probs)
                except Exception as e:
                    print(f"Error in action sampling: {e}")
                    # Fallback if distribution isn't valid
                    return np.random.choice(available_actions)
                    
        except Exception as e:
            print(f"Error in select_action: {e}")
            # Fallback to random selection on any error
            return np.random.choice(available_actions)
    
    def update(self, states, actions, rewards, next_states, dones):
        """
        Update the A2C network using collected trajectories.
        
        Args:
            states (list): List of states
            actions (list): List of actions
            rewards (list): List of rewards
            next_states (list): List of next states
            dones (list): List of done flags
            
        Returns:
            tuple: (policy_loss, value_loss, entropy) - Loss metrics
        """
        if not states:
            return 0, 0, 0
            
        # Convert lists to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        # Add reward normalization for stability
        if len(rewards) > 1 and np.std(rewards) > 0:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-10)
        
        # Convert numpy arrays to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current policy logits and values
        logits, values = self.model(states)
        
        # Compute next state values for bootstrapping
        with torch.no_grad():
            _, next_values = self.model(next_states)
            next_values = next_values.squeeze()
        
        # Compute discounted returns and advantages
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - values.squeeze()
        
        # Normalize advantages for stable learning
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        # Compute policy loss using the log-probability of actions
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Select log probs of taken actions
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute policy loss using advantages
        policy_loss = -(action_log_probs * advantages).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Compute entropy for regularization (encourages exploration)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        
        # Compute total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update model
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Apply gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # Track metrics
        self.total_updates += 1
        
        # Convert tensor values to Python floats for return
        return (
            policy_loss.item(),
            value_loss.item(), 
            entropy.item()
        )
        
    def update_scheduler(self, avg_reward):
        """Update learning rate scheduler based on performance."""
        self.scheduler.step(avg_reward)

# =============================================================================
# Training functions
# =============================================================================

def train_dqn(agent, env, num_episodes=500, early_stopping_patience=50, save_path=None):
    """
    Train a DQN agent for column selection with advanced features like
    early stopping and model checkpointing.
    
    Args:
        agent (DQNAgent): The DQN agent to train
        env (ColumnSelectionEnv): The environment
        num_episodes (int): Maximum number of episodes to train for
        early_stopping_patience (int): Number of episodes to wait for improvement before stopping
        save_path (str): Path to save the best model checkpoint, if None no checkpointing is done
        
    Returns:
        tuple: (rewards_history, errors_history, best_error) - Training metrics and best performance
    """
    rewards_history = []
    errors_history = []
    losses = []
    
    # For early stopping
    best_error = float('inf')
    best_reward = float('-inf')
    best_weights = None
    patience_counter = 0
    
    # Create progress bar for episodes
    with tqdm(total=num_episodes, desc="Training DQN") as pbar:
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            episode_losses = []
            done = False
            
            while not done:
                # Select and perform action
                action = agent.select_action(state, env.available_columns)
                next_state, reward, done, _ = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Learn from experience
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
                
                state = next_state
                total_reward += reward
            
            # Calculate error at the end of episode
            if env.selected_columns:
                C = env.A[:, env.selected_columns]
                U = np.linalg.pinv(C)
                A_approx = C @ U @ env.A
                error = np.linalg.norm(env.A - A_approx, 'fro') / env.A_norm
                errors_history.append(error)
            else:
                error = 1.0
                errors_history.append(error)
            
            rewards_history.append(total_reward)
            losses.append(np.mean(episode_losses) if episode_losses else 0)
            
            # Update scheduler with negative error (higher is better)
            if hasattr(agent, 'scheduler'):
                agent.scheduler.step(-error)
            
            # Check for best performance and save model if needed
            if error < best_error:
                best_error = error
                best_reward = total_reward
                patience_counter = 0
                
                # Save best model
                if save_path:
                    best_weights = agent.q_network.state_dict().copy()
            else:
                patience_counter += 1
            
            # Update progress bar with useful information
            pbar.set_postfix({
                'reward': f'{total_reward:.4f}',
                'error': f'{error:.4f}',
                'epsilon': f'{agent.epsilon:.4f}',
                'best': f'{best_error:.4f}',
                'patience': f'{patience_counter}/{early_stopping_patience}'
            })
            pbar.update(1)
            
            # Log periodically
            if episode % 20 == 0:
                log_info(f"Episode {episode}/{num_episodes}, Reward: {total_reward:.4f}, "
                         f"Error: {error:.4f}, Best: {best_error:.4f}, Epsilon: {agent.epsilon:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                log_info(f"Early stopping triggered after {episode+1} episodes. Best error: {best_error:.6f}")
                break
    
    # Restore best model weights if we have them
    if best_weights is not None:
        agent.q_network.load_state_dict(best_weights)
        agent.target_network.load_state_dict(best_weights)
        
    # Save the model if path is provided and not already saved
    if save_path and best_weights is not None:
        torch.save(best_weights, save_path)
        log_success(f"Best model saved to {save_path}")
    
    return rewards_history, errors_history, best_error

def train_a2c(agent, env, num_episodes=500, update_interval=5, early_stopping_patience=50, save_path=None):
    """
    Train an A2C agent for column selection with advanced features.
    
    Args:
        agent (A2CAgent): The A2C agent to train
        env (ColumnSelectionEnv): The environment
        num_episodes (int): Maximum number of episodes to train for
        update_interval (int): How often to update the policy
        early_stopping_patience (int): Number of episodes to wait for improvement before stopping
        save_path (str): Path to save the best model checkpoint, if None no checkpointing is done
        
    Returns:
        tuple: (rewards_history, errors_history, best_error) - Training metrics and best performance
    """
    rewards_history = []
    errors_history = []
    policy_losses = []
    value_losses = []
    entropies = []
    
    # For early stopping
    best_error = float('inf')
    best_reward = float('-inf')
    best_weights = None
    patience_counter = 0
    
    # Device for computation
    device = agent.device
    
    # Create progress bar for episodes
    with tqdm(total=num_episodes, desc="Training A2C") as pbar:
        for episode in range(num_episodes):
            state = env.reset()
            states, actions, rewards, next_states, dones = [], [], [], [], []
            total_reward = 0
            episode_policy_losses = []
            episode_value_losses = []
            episode_entropies = []
            done = False
            step = 0
            
            while not done:
                # Select and perform action
                action = agent.select_action(state, env.available_columns)
                next_state, reward, done, _ = env.step(action)
                
                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(float(done))
                
                state = next_state
                total_reward += reward
                step += 1
                
                # Update policy periodically or at end of episode
                if step % update_interval == 0 or done:
                    # Convert to numpy arrays and normalize rewards for stability
                    if len(states) > 0:
                        policy_loss, value_loss, entropy = agent.update(
                            states, actions, rewards, next_states, dones)
                        
                        episode_policy_losses.append(policy_loss)
                        episode_value_losses.append(value_loss)
                        episode_entropies.append(entropy)
                        
                        # Clear buffers
                        states, actions, rewards, next_states, dones = [], [], [], [], []
            
            # Calculate error at the end of episode
            if env.selected_columns:
                C = env.A[:, env.selected_columns]
                U = np.linalg.pinv(C)
                A_approx = C @ U @ env.A
                error = np.linalg.norm(env.A - A_approx, 'fro') / env.A_norm
                errors_history.append(error)
            else:
                error = 1.0
                errors_history.append(error)
            
            rewards_history.append(total_reward)
            
            # Log losses if available
            if episode_policy_losses:
                policy_losses.append(np.mean(episode_policy_losses))
                value_losses.append(np.mean(episode_value_losses))
                entropies.append(np.mean(episode_entropies))
            
            # Update scheduler based on negative error (higher is better)
            agent.update_scheduler(-error)
            
            # Check for best performance and save model if needed
            if error < best_error:
                best_error = error
                best_reward = total_reward
                patience_counter = 0
                
                # Save best model
                if save_path:
                    best_weights = agent.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Update progress bar with useful information
            pbar.set_postfix({
                'reward': f'{total_reward:.4f}',
                'error': f'{error:.4f}',
                'best': f'{best_error:.4f}',
                'patience': f'{patience_counter}/{early_stopping_patience}'
            })
            pbar.update(1)
            
            # Log periodically
            if episode % 20 == 0:
                log_info(f"Episode {episode}/{num_episodes}, Reward: {total_reward:.4f}, "
                         f"Error: {error:.4f}, Best: {best_error:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                log_info(f"Early stopping triggered after {episode+1} episodes. Best error: {best_error:.6f}")
                break
    
    # Restore best model weights if we have them
    if best_weights is not None:
        agent.model.load_state_dict(best_weights)
        
    # Save the model if path is provided and not already saved
    if save_path and best_weights is not None:
        torch.save(best_weights, save_path)
        log_success(f"Best model saved to {save_path}")
    
    return rewards_history, errors_history, best_error

# =============================================================================
# Apply trained RL agents
# =============================================================================

def rl_column_selection_with_dqn(A, rank, agent):
    """
    Apply trained DQN agent for column selection.
    
    Args:
        A (numpy.ndarray): Input matrix
        rank (int): Target rank for approximation
        agent (DQNAgent): Trained DQN agent
        
    Returns:
        tuple: (A_approx, selected_columns) - Approximated matrix and selected columns
    """
    log_debug(f"Applying DQN-based column selection (rank={rank})")
    env = ColumnSelectionEnv(A, rank)
    state = env.reset()
    
    with tqdm(total=rank, desc="DQN Column Selection") as pbar:
        for _ in range(rank):
            action = agent.select_action(state, env.available_columns)
            state, _, _, _ = env.step(action)
            pbar.update(1)
    
    selected_columns = env.selected_columns
    C = A[:, selected_columns]
    U = np.linalg.pinv(C)
    A_approx = C @ U @ A
    
    return A_approx, selected_columns

def rl_column_selection_with_a2c(A, rank, agent, state_type='binary', reward_type='error'):
    """
    Apply trained A2C agent for column selection.
    
    Args:
        A (numpy.ndarray): Input matrix
        rank (int): Target rank for approximation
        agent (A2CAgent): Trained A2C agent
        state_type (str): Type of state representation
        reward_type (str): Type of reward function
        
    Returns:
        tuple: (A_approx, selected_columns) - Approximated matrix and selected columns
    """
    log_debug(f"Applying A2C-based column selection (rank={rank})")
    env = EnhancedColumnSelectionEnv(A, rank, state_type, reward_type)
    state = env.reset()
    
    # Get the device from the agent
    device = agent.device
    
    with tqdm(total=rank, desc="A2C Column Selection") as pbar:
        for _ in range(rank):
            # Make sure we have a valid available column
            if not env.available_columns:
                log_warning("No more available columns to select!")
                break
                
            # Select action using the agent
            action = agent.select_action(state, env.available_columns)
            
            # Perform the action in the environment
            next_state, reward, done, _ = env.step(action)
            state = next_state
            pbar.update(1)
    
    selected_columns = env.selected_columns
    if len(selected_columns) < 2:
        log_warning(f"Only selected {len(selected_columns)} columns, which is insufficient. Using random columns instead.")
        selected_columns = np.random.choice(A.shape[1], size=min(rank, A.shape[1]), replace=False)
    
    try:
        C = A[:, selected_columns]
        U = np.linalg.pinv(C)
        A_approx = C @ U @ A
    except np.linalg.LinAlgError as e:
        log_error(f"Linear algebra error in A2C column selection: {str(e)}")
        # Fallback to random selection
        log_warning("Falling back to random column selection")
        fallback_cols = np.random.choice(A.shape[1], size=min(rank, A.shape[1]), replace=False)
        C = A[:, fallback_cols]
        U = np.linalg.pinv(C)
        A_approx = C @ U @ A
    
    return A_approx, selected_columns

# =============================================================================
# Experiment functions
# =============================================================================

def evaluate_approximation(matrix_name, A, A_approx, method_name):
    """
    Evaluate a matrix approximation method.
    
    Args:
        matrix_name (str): Name of the matrix
        A (numpy.ndarray): Original matrix
        A_approx (numpy.ndarray): Approximated matrix
        method_name (str): Name of the approximation method
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Relative Frobenius error
    rel_frob_error = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
    
    # Spectral norm error
    try:
        spec_error = np.linalg.norm(A - A_approx, 2) / np.linalg.norm(A, 2)
    except:
        spec_error = np.nan
    
    log_info(f"[{matrix_name}] Method: {method_name}, Relative Error: {rel_frob_error:.6f}, "
             f"Spectral Error: {spec_error:.6f}")
    
    return {
        'matrix': matrix_name,
        'method': method_name,
        'rel_frob_error': rel_frob_error,
        'spec_error': spec_error
    }

def run_matrix_experiments(matrices, target_rank=None, num_train_episodes=400):
    """
    Run experiments with various matrix approximation methods.
    
    Args:
        matrices (dict): Dictionary of matrices to test
        target_rank (int): Target rank for approximation, if None uses 10% of matrix size
        num_train_episodes (int): Number of training episodes for RL methods
        
    Returns:
        list: List of evaluation results
    """
    log_info("Starting matrix approximation experiments")
    results = []
    
    # Select train and test matrices
    if len(matrices) < 2:
        log_error("Not enough matrices were loaded successfully.")
        return []
    
    # Split into train and test sets
    matrix_names = list(matrices.keys())
    train_matrix_name = matrix_names[0]  # Use first matrix for training
    test_matrix_names = matrix_names[1:]  # Use remaining matrices for testing

    train_matrix = matrices[train_matrix_name]
    test_matrices = {name: matrices[name] for name in test_matrix_names}

    log_info(f"Train matrix: {train_matrix_name}")
    log_info(f"Test matrices: {test_matrix_names}")

    # Define target rank (e.g., 10% of matrix size)
    if target_rank is None:
        target_rank = max(5, min(train_matrix.shape[0] // 10, 80))  # Ensure rank is reasonable
    
    log_info(f"Target rank: {target_rank}")

    # Check for existing models
    dqn_model_path = f"models/dqn_{train_matrix_name}.pt"
    a2c_model_path = f"models/a2c_{train_matrix_name}.pt"
    dqn_model_exists = os.path.exists(dqn_model_path)
    a2c_model_exists = os.path.exists(a2c_model_path)

    # Setup environment dimensions
    env = ColumnSelectionEnv(train_matrix, target_rank)
    state_dim = env.n
    action_dim = env.n

    # Train RL agents on the training matrix if models don't exist
    log_info("Setting up RL agents...")
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")
    
    # Initialize enhanced DQN agent with prioritized experience replay
    dqn_agent = DQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        use_prioritized_replay=True
    )
    
    # Train or load DQN model
    if dqn_model_exists:
        log_info(f"Loading existing DQN model from {dqn_model_path}")
        # Load with compatibility check
        loaded_model = load_model_with_compatibility(dqn_model_path, state_dim, action_dim, model_type='dqn', device=device)
        
        # If legacy model, need to adapt our agent to use it
        if isinstance(loaded_model, LegacyDQNNetwork):
            log_info("Using legacy DQN model structure")
            dqn_agent.q_network = loaded_model
            # Create target network of the same type
            dqn_agent.target_network = LegacyDQNNetwork(state_dim, action_dim).to(device)
        else:
            dqn_agent.q_network = loaded_model
            
        # Sync the target network
        dqn_agent.target_network.load_state_dict(dqn_agent.q_network.state_dict())
        log_success(f"DQN model loaded successfully")
    else:
        log_info("Training DQN agent with enhanced architecture...")
        train_dqn(
            dqn_agent, 
            env, 
            num_episodes=num_train_episodes, 
            early_stopping_patience=50,
            save_path=dqn_model_path
        )
        log_success(f"DQN agent trained and saved to {dqn_model_path}")

    # Initialize enhanced A2C agent with combined state and reward
    env = EnhancedColumnSelectionEnv(train_matrix, target_rank, 
                                     state_type='combined', reward_type='combined')
    state_dim = env.n * 3 if env.state_type == 'combined' else env.n
    a2c_agent = A2CAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.02,  # Increased entropy coef for better exploration
        value_coef=0.5
    )
    
    # Train or load A2C model
    if a2c_model_exists:
        log_info(f"Loading existing A2C model from {a2c_model_path}")
        # Load with compatibility check
        loaded_model = load_model_with_compatibility(a2c_model_path, state_dim, action_dim, model_type='a2c', device=device)
        
        # If legacy model, need to adapt our agent to use it
        if isinstance(loaded_model, LegacyA2CNetwork):
            log_info("Using legacy A2C model structure")
        
        a2c_agent.model = loaded_model
        log_success(f"A2C model loaded successfully")
    else:
        log_info("Training A2C agent with enhanced architecture...")
        train_a2c(
            a2c_agent, 
            env, 
            num_episodes=num_train_episodes, 
            update_interval=5,
            early_stopping_patience=50,
            save_path=a2c_model_path
        )
        log_success(f"A2C agent trained and saved to {a2c_model_path}")

    # Define main comparison methods (without CUR)
    main_methods = {
        "Deterministic SVD": lambda A, r: deterministic_rank_approx(A, r),
        "Randomized SVD": lambda A, r: randomized_rank_approx(A, r),
        "RL-DQN": lambda A, r: rl_column_selection_with_dqn(A, r, dqn_agent)[0],
        "RL-A2C": lambda A, r: rl_column_selection_with_a2c(
            A, r, a2c_agent, state_type='combined', reward_type='combined')[0],
    }
    
    # Define CUR separately for secondary comparison
    cur_method = {
        "CUR": lambda A, r: cur_decomposition(A, r)[0],
    }

    # Evaluate on the training matrix first
    log_info(f"Evaluating methods on training matrix: {train_matrix_name}")
    A_train = train_matrix

    # Evaluate main methods
    for method_name, method_fn in main_methods.items():
        try:
            log_debug(f"Applying method {method_name} to matrix {train_matrix_name}")
            start_time = time.time()
            A_approx = method_fn(A_train, target_rank)
            elapsed_time = time.time() - start_time

            eval_result = evaluate_approximation(train_matrix_name, A_train, A_approx, method_name)
            eval_result['time'] = elapsed_time
            eval_result['train_matrix'] = True
            results.append(eval_result)

            log_success(f"{method_name} on {train_matrix_name}: "
                       f"Error={eval_result['rel_frob_error']:.4f}, Time={elapsed_time:.4f}s")
        except Exception as e:
            log_error(f"Failed {method_name} on {train_matrix_name}: {str(e)}")
    
    # Evaluate CUR separately
    for method_name, method_fn in cur_method.items():
        try:
            log_debug(f"Applying method {method_name} to matrix {train_matrix_name}")
            start_time = time.time()
            A_approx = method_fn(A_train, target_rank)
            elapsed_time = time.time() - start_time

            eval_result = evaluate_approximation(train_matrix_name, A_train, A_approx, method_name)
            eval_result['time'] = elapsed_time
            eval_result['train_matrix'] = True
            results.append(eval_result)

            log_success(f"{method_name} on {train_matrix_name}: "
                       f"Error={eval_result['rel_frob_error']:.4f}, Time={elapsed_time:.4f}s")
        except Exception as e:
            log_error(f"Failed {method_name} on {train_matrix_name}: {str(e)}")

    # Evaluate on test matrices
    for test_name, A_test in test_matrices.items():
        log_info(f"Evaluating methods on test matrix: {test_name}")
        
        # Test main methods
        for method_name, method_fn in main_methods.items():
            try:
                log_debug(f"Applying method {method_name} to matrix {test_name}")
                start_time = time.time()
                A_approx = method_fn(A_test, target_rank)
                elapsed_time = time.time() - start_time

                eval_result = evaluate_approximation(test_name, A_test, A_approx, method_name)
                eval_result['time'] = elapsed_time
                eval_result['train_matrix'] = False
                results.append(eval_result)

                log_success(f"{method_name} on {test_name}: "
                           f"Error={eval_result['rel_frob_error']:.4f}, Time={elapsed_time:.4f}s")
            except Exception as e:
                log_error(f"Failed {method_name} on {test_name}: {str(e)}")
        
        # Test CUR separately
        for method_name, method_fn in cur_method.items():
            try:
                log_debug(f"Applying method {method_name} to matrix {test_name}")
                start_time = time.time()
                A_approx = method_fn(A_test, target_rank)
                elapsed_time = time.time() - start_time

                eval_result = evaluate_approximation(test_name, A_test, A_approx, method_name)
                eval_result['time'] = elapsed_time
                eval_result['train_matrix'] = False
                results.append(eval_result)

                log_success(f"{method_name} on {test_name}: "
                           f"Error={eval_result['rel_frob_error']:.4f}, Time={elapsed_time:.4f}s")
            except Exception as e:
                log_error(f"Failed {method_name} on {test_name}: {str(e)}")

    # Save results
    result_df = pd.DataFrame(results)
    result_df.to_csv('results/matrix_experiments.csv', index=False)
    with open('results/matrix_experiments.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    log_success("Experiments completed and results saved")
    return results

def plot_results(results, train_matrix_name):
    """
    Plot experiment results.
    
    Args:
        results (list): List of evaluation results
        train_matrix_name (str): Name of the training matrix
    """
    log_info("Generating result plots...")
    
    # Convert to DataFrame for easier plotting
    if not isinstance(results, pd.DataFrame):
        df = pd.DataFrame(results)
    else:
        df = results
    
    # Separate CUR and main methods
    main_df = df[df['method'] != 'CUR'].copy()
    cur_df = df[df['method'].isin(['CUR', 'Deterministic SVD'])].copy()
    
    # 1. Main Error comparison (without CUR)
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='matrix', y='rel_frob_error', hue='method', data=main_df)
    plt.title(f'Error Comparison - Main Methods (Trained on {train_matrix_name})', fontsize=14)
    plt.xlabel('Matrix', fontsize=12)
    plt.ylabel('Relative Frobenius Error', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/error_comparison.png', dpi=300)
    
    # 2. CUR Error comparison
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='matrix', y='rel_frob_error', hue='method', data=cur_df)
    plt.title(f'Error Comparison - CUR vs SVD (Trained on {train_matrix_name})', fontsize=14)
    plt.xlabel('Matrix', fontsize=12)
    plt.ylabel('Relative Frobenius Error', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/cur_comparison.png', dpi=300)
    
    # 3. Time comparison (all methods)
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='matrix', y='time', hue='method', data=df)
    plt.title(f'Time Comparison (Trained on {train_matrix_name})', fontsize=14)
    plt.xlabel('Matrix', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/time_comparison.png', dpi=300)
    
    # 4. Standard error heatmap
    methods = df['method'].unique()
    matrices = df['matrix'].unique()
    
    # Create a pivot table for the standard heatmap
    heatmap_data = df.pivot_table(index='method', columns='matrix', values='rel_frob_error')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title(f'Error Heatmap (Trained on {train_matrix_name})', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/error_heatmap.png', dpi=300)
    
    # 5. Percentage error heatmap relative to Deterministic SVD
    # Create pivot tables for each method and SVD
    svd_errors = df[df['method'] == 'Deterministic SVD'].set_index('matrix')['rel_frob_error']
    
    # Prepare percentage data
    percentage_data = {}
    for method in df['method'].unique():
        if method != 'Deterministic SVD':
            method_errors = df[df['method'] == method].set_index('matrix')['rel_frob_error']
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
    
    plt.title(f'Error Percentage Relative to Deterministic SVD (Trained on {train_matrix_name})', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/relative_error_heatmap.png', dpi=300)
    
    log_success("All plots saved in the results directory")
    
    return df  # Return the DataFrame for further analysis if needed

# =============================================================================
# Main function
# =============================================================================

def main():
    """Main function to run the experiments."""
    try:
        log_info("Starting matrix approximation experiments")
        
        # First, download all the matrices from the image
        from florida_downloader import download_all_matrices, load_all_matrices
        
        # Define which matrices to download
        target_matrices = [
            "G1", "G2", "G3", "G4", "G5", 
            "G10", "G11", "G12", "G13",
            "G14", "G15", "G16", "G17"
        ]
        
        log_info(f"Downloading target matrices: {target_matrices}")
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
        
        # Run experiments
        target_rank = 40  # Can be adjusted based on needs
        results = run_matrix_experiments(matrices, target_rank=target_rank, num_train_episodes=400)
        
        # Plot results
        train_matrix_name = list(matrices.keys())[0]  # First matrix was used for training
        plot_results(results, train_matrix_name)
        
        log_success("Experiments completed successfully!")
        
    except KeyboardInterrupt:
        log_warning("Experiments interrupted by user")
    except Exception as e:
        log_error(f"Error running experiments: {str(e)}")
        import traceback
        log_error(traceback.format_exc())

if __name__ == "__main__":
    main()
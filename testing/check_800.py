import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import scipy.io as sio
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init(autoreset=True)

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Constants
BASE_URL = "https://sparse.tamu.edu/mat"

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

# Helper functions for matrix operations

def download_matrix_from_ssmc(matrix_id, matrix_name, group):
    """Download a matrix from SuiteSparse Matrix Collection."""
    print(f"Downloading matrix {matrix_name} (ID: {matrix_id})...")
    
    url = f"{BASE_URL}/{group}/{matrix_name}.mat"
    local_path = f"data/{matrix_name}.mat"
    
    # Skip download if the file already exists
    if os.path.exists(local_path):
        print(f"Matrix file already exists: {local_path}")
        return local_path
    
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded matrix to {local_path}")
            return local_path
        else:
            print(f"Failed to download matrix: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading matrix: {str(e)}")
        return None

def load_florida_matrix(file_path):
    """Load a matrix from the SuiteSparse Matrix Collection."""
    try:
        mat_data = sio.loadmat(file_path)
        
        # Look for the matrix in the mat file
        # SuiteSparse matrices are typically stored with the name 'Problem'
        if 'Problem' in mat_data:
            problem = mat_data['Problem']
            # Extract the matrix (typically named 'A')
            if hasattr(problem, 'keys') and 'A' in problem:
                matrix = problem['A']
            elif isinstance(problem, np.ndarray) and problem.dtype == np.object:
                # Sometimes the matrix is in a different structure
                for item in problem[0]:
                    if isinstance(item, np.ndarray) and item.shape:
                        matrix = item
                        break
            else:
                # Try alternative formats
                for key in mat_data:
                    if key not in ['__header__', '__version__', '__globals__']:
                        matrix = mat_data[key]
                        break
        else:
            # Try alternative formats
            for key in mat_data:
                if key not in ['__header__', '__version__', '__globals__']:
                    matrix = mat_data[key]
                    break
        
        # Convert to dense if it's sparse
        if sp.issparse(matrix):
            matrix = matrix.toarray()
        
        return matrix
    except Exception as e:
        print(f"Error loading matrix from {file_path}: {str(e)}")
        raise

def preprocess_florida_matrix(A):
    """Preprocess the matrix for experiments."""
    # Handle NaN and Inf values
    A = np.nan_to_num(A)
    
    # If the matrix has extremely large values, normalize it
    max_val = np.max(np.abs(A))
    if max_val > 1e5:
        A = A / max_val
    
    return A

# Matrix approximation methods

def deterministic_rank_approx(A, rank):
    """Compute a deterministic low-rank approximation using SVD."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]

def randomized_rank_approx(A, rank, n_oversample=10, n_iter=7):
    """Compute a randomized low-rank approximation."""
    m, n = A.shape
    rank_target = min(rank + n_oversample, min(m, n))
    
    # Random projection
    Q = np.random.normal(size=(n, rank_target))
    
    # Power iteration
    for _ in range(n_iter):
        Q = A @ (A.T @ Q)
        Q, _ = np.linalg.qr(Q, mode='reduced')
    
    # Project and compute SVD
    B = Q.T @ A
    U_B, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_B
    
    return U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]

def cur_decomposition(A, rank, c_factor=5):
    """Compute a CUR decomposition for low-rank approximation."""
    m, n = A.shape
    
    # Compute column norms for sampling probabilities
    col_norms = np.sum(A**2, axis=0)
    col_probs = col_norms / np.sum(col_norms)
    
    # Sample columns
    c_indices = np.random.choice(n, size=rank*c_factor, replace=False, p=col_probs)
    C = A[:, c_indices]
    
    # Compute row norms for sampling probabilities using C
    row_norms = np.sum(C**2, axis=1)
    row_probs = row_norms / np.sum(row_norms)
    
    # Sample rows
    r_indices = np.random.choice(m, size=rank*c_factor, replace=False, p=row_probs)
    R = A[r_indices, :]
    
    # Compute intersection
    U = np.linalg.pinv(C[r_indices, :]) @ A[r_indices, :] @ np.linalg.pinv(R[:, c_indices]).T
    
    # Compute approximation
    A_approx = C @ U @ R
    
    return A_approx, C, U, R

# RL Environment for Column Selection

class ColumnSelectionEnv:
    """Environment for RL-based column selection."""
    def __init__(self, A, target_rank):
        self.A = A
        self.A_norm = np.linalg.norm(A, 'fro')
        self.m, self.n = A.shape
        self.target_rank = target_rank
        self.selected_columns = []
        self.available_columns = list(range(self.n))
        
    def reset(self):
        self.selected_columns = []
        self.available_columns = list(range(self.n))
        return self._get_state()
    
    def _get_state(self):
        # Binary vector indicating which columns have been selected
        state = np.zeros(self.n)
        state[self.selected_columns] = 1
        return state
    
    def step(self, action):
        if action in self.available_columns:
            self.selected_columns.append(action)
            self.available_columns.remove(action)
            
            # Calculate current approximation error
            if len(self.selected_columns) >= 2:  # Need at least 2 columns for meaningful approximation
                C = self.A[:, self.selected_columns]
                U = np.linalg.pinv(C)
                A_approx = C @ U @ self.A
                error = np.linalg.norm(self.A - A_approx, 'fro') / self.A_norm
                reward = -error  # Negative error as reward
            else:
                reward = 0
            
            done = len(self.selected_columns) >= self.target_rank
            
            return self._get_state(), reward, done, {}
        else:
            return self._get_state(), -1, False, {}  # Penalty for invalid action

class EnhancedColumnSelectionEnv(ColumnSelectionEnv):
    """Enhanced environment with different state and reward options."""
    def __init__(self, A, target_rank, state_type='binary', reward_type='error'):
        super().__init__(A, target_rank)
        self.state_type = state_type
        self.reward_type = reward_type
        
        # Pre-compute column leverage scores for state representation
        U, _, _ = np.linalg.svd(A, full_matrices=False)
        self.leverage_scores = np.sum(U[:, :min(A.shape)]**2, axis=1)
    
    def _get_state(self):
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
            return self._get_state(), -1, False, {}

# RL Agents

class DQNNetwork(nn.Module):
    """Deep Q-Network for column selection."""
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """DQN agent for column selection."""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay buffer
        self.buffer = []
        self.buffer_size = 10000
        self.batch_size = 64
        self.update_freq = 10
        self.steps = 0
    
    def select_action(self, state, available_actions):
        if not available_actions:
            return None
        
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor).detach().numpy()[0]
            
            # Mask unavailable actions
            q_values_masked = np.ones_like(q_values) * float('-inf')
            q_values_masked[available_actions] = q_values[available_actions]
            
            return np.argmax(q_values_masked)
    
    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def replay(self):
        if len(self.buffer) < self.batch_size:
            return
        
        batch = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions)
        
        # Target Q values
        next_q = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Update network
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class A2CNetwork(nn.Module):
    """Actor-Critic Network for column selection."""
    def __init__(self, state_dim, action_dim):
        super(A2CNetwork, self).__init__()
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

class A2CAgent:
    """A2C agent for column selection."""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.model = A2CNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def select_action(self, state, available_actions):
        if not available_actions:
            return None
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, _ = self.model(state_tensor)
        logits = logits.detach().numpy()[0]
        
        # Mask unavailable actions with large negative values
        mask = np.ones_like(logits) * float('-inf')
        mask[available_actions] = 0
        logits = logits + mask
        
        # Apply softmax to get probabilities
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        
        # Handle numerical issues
        if np.isnan(probs).any() or np.sum(probs) == 0:
            return np.random.choice(available_actions)
        
        try:
            action = np.random.choice(self.action_dim, p=probs)
            return action
        except:
            return np.random.choice(available_actions)
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Get current policy logits and values
        logits, values = self.model(states)
        
        # Get next state values for bootstrapping
        _, next_values = self.model(next_states)
        next_values = next_values.squeeze()
        
        # Calculate advantages and returns
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - values.squeeze()
        
        # Policy loss
        policy_loss = 0
        for i, action in enumerate(actions):
            policy_loss -= logits[i, action] * advantages[i]
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy regularization
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Update model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.item()

# Training functions

def train_dqn(agent, env, num_episodes=300):
    """Train a DQN agent for column selection."""
    rewards_history = []
    errors_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select and perform action
            action = agent.select_action(state, env.available_columns)
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Learn from experience
            agent.replay()
            
            state = next_state
            total_reward += reward
        
        # Track progress
        if env.selected_columns:
            C = env.A[:, env.selected_columns]
            U = np.linalg.pinv(C)
            A_approx = C @ U @ env.A
            error = np.linalg.norm(env.A - A_approx, 'fro') / env.A_norm
            errors_history.append(error)
        else:
            errors_history.append(1.0)
        
        rewards_history.append(total_reward)
        
        if episode % 20 == 0:
            print(f"Episode {episode}/{num_episodes}, Reward: {total_reward:.4f}, Error: {errors_history[-1]:.4f}, Epsilon: {agent.epsilon:.4f}")
    
    return rewards_history, errors_history

def train_a2c(agent, env, num_episodes=300, update_interval=5):
    """Train an A2C agent for column selection."""
    rewards_history = []
    errors_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []
        total_reward = 0
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
            
            # Update policy periodically
            if step % update_interval == 0 or done:
                agent.update(states, actions, rewards, next_states, dones)
                states, actions, rewards, next_states, dones = [], [], [], [], []
        
        # Track progress
        if env.selected_columns:
            C = env.A[:, env.selected_columns]
            U = np.linalg.pinv(C)
            A_approx = C @ U @ env.A
            error = np.linalg.norm(env.A - A_approx, 'fro') / env.A_norm
            errors_history.append(error)
        else:
            errors_history.append(1.0)
        
        rewards_history.append(total_reward)
        
        if episode % 20 == 0:
            print(f"Episode {episode}/{num_episodes}, Reward: {total_reward:.4f}, Error: {errors_history[-1]:.4f}")
    
    return rewards_history, errors_history

# Apply trained RL agents

def rl_column_selection_with_dqn(A, rank, agent):
    """Apply trained DQN agent for column selection."""
    m, n = A.shape
    env = ColumnSelectionEnv(A, rank)
    state = env.reset()
    
    for _ in range(rank):
        action = agent.select_action(state, env.available_columns)
        state, _, _, _ = env.step(action)
    
    selected_columns = env.selected_columns
    C = A[:, selected_columns]
    U = np.linalg.pinv(C)
    A_approx = C @ U @ A
    
    return A_approx, selected_columns

def rl_column_selection_with_a2c(A, rank, agent, state_type='binary', reward_type='error'):
    """Apply trained A2C agent for column selection."""
    m, n = A.shape
    env = EnhancedColumnSelectionEnv(A, rank, state_type, reward_type)
    state = env.reset()
    
    for _ in range(rank):
        action = agent.select_action(state, env.available_columns)
        state, _, _, _ = env.step(action)
    
    selected_columns = env.selected_columns
    C = A[:, selected_columns]
    U = np.linalg.pinv(C)
    A_approx = C @ U @ A
    
    return A_approx, selected_columns

# Main experiment function for Florida matrices

def florida_matrix_experiments():
    """Run experiments with Florida matrices from SuiteSparse."""
    # Define matrices from the table in the image
    matrix_info_list = [
        {"id": 469, "name": "G1", "group": "Gset", "size": (800, 800), "type": "Undirected Random Graph"},
        {"id": 470, "name": "G10", "group": "Gset", "size": (800, 800), "type": "Undirected Weighted Random Graph"},
        {"id": 471, "name": "G11", "group": "Gset", "size": (800, 800), "type": "Undirected Weighted Random Graph"},
        {"id": 472, "name": "G12", "group": "Gset", "size": (800, 800), "type": "Undirected Weighted Random Graph"},
        {"id": 473, "name": "G13", "group": "Gset", "size": (800, 800), "type": "Undirected Weighted Random Graph"},
        {"id": 474, "name": "G14", "group": "Gset", "size": (800, 800), "type": "Duplicate Undirected Random Graph"},
        {"id": 475, "name": "G15", "group": "Gset", "size": (800, 800), "type": "Duplicate Undirected Random Graph"},
        {"id": 476, "name": "G16", "group": "Gset", "size": (800, 800), "type": "Duplicate Undirected Random Graph"},
        {"id": 477, "name": "G17", "group": "Gset", "size": (800, 800), "type": "Duplicate Undirected Random Graph"},
        {"id": 478, "name": "G18", "group": "Gset", "size": (800, 800), "type": "Undirected Weighted Random Graph"},
        {"id": 479, "name": "G19", "group": "Gset", "size": (800, 800), "type": "Undirected Weighted Random Graph"},
        {"id": 480, "name": "G2", "group": "Gset", "size": (800, 800), "type": "Undirected Random Graph"},
        {"id": 481, "name": "G20", "group": "Gset", "size": (800, 800), "type": "Undirected Weighted Random Graph"},
        {"id": 482, "name": "G21", "group": "Gset", "size": (800, 800), "type": "Undirected Weighted Random Graph"},
        {"id": 491, "name": "G3", "group": "Gset", "size": (800, 800), "type": "Undirected Random Graph"},
        {"id": 502, "name": "G4", "group": "Gset", "size": (800, 800), "type": "Undirected Random Graph"},
        {"id": 513, "name": "G5", "group": "Gset", "size": (800, 800), "type": "Undirected Random Graph"},
        {"id": 524, "name": "G6", "group": "Gset", "size": (800, 800), "type": "Undirected Weighted Random Graph"},
        {"id": 533, "name": "G7", "group": "Gset", "size": (800, 800), "type": "Undirected Random Graph"},
        {"id": 534, "name": "G8", "group": "Gset", "size": (800, 800), "type": "Undirected Random Graph"},
        {"id": 535, "name": "G9", "group": "Gset", "size": (800, 800), "type": "Undirected Random Graph"}
    ]

    # Download and process matrices
    # Choose a subset for faster testing
    selected_matrices = matrix_info_list[:5]  # You can adjust this for testing
    
    # Download all matrices
    matrices = {}
    for matrix_info in selected_matrices:
        matrix_id = matrix_info["id"]
        matrix_name = matrix_info["name"]
        group = matrix_info["group"]

        print(f"Processing matrix {matrix_name}...")
        matrix_path = download_matrix_from_ssmc(matrix_id, matrix_name, group)

        if matrix_path:
            try:
                A = load_florida_matrix(matrix_path)
                A = preprocess_florida_matrix(A)
                matrices[matrix_name] = A
                print(f"Loaded {matrix_name} ({A.shape})")
            except Exception as e:
                print(f"Error loading {matrix_name}: {str(e)}")
    
    # Select train and test matrices
    if len(matrices) < 2:
        print("Error: Not enough matrices were loaded successfully.")
        return
    
    # Split into train and test sets
    matrix_names = list(matrices.keys())
    train_matrix_name = matrix_names[0]  # Use first matrix for training
    test_matrix_names = matrix_names[1:]  # Use remaining matrices for testing

    train_matrix = matrices[train_matrix_name]
    test_matrices = {name: matrices[name] for name in test_matrix_names}

    print(f"\nTrain matrix: {train_matrix_name}")
    print(f"Test matrices: {test_matrix_names}")

    # Define target rank (e.g., 10% of matrix size)
    target_rank = max(5, min(train_matrix.shape[0] // 10, 80))  # Ensure rank is reasonable
    print(f"Target rank: {target_rank}")

    # Train RL agents on the training matrix
    print("\nTraining RL agents...")
    
    # Train DQN agent
    env = ColumnSelectionEnv(train_matrix, target_rank)
    state_dim = env.n
    action_dim = env.n
    dqn_agent = DQNAgent(state_dim, action_dim)
    
    print("Training DQN agent...")
    train_dqn(dqn_agent, env, num_episodes=200)
    torch.save(dqn_agent.q_network.state_dict(), f"models/dqn_{train_matrix_name}.pt")

    # Train A2C agent
    env = EnhancedColumnSelectionEnv(train_matrix, target_rank, state_type='combined', reward_type='combined')
    state_dim = env.n * 3 if env.state_type == 'combined' else env.n
    a2c_agent = A2CAgent(state_dim, action_dim)
    
    print("Training A2C agent...")
    train_a2c(a2c_agent, env, num_episodes=200)
    torch.save(a2c_agent.model.state_dict(), f"models/a2c_{train_matrix_name}.pt")

    # Define baseline methods
    base_methods = {
        "Deterministic SVD": lambda A, r: deterministic_rank_approx(A, r),
        "Randomized SVD": lambda A, r: randomized_rank_approx(A, r),
        "CUR": lambda A, r: cur_decomposition(A, r)[0],
    }

    # Define RL methods
    rl_methods = {
        "RL-DQN": lambda A, r: rl_column_selection_with_dqn(A, r, dqn_agent)[0],
        "RL-A2C": lambda A, r: rl_column_selection_with_a2c(A, r, a2c_agent, state_type='combined', reward_type='combined')[0],
    }

    # Evaluate on test matrices
    results = {}
    print("\nEvaluating methods on test matrices...")

    for test_name, A_test in test_matrices.items():
        results[test_name] = {}
        print(f"\nTesting on {test_name}")

        # Test all methods
        for method_name, method_fn in {**base_methods, **rl_methods}.items():
            try:
                start_time = time.time()
                Ar = method_fn(A_test, target_rank)
                rel_error = np.linalg.norm(A_test - Ar, 'fro') / np.linalg.norm(A_test, 'fro')
                elapsed_time = time.time() - start_time

                results[test_name][method_name] = {
                    'error': rel_error,
                    'time': elapsed_time
                }
                print(f"{method_name}: Error={rel_error:.4f}, Time={elapsed_time:.4f}s")
            except Exception as e:
                print(f"Failed {method_name} on {test_name}: {str(e)}")
                results[test_name][method_name] = {'error': np.nan, 'time': np.nan}

    # Save and plot results
    with open('results/florida_matrix_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    plot_results(results, train_matrix_name)
    return results

def plot_results(results, train_matrix_name):
    """Plot results for matrix experiments."""
    test_names = list(results.keys())
    if not test_names:
        print("No results to plot.")
        return
        
    methods = list(results[test_names[0]].keys())

    # Prepare error and time data
    error_data = []
    time_data = []

    for test_name in test_names:
        for method in methods:
            if method in results[test_name]:
                error_data.append({
                    'Matrix': test_name,
                    'Method': method,
                    'Error': results[test_name][method]['error']
                })
                time_data.append({
                    'Matrix': test_name,
                    'Method': method,
                    'Time (s)': results[test_name][method]['time']
                })

    # Create DataFrames
    error_df = pd.DataFrame(error_data)
    time_df = pd.DataFrame(time_data)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Matrix', y='Error', hue='Method', data=error_df)
    plt.title(f'Error Comparison (Trained on {train_matrix_name})')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('results/florida_matrix_error.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Matrix', y='Time (s)', hue='Method', data=time_df)
    plt.title(f'Time Comparison (Trained on {train_matrix_name})')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('results/florida_matrix_time.png', dpi=300)
    plt.close()

    # Create heatmap for relative performance
    print("\nCreating heatmap of relative performance...")
    method_matrix_errors = {}
    for method in methods:
        method_errors = []
        for test_name in test_names:
            if method in results[test_name] and not np.isnan(results[test_name][method]['error']):
                method_errors.append((test_name, results[test_name][method]['error']))
        if method_errors:
            method_matrix_errors[method] = dict(method_errors)
    
    # Create a DataFrame for the heatmap
    heatmap_data = pd.DataFrame(method_matrix_errors).T
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title(f'Error Heatmap (Trained on {train_matrix_name})')
    plt.tight_layout()
    plt.savefig('results/florida_matrix_heatmap.png', dpi=300)
    plt.close()

# Function to compare different Florida matrix families
def compare_florida_matrix_families():
    """Compare RL-based methods across different Florida matrix families."""
    # Define matrix families to compare
    matrix_families = {
        "Random Graphs": ["G1", "G2", "G3", "G4", "G5"],
        "Weighted Random Graphs": ["G10", "G11", "G12", "G13"],
        "Duplicate Random Graphs": ["G14", "G15", "G16", "G17"]
    }
    
    # Define matrix info for all matrices
    all_matrix_info = {
        "G1": {"id": 469, "group": "Gset"},
        "G2": {"id": 480, "group": "Gset"},
        "G3": {"id": 491, "group": "Gset"},
        "G4": {"id": 502, "group": "Gset"},
        "G5": {"id": 513, "group": "Gset"},
        "G10": {"id": 470, "group": "Gset"},
        "G11": {"id": 471, "group": "Gset"},
        "G12": {"id": 472, "group": "Gset"},
        "G13": {"id": 473, "group": "Gset"},
        "G14": {"id": 474, "group": "Gset"},
        "G15": {"id": 475, "group": "Gset"},
        "G16": {"id": 476, "group": "Gset"},
        "G17": {"id": 477, "group": "Gset"}
    }
    
    # Download and process matrices
    matrices = {}
    for matrix_name, info in all_matrix_info.items():
        matrix_id = info["id"]
        group = info["group"]
        
        print(f"Processing matrix {matrix_name}...")
        matrix_path = download_matrix_from_ssmc(matrix_id, matrix_name, group)
        
        if matrix_path:
            try:
                A = load_florida_matrix(matrix_path)
                A = preprocess_florida_matrix(A)
                matrices[matrix_name] = A
                print(f"Loaded {matrix_name} ({A.shape})")
            except Exception as e:
                print(f"Error loading {matrix_name}: {str(e)}")
    
    # Define target rank
    target_rank = 80  # Fixed rank for all experiments
    
    # Results storage
    family_results = {}
    
    # For each family, train on one matrix and test on others
    for family_name, family_matrices in matrix_families.items():
        if not all(name in matrices for name in family_matrices):
            print(f"Skipping family {family_name} due to missing matrices")
            continue
            
        print(f"\n== Processing family: {family_name} ==")
        family_results[family_name] = {}
        
        # Use first matrix in family as training matrix
        train_matrix_name = family_matrices[0]
        train_matrix = matrices[train_matrix_name]
        test_matrix_names = family_matrices[1:]
        
        print(f"Training on {train_matrix_name}")
        
        # Train DQN agent
        env = ColumnSelectionEnv(train_matrix, target_rank)
        state_dim = env.n
        action_dim = env.n
        dqn_agent = DQNAgent(state_dim, action_dim)
        train_dqn(dqn_agent, env, num_episodes=200)
        
        # Train A2C agent
        env = EnhancedColumnSelectionEnv(train_matrix, target_rank, state_type='combined', reward_type='combined')
        state_dim = env.n * 3 if env.state_type == 'combined' else env.n
        a2c_agent = A2CAgent(state_dim, action_dim)
        train_a2c(a2c_agent, env, num_episodes=200)
        
        # Define methods
        methods = {
            "Deterministic SVD": lambda A, r: deterministic_rank_approx(A, r),
            "Randomized SVD": lambda A, r: randomized_rank_approx(A, r),
            "CUR": lambda A, r: cur_decomposition(A, r)[0],
            "RL-DQN": lambda A, r: rl_column_selection_with_dqn(A, r, dqn_agent)[0],
            "RL-A2C": lambda A, r: rl_column_selection_with_a2c(A, r, a2c_agent, state_type='combined', reward_type='combined')[0],
        }
        
        # Test on all matrices in the family
        for test_name in [train_matrix_name] + test_matrix_names:
            test_matrix = matrices[test_name]
            family_results[family_name][test_name] = {}
            
            print(f"Testing on {test_name}")
            for method_name, method_fn in methods.items():
                try:
                    start_time = time.time()
                    Ar = method_fn(test_matrix, target_rank)
                    rel_error = np.linalg.norm(test_matrix - Ar, 'fro') / np.linalg.norm(test_matrix, 'fro')
                    elapsed_time = time.time() - start_time
                    
                    family_results[family_name][test_name][method_name] = {
                        'error': rel_error,
                        'time': elapsed_time
                    }
                    print(f"{method_name}: Error={rel_error:.4f}, Time={elapsed_time:.4f}s")
                except Exception as e:
                    print(f"Failed {method_name} on {test_name}: {str(e)}")
                    family_results[family_name][test_name][method_name] = {'error': np.nan, 'time': np.nan}
    
    # Save results
    with open('results/family_comparison_results.pkl', 'wb') as f:
        pickle.dump(family_results, f)
    
    plot_family_comparison(family_results)
    return family_results

def plot_family_comparison(family_results):
    """Plot comparison results across different matrix families."""
    # Create a figure for each family
    for family_name, family_data in family_results.items():
        test_names = list(family_data.keys())
        if not test_names:
            continue
            
        methods = list(family_data[test_names[0]].keys())
        
        # Prepare data for plotting
        error_data = []
        for test_name in test_names:
            for method in methods:
                if method in family_data[test_name]:
                    error_data.append({
                        'Matrix': test_name,
                        'Method': method,
                        'Error': family_data[test_name][method]['error']
                    })
        
        # Create DataFrame
        error_df = pd.DataFrame(error_data)
        
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Matrix', y='Error', hue='Method', data=error_df)
        plt.title(f'Error Comparison - {family_name}')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(f'results/family_{family_name.replace(" ", "_")}_error.png', dpi=300)
        plt.close()

# Run the main experiment
if __name__ == "__main__":
    print("Running Florida matrix experiments...")
    florida_matrix_experiments()
    
    # Uncomment to run the family comparison experiment
    # print("\nRunning family comparison experiments...")
    # compare_florida_matrix_families()
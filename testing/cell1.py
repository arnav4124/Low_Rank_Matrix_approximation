import numpy as np
import scipy.linalg as la
import time
from numpy.linalg import norm
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

###############################
# Rank Approximation Functions
###############################

def deterministic_rank_approx(A, rank):
    U, s, VT = la.svd(A, full_matrices=False)
    Ar = (U[:, :rank] * s[:rank]) @ VT[:rank, :]
    return Ar

def randomized_rank_approx(A, rank, oversample=5):
    random_matrix = np.random.randn(A.shape[1], rank + oversample)
    Y = A @ random_matrix
    Q, _ = la.qr(Y, mode='economic')
    B = Q.T @ A
    U_hat, s, VT = la.svd(B, full_matrices=False)
    Ar = Q @ (U_hat[:, :rank] * s[:rank]) @ VT[:rank, :]
    return Ar

#########################################
# RL Environment for Column Selection
#########################################

class ColumnSelectionEnv:
    def __init__(self, A, target_rank):
        self.A = A
        self.target_rank = target_rank
        self.m, self.n = A.shape
        self.reset()

    def reset(self):
        self.selected = []
        self.mask = np.zeros(self.n, dtype=np.float32)
        self.prev_error = norm(self.A, 'fro')
        return self._get_state()

    def _get_state(self):
        return self.mask.copy()

    def step(self, action):
        if self.mask[action] == 1:  ## already selected
            reward = -10.0          ## penalize for selecting the same column
            done = True             ## end the episode
            return self._get_state(), reward, done  

        self.selected.append(action)    ## add selected column
        self.mask[action] = 1.0         ## mark it as selected

        Ac = self.A[:, self.selected]                   ## selected columns
        A_approx = Ac @ la.pinv(Ac) @ self.A            ## approximate matrix
        current_error = norm(self.A - A_approx, 'fro')  ## Frobenius norm error

        reward = self.prev_error - current_error        ## reward is the difference in error
        self.prev_error = current_error                 ## update previous error

        done = len(self.selected) == self.target_rank   ## check if target rank is reached
        return self._get_state(), reward, done          ## return state, reward, done

#########################################
# Deep Q-Network and Agent Definition
#########################################

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        '''
        input dim - dimension of input (state representation)
        output dim - the dimension in which we want output
        hidden dim - This argument determines the number of neurons in the hidden layers of the network (default is 128).
        '''
        super(QNetwork, self).__init__()
        '''
        fc1: A fully connected layer that transforms the input to hidden_dim dimensions
        relu: A ReLU activation function that introduces non-linearity used after the first layer
        fc2: A second fully connected layer from hidden_dim to hidden_dim 
        fc3: A third fully connected layer that outputs the final values (typically Q-values for each action)
        '''
        self.fc1 = nn.Linear(input_dim, hidden_dim)     ## Input layer
        self.relu = nn.ReLU()                           ## ReLU activation function
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)    ## Hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)    ## Output layer

    def forward(self, x):
        x = self.relu(self.fc1(x))          ## Apply ReLU activation after the first layer
        x = self.relu(self.fc2(x))          ## Apply ReLU activation after the second layer
        x = self.fc3(x)                     ## Output layer
        return x

### This is the normal neural network code we write this
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3,
                 gamma=0.99, buffer_size=10000, batch_size=64):
        '''
        What it is: Batch size refers to the number of experiences
        (state, action, reward, next_state, done tuples) that are sampled
        from the replay buffer to update the Q-network's parameters in each training iteration'''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        '''
        What it is: load_state_dict is a PyTorch function used to load a model's
        parameters (weights and biases) from a state dictionary.
Why it's used:

    Initializing Models: You can use it to initialize a new model with the parameters
    of a pre-trained model.
    Copying Parameters: In your code, it's used to copy the parameters
    of the Q-network to the target network:
    self.target_network.load_state_dict(self.q_network.state_dict()).


        '''
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)    ## Q-network
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)   ## Target network
        self.target_network.load_state_dict(self.q_network.state_dict())    ## Copy parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)   ## Optimizer for Q-network

    def select_action(self, state, available_actions, epsilon):
        if random.random() < epsilon:   ## Epsilon-greedy action selection
            return random.choice(available_actions)     
        else:
            self.q_network.eval()   ## Set the Q-network to evaluation mode
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)    ## Convert state to tensor
            with torch.no_grad():   ## Disable gradient calculation
                q_values = self.q_network(state_tensor).cpu().numpy().flatten() ## Get Q-values
            q_values_masked = np.full_like(q_values, -np.inf)   ## Initialize masked Q-values
            q_values_masked[available_actions] = q_values[available_actions]    ## Mask Q-values for available actions
            action = int(np.argmax(q_values_masked))    ## Select action with max Q-value
            return action

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

#########################################
# Training Loop for the DQN Agent
#########################################

def train_dqn(agent, env, num_episodes=500, epsilon_start=1.0, epsilon_end=0.1,
              epsilon_decay=0.995, target_update_freq=10):
    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            available_actions = [i for i, v in enumerate(state) if v == 0]  ## get available actions
            action = agent.select_action(state, available_actions, epsilon) ## select action this is done using the epsilon greedy method
            next_state, reward, done = env.step(action)                     ## take a step in the enivironment
            # so this will go till we keep on not selecting any wrong col
            agent.store(state, action, reward, next_state, done)    ## store the experience in the replay buffer
            agent.update()                                          ## update the Q-network
            state = next_state                                      ## update the state
            total_reward += reward                                  ## accumulate the reward

        episode_rewards.append(total_reward)    ## store the total reward for this episode
        if episode % target_update_freq == 0:   ## update the target network 
            agent.update_target()

        epsilon = max(epsilon * epsilon_decay, epsilon_end) ## decay epsilon
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.4f}, Epsilon: {epsilon:.4f}")
    return episode_rewards

#########################################
# RL-based Column Selection using DQN
#########################################
def rl_column_selection_with_dqn(A, target_rank, agent):
    env = ColumnSelectionEnv(A, target_rank) ## create the environment
    state = env.reset() 
    selected_columns = []
    done = False

    while not done:
        available_actions = [i for i, v in enumerate(state) if v == 0]  ## get available actions
        action = agent.select_action(state, available_actions, epsilon=0.0) ## select action based on the policy
        selected_columns.append(action)
        state, reward, done = env.step(action)

    Ac = A[:, selected_columns]
    Ar = Ac @ la.pinv(Ac) @ A
    return Ar, selected_columns

#########################################
# Testing All Methods and Printing Table
#########################################

def test_methods():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    m, n, true_rank = 100, 80, 10

    A = np.dot(np.random.randn(m, true_rank), np.random.randn(true_rank, n))
    A += 0.01 * np.random.randn(m, n)

    approx_rank = 10

    start = time.time()
    Ar_det = deterministic_rank_approx(A, approx_rank)
    time_det = time.time() - start
    err_det = norm(A - Ar_det, 'fro')
    print(f'Deterministic SVD Error: {err_det:.4f}, Time: {time_det:.4f} sec')

    start = time.time()
    Ar_rand = randomized_rank_approx(A, approx_rank)
    time_rand = time.time() - start
    err_rand = norm(A - Ar_rand, 'fro')
    print(f'Randomized SVD Error: {err_rand:.4f}, Time: {time_rand:.4f} sec')

    env = ColumnSelectionEnv(A, approx_rank)
    state_dim = env.n
    action_dim = env.n
    agent = DQNAgent(state_dim, action_dim, hidden_dim=128, lr=1e-3,
                     gamma=0.99, buffer_size=5000, batch_size=64)
    print("Training DQN agent for RL-based column selection...")
    train_dqn(agent, env, num_episodes=500, epsilon_start=1.0, epsilon_end=0.1)

    start = time.time()
    Ar_rl, selected_columns = rl_column_selection_with_dqn(A, approx_rank, agent)
    time_rl = time.time() - start
    err_rl = norm(A - Ar_rl, 'fro')
    print(f'RL-based Column Selection (DQN) Error: {err_rl:.4f}, Time: {time_rl:.4f} sec')
    print("Selected columns:", selected_columns)

    # Print a properly aligned GitHub-style markdown table.
    header = "| {:<35} | {:>10} | {:>12} |".format("Method", "Error", "Time (sec)")
    divider = "|{:-<37}|{:-<12}|{:-<14}|".format("", "", "")
    row1 = "| {:<35} | {:>10.4f} | {:>12.4f} |".format("Deterministic SVD", err_det, time_det)
    row2 = "| {:<35} | {:>10.4f} | {:>12.4f} |".format("Randomized SVD", err_rand, time_rand)
    row3 = "| {:<35} | {:>10.4f} | {:>12.4f} |".format("RL-based Column Selection (DQN)", err_rl, time_rl)

    print("\n" + header)
    print(divider)
    print(row1)
    print(row2)
    print(row3)

if __name__ == '__main__':
    test_methods()

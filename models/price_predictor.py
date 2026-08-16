import numpy as np

class LSTMLikePredictor:
    """
    A lightweight, robust predictive model designed to simulate
    LSTM temporal dependencies and non-linear patterns using an optimized 
    Recursive Feature Ridge Regression structure.
    
    This avoids heavy framework overhead (PyTorch/TensorFlow) while providing
    highly stable, online-trainable time-series predictions.
    """
    def __init__(self, input_dim=5, hidden_dim=8, l2_reg=1.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l2_reg = l2_reg
        
        # Initialize weight matrices (recurrent simulation)
        # W_h: Hidden state transitions, W_x: Input mappings
        np.random.seed(42)
        self.W_x = np.random.normal(0, 1.0 / np.sqrt(input_dim), (hidden_dim, input_dim))
        self.W_h = np.random.normal(0, 1.0 / np.sqrt(hidden_dim), (hidden_dim, hidden_dim))
        self.b_h = np.zeros((hidden_dim, 1))
        
        # Output weights (Ridge regression on hidden state)
        self.W_out = np.zeros((1, hidden_dim))
        self.b_out = 0.0

    def _tanh(self, x):
        return np.tanh(x)

    def _extract_hidden_states(self, X_seq):
        """
        Processes a temporal sequence of inputs and extracts the final hidden state.
        X_seq: shape (seq_len, input_dim)
        """
        h = np.zeros((self.hidden_dim, 1))
        for t in range(X_seq.shape[0]):
            x_t = X_seq[t].reshape(-1, 1)
            h = self._tanh(np.dot(self.W_x, x_t) + np.dot(self.W_h, h) + self.b_h)
        return h.flatten()

    def fit(self, X_sequences, y_targets):
        """
        Trains the output layer via Ridge Regression using extracted recurrent features.
        X_sequences: list or array of sequences, each shape (seq_len, input_dim)
        y_targets: array of shape (N_samples,) containing future returns
        """
        N = len(X_sequences)
        if N < 5:
            return self
            
        H = np.zeros((N, self.hidden_dim))
        for i in range(N):
            H[i] = self._extract_hidden_states(X_sequences[i])
            
        # Ridge regression closed-form solver: W = (H^T * H + lambda * I)^(-1) * H^T * y
        I = np.eye(self.hidden_dim)
        HT_H = np.dot(H.T, H) + self.l2_reg * I
        HT_y = np.dot(H.T, y_targets)
        
        try:
            self.W_out = np.linalg.solve(HT_H, HT_y).reshape(1, -1)
            self.b_out = np.mean(y_targets - np.dot(H, self.W_out.T).flatten())
        except np.linalg.LinAlgError:
            # Fallback in case of singular matrix
            self.W_out = np.zeros((1, self.hidden_dim))
            self.b_out = float(np.mean(y_targets))
            
        return self

    def predict(self, X_seq):
        """
        Predicts future price return based on a single temporal sequence of features.
        X_seq: shape (seq_len, input_dim)
        """
        h = self._extract_hidden_states(X_seq)
        prediction = np.dot(self.W_out, h.reshape(-1, 1)) + self.b_out
        return float(prediction[0, 0])


class PPOTRAgent:
    """
    A complete, pure-numpy Actor-Critic Reinforcement Learning Agent
    modeled after the PPO (Proximal Policy Optimization) framework.
    
    Optimizes trading exposure (Target Position) based on state vectors.
    """
    def __init__(self, state_dim=4, action_dim=1, lr=0.01, clip_epsilon=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        
        # Policy Network (Actor): outputs mean of action distribution
        # Assuming action space is single-dimensional: target portfolio exposure in [-1.0, 1.0]
        np.random.seed(88)
        self.actor_w = np.random.normal(0, 0.1, (state_dim, action_dim))
        self.actor_b = np.zeros((action_dim, 1))
        self.action_std = 0.2  # Exploration variance
        
        # Value Network (Critic): outputs state value V(s)
        self.critic_w = np.random.normal(0, 0.1, (state_dim, 1))
        self.critic_b = 0.0

    def get_action(self, state):
        """
        Given a state vector, sample an action from the policy distribution (Gaussian)
        and return the chosen action along with its log-probability.
        state: shape (state_dim,)
        """
        state_col = state.reshape(-1, 1)
        mean = np.dot(self.actor_w.T, state_col) + self.actor_b
        mean = float(mean[0, 0])
        mean_clipped = np.clip(mean, -1.0, 1.0)
        
        # Sample action using Gaussian exploration
        action = np.random.normal(mean_clipped, self.action_std)
        action_clipped = np.clip(action, -1.0, 1.0)
        
        # Calculate log-probability under the policy
        variance = self.action_std ** 2
        log_prob = -0.5 * np.log(2 * np.pi * variance) - ((action_clipped - mean_clipped) ** 2) / (2 * variance)
        
        return float(action_clipped), float(log_prob)

    def get_value(self, state):
        """
        Estimates the state value V(s).
        """
        state_col = state.reshape(-1, 1)
        val = np.dot(self.critic_w.T, state_col) + self.critic_b
        return float(val[0, 0])

    def train_step(self, states, actions, log_probs_old, rewards, next_states, terminals):
        """
        Executes a localized PPO Actor-Critic update step on collected trajectories.
        """
        states = np.array(states) # (Batch_Size, state_dim)
        actions = np.array(actions) # (Batch_Size,)
        log_probs_old = np.array(log_probs_old) # (Batch_Size,)
        rewards = np.array(rewards) # (Batch_Size,)
        next_states = np.array(next_states) # (Batch_Size, state_dim)
        terminals = np.array(terminals) # (Batch_Size,)
        
        # 1. Compute target values and advantages (TD residual)
        gamma = 0.99
        values = np.array([self.get_value(s) for s in states])
        next_values = np.array([self.get_value(ns) for ns in next_states])
        
        targets = rewards + gamma * next_values * (1.0 - terminals)
        advantages = targets - values
        
        # Normalize advantages for training stability
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # 2. Update Critic Weights (Minimize MSE Loss)
        # Gradient of Critic Loss = -2 * (Target - V(s)) * State
        critic_gradients_w = np.zeros_like(self.critic_w)
        critic_gradients_b = 0.0
        
        for idx in range(len(states)):
            diff = targets[idx] - values[idx]
            critic_gradients_w += -2.0 * diff * states[idx].reshape(-1, 1)
            critic_gradients_b += -2.0 * diff
            
        # Gradient descent step
        self.critic_w -= self.lr * (critic_gradients_w / len(states))
        self.critic_b -= self.lr * (critic_gradients_b / len(states))
        
        # 3. Update Actor Weights (PPO Clipped Objective)
        actor_gradients_w = np.zeros_like(self.actor_w)
        actor_gradients_b = np.zeros_like(self.actor_b)
        
        for idx in range(len(states)):
            state_col = states[idx].reshape(-1, 1)
            mean = np.dot(self.actor_w.T, state_col) + self.actor_b
            mean = float(mean[0, 0])
            mean_clipped = np.clip(mean, -1.0, 1.0)
            
            variance = self.action_std ** 2
            log_prob_new = -0.5 * np.log(2 * np.pi * variance) - ((actions[idx] - mean_clipped) ** 2) / (2 * variance)
            
            # Probability ratio r_t(theta)
            ratio = np.exp(log_prob_new - log_probs_old[idx])
            
            # Clipped objective
            surr1 = ratio * advantages[idx]
            surr2 = np.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages[idx]
            
            # If surr1 < surr2, optimize policy based on surr1 gradient
            if surr1 < surr2 or (ratio < 1.0 - self.clip_epsilon and advantages[idx] > 0) or (ratio > 1.0 + self.clip_epsilon and advantages[idx] < 0):
                # Gradient of log_prob_new with respect to actor weights:
                # d_log_prob / d_mean = (action - mean) / variance
                # d_mean / d_w = state
                grad_mean = (actions[idx] - mean_clipped) / variance
                actor_gradients_w += -ratio * advantages[idx] * grad_mean * state_col
                actor_gradients_b += -ratio * advantages[idx] * grad_mean
                
        # Gradient descent step
        self.actor_w -= self.lr * (actor_gradients_w / len(states))
        self.actor_b -= self.lr * (actor_gradients_b / len(states))
        
        return self

import numpy as np

class LSTMLikePredictor:
    """
    A genuine, mathematically rigorous Long Short-Term Memory (LSTM) Neural Network 
    implemented entirely in pure NumPy.
    
    Contains an authentic LSTM Cell with Forget, Input, Output, and Candidate Cell Gates.
    Includes forward propagation through time (BPTT) and online training gradient descents.
    Requires no external heavy frameworks (PyTorch/TensorFlow).
    """
    # P0-5 (audit §4.9) : hidden_dim=24 par défaut = MÊME archi que le live
    # (main.py). Tout backtest/script qui instancie LSTMLikePredictor() sans
    # préciser hidden_dim hérite de l'archi déployée — plus de dérive 8 vs 24.
    def __init__(self, input_dim=5, hidden_dim=24, output_dim=1, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        
        # Combined dimension: [h_prev, x_t]
        concat_dim = hidden_dim + input_dim
        
        # Initialize weight matrices with Xavier/Glorot Normal initialization
        np.random.seed(42)
        
        # Forget Gate
        self.W_f = np.random.normal(0, np.sqrt(2.0 / concat_dim), (hidden_dim, concat_dim))
        self.b_f = np.zeros((hidden_dim, 1))
        
        # Input Gate
        self.W_i = np.random.normal(0, np.sqrt(2.0 / concat_dim), (hidden_dim, concat_dim))
        self.b_i = np.zeros((hidden_dim, 1))
        
        # Candidate Cell State
        self.W_c = np.random.normal(0, np.sqrt(2.0 / concat_dim), (hidden_dim, concat_dim))
        self.b_c = np.zeros((hidden_dim, 1))
        
        # Output Gate
        self.W_o = np.random.normal(0, np.sqrt(2.0 / concat_dim), (hidden_dim, concat_dim))
        self.b_o = np.zeros((hidden_dim, 1))
        
        # Output Projection Layer (from hidden state to price prediction)
        self.W_out = np.random.normal(0, np.sqrt(2.0 / hidden_dim), (output_dim, hidden_dim))
        self.b_out = np.zeros((output_dim, 1))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _tanh(self, x):
        return np.tanh(x)

    def forward(self, X_seq):
        """
        Executes a complete forward pass through time over an input sequence.
        X_seq: shape (seq_len, input_dim)
        
        Returns:
          - h_states: hidden states for all steps
          - c_states: cell states for all steps
          - f_gates, i_gates, c_bar_gates, o_gates: gate outputs for backprop
          - final_prediction: float value
        """
        seq_len = X_seq.shape[0]
        
        h_states = {}
        c_states = {}
        f_gates = {}
        i_gates = {}
        c_bar_gates = {}
        o_gates = {}
        
        # Initialize hidden and cell states at t = -1 to 0
        h_states[-1] = np.zeros((self.hidden_dim, 1))
        c_states[-1] = np.zeros((self.hidden_dim, 1))
        
        for t in range(seq_len):
            x_t = X_seq[t].reshape(-1, 1)
            
            # Concatenate h_{t-1} and x_t
            concat = np.vstack((h_states[t-1], x_t))
            
            # Forget Gate
            f_gates[t] = self._sigmoid(np.dot(self.W_f, concat) + self.b_f)
            
            # Input Gate
            i_gates[t] = self._sigmoid(np.dot(self.W_i, concat) + self.b_i)
            
            # Candidate Cell State
            c_bar_gates[t] = self._tanh(np.dot(self.W_c, concat) + self.b_c)
            
            # Update Cell State: C_t = f_t * C_{t-1} + i_t * C_tilde_t
            c_states[t] = f_gates[t] * c_states[t-1] + i_gates[t] * c_bar_gates[t]
            
            # Output Gate
            o_gates[t] = self._sigmoid(np.dot(self.W_o, concat) + self.b_o)
            
            # Update Hidden State: h_t = o_t * tanh(C_t)
            h_states[t] = o_gates[t] * self._tanh(c_states[t])
            
        # Compute final output projection: y = W_out * h_last + b_out
        final_prediction = np.dot(self.W_out, h_states[seq_len - 1]) + self.b_out
        
        return h_states, c_states, f_gates, i_gates, c_bar_gates, o_gates, float(final_prediction[0, 0])

    def fit(self, X_sequences, y_targets, epochs=5):
        """
        Trains the entire LSTM network (including all gates) over multiple epochs
        using backpropagation through time (BPTT).
        """
        N = len(X_sequences)
        if N < 5:
            return self
            
        for epoch in range(epochs):
            for idx in range(N):
                X_seq = np.array(X_sequences[idx])
                y_target = y_targets[idx]
                
                # 1. Forward Pass
                h_states, c_states, f_gates, i_gates, c_bar_gates, o_gates, pred = self.forward(X_seq)
                
                # 2. Backpropagation through time (BPTT)
                seq_len = X_seq.shape[0]
                dy = pred - y_target
                
                # Gradients for Output Projection Layer
                dW_out = dy * h_states[seq_len - 1].T
                db_out = dy
                
                # Initialize gradients for gates with zero
                dW_f, dW_i, dW_c, dW_o = np.zeros_like(self.W_f), np.zeros_like(self.W_i), np.zeros_like(self.W_c), np.zeros_like(self.W_o)
                db_f, db_i, db_c, db_o = np.zeros_like(self.b_f), np.zeros_like(self.b_i), np.zeros_like(self.b_c), np.zeros_like(self.b_o)
                
                # Backpropagate through hidden states
                dh_next = np.dot(self.W_out.T, dy)
                dc_next = np.zeros_like(c_states[-1])
                
                for t in reversed(range(seq_len)):
                    x_t = X_seq[t].reshape(-1, 1)
                    concat = np.vstack((h_states[t-1], x_t))
                    
                    # Gradient of loss with respect to h_t
                    dh = dh_next
                    
                    # Gradient with respect to Output Gate
                    do = dh * self._tanh(c_states[t])
                    do_net = do * o_gates[t] * (1.0 - o_gates[t]) # derivative of sigmoid
                    
                    dW_o += np.dot(do_net, concat.T)
                    db_o += do_net
                    
                    # Gradient with respect to Cell State
                    dc = dh * o_gates[t] * (1.0 - self._tanh(c_states[t])**2) + dc_next
                    
                    # Gradient with respect to Candidate Cell State
                    dc_bar = dc * i_gates[t]
                    dc_bar_net = dc_bar * (1.0 - c_bar_gates[t]**2) # derivative of tanh
                    
                    dW_c += np.dot(dc_bar_net, concat.T)
                    db_c += dc_bar_net
                    
                    # Gradient with respect to Input Gate
                    di = dc * c_bar_gates[t]
                    di_net = di * i_gates[t] * (1.0 - i_gates[t])
                    
                    dW_i += np.dot(di_net, concat.T)
                    db_i += di_net
                    
                    # Gradient with respect to Forget Gate
                    df = dc * c_states[t-1]
                    df_net = df * f_gates[t] * (1.0 - f_gates[t])
                    
                    dW_f += np.dot(df_net, concat.T)
                    db_f += df_net
                    
                    # Update dh_next and dc_next for previous time step t-1
                    dconcat = (
                        np.dot(self.W_f.T, df_net) +
                        np.dot(self.W_i.T, di_net) +
                        np.dot(self.W_c.T, dc_bar_net) +
                        np.dot(self.W_o.T, do_net)
                    )
                    dh_next = dconcat[:self.hidden_dim, :]
                    dc_next = dc * f_gates[t]
                    
                # 3. Apply Gradient Descent Weight Updates (with gradient clipping to avoid explosions)
                clip_val = 1.0
                for grad_arr in [dW_f, dW_i, dW_c, dW_o, dW_out, db_f, db_i, db_c, db_o]:
                    np.clip(grad_arr, -clip_val, clip_val, out=grad_arr)
                db_out = max(-clip_val, min(clip_val, db_out))
                    
                self.W_f -= self.lr * dW_f
                self.b_f -= self.lr * db_f
                self.W_i -= self.lr * dW_i
                self.b_i -= self.lr * db_i
                self.W_c -= self.lr * dW_c
                self.b_c -= self.lr * db_c
                self.W_o -= self.lr * dW_o
                self.b_o -= self.lr * db_o
                self.W_out -= self.lr * dW_out
                self.b_out -= self.lr * db_out
                
        return self

    def predict(self, X_seq):
        """
        Predicts future price return using a single sequence of technical features.
        X_seq: shape (seq_len, input_dim)
        """
        _, _, _, _, _, _, pred = self.forward(X_seq)
        return pred


class PPOTRAgent:
    """
    A complete Actor-Critic Reinforcement Learning Agent (PPO) with a hidden
    layer (audit B9-1) - a genuinely non-linear policy, not a linear one.
    Pure numpy, deterministic seeds, same API as before (get_action / get_value /
    train_step) so the autonomous loop keeps working.
    """

    def __init__(self, state_dim=4, action_dim=1, hidden_dim=16, lr=0.01, clip_epsilon=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.clip_epsilon = clip_epsilon
        self.lr = lr

        np.random.seed(88)
        # Actor: input -> hidden -> action
        self.actor_w1 = np.random.normal(0, 0.1, (state_dim, hidden_dim))
        self.actor_b1 = np.zeros((1, hidden_dim))
        self.actor_w2 = np.random.normal(0, 0.1, (hidden_dim, action_dim))
        self.actor_b2 = np.zeros((1, action_dim))
        self.action_std = 0.2

        # Critic: input -> hidden -> value
        self.critic_w1 = np.random.normal(0, 0.1, (state_dim, hidden_dim))
        self.critic_b1 = np.zeros((1, hidden_dim))
        self.critic_w2 = np.random.normal(0, 0.1, (hidden_dim, 1))
        self.critic_b2 = 0.0

    def _actor_mean(self, state_col):
        h = np.tanh(np.dot(state_col.T, self.actor_w1) + self.actor_b1)
        mean = float((np.dot(h, self.actor_w2) + self.actor_b2)[0, 0])
        return np.clip(mean, -1.0, 1.0)

    def _critic_value(self, state_col):
        h = np.tanh(np.dot(state_col.T, self.critic_w1) + self.critic_b1)
        return float((np.dot(h, self.critic_w2) + self.critic_b2)[0, 0])

    def get_action(self, state):
        state_col = state.reshape(-1, 1)
        mean_clipped = self._actor_mean(state_col)
        action = np.random.normal(mean_clipped, self.action_std)
        action_clipped = np.clip(action, -1.0, 1.0)
        variance = self.action_std ** 2
        log_prob = -0.5 * np.log(2 * np.pi * variance) - ((action_clipped - mean_clipped) ** 2) / (2 * variance)
        return float(action_clipped), float(log_prob)

    def get_value(self, state):
        return self._critic_value(state.reshape(-1, 1))

    def train_step(self, states, actions, log_probs_old, rewards, next_states, terminals):
        states = np.array(states)
        actions = np.array(actions)
        log_probs_old = np.array(log_probs_old)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        terminals = np.array(terminals)

        gamma = 0.99
        values = np.array([self.get_value(s) for s in states])
        next_values = np.array([self.get_value(ns) for ns in next_states])

        targets = rewards + gamma * next_values * (1.0 - terminals)
        advantages = targets - values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        n = len(states)
        # Critic update (MSE on targets)
        g_w1 = np.zeros_like(self.critic_w1)
        g_b1 = np.zeros_like(self.critic_b1)
        g_w2 = np.zeros_like(self.critic_w2)
        g_b2 = 0.0
        for idx in range(n):
            state_col = states[idx].reshape(-1, 1)
            h = np.tanh(np.dot(state_col.T, self.critic_w1) + self.critic_b1)
            val = float((np.dot(h, self.critic_w2) + self.critic_b2)[0, 0])
            diff = targets[idx] - val
            dh = 1.0 - h ** 2
            g_w2 += -2.0 * diff * h.T
            g_b2 += -2.0 * diff
            g_w1 += -2.0 * diff * np.outer(state_col.ravel(), (dh * self.critic_w2.T).ravel())
            g_b1 += -2.0 * diff * (self.critic_w2.T * dh)
        self.critic_w1 -= self.lr * (g_w1 / n)
        self.critic_b1 -= self.lr * (g_b1 / n)
        self.critic_w2 -= self.lr * (g_w2 / n)
        self.critic_b2 -= self.lr * (g_b2 / n)

        # Actor update (clipped PPO objective)
        g_w1 = np.zeros_like(self.actor_w1)
        g_b1 = np.zeros_like(self.actor_b1)
        g_w2 = np.zeros_like(self.actor_w2)
        g_b2 = np.zeros_like(self.actor_b2)
        for idx in range(n):
            state_col = states[idx].reshape(-1, 1)
            h = np.tanh(np.dot(state_col.T, self.actor_w1) + self.actor_b1)
            mean_clipped = float((np.dot(h, self.actor_w2) + self.actor_b2)[0, 0])
            mean_clipped = np.clip(mean_clipped, -1.0, 1.0)
            variance = self.action_std ** 2
            log_prob_new = -0.5 * np.log(2 * np.pi * variance) - ((actions[idx] - mean_clipped) ** 2) / (2 * variance)

            ratio = np.exp(log_prob_new - log_probs_old[idx])
            surr1 = ratio * advantages[idx]
            surr2 = np.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages[idx]
            if surr1 < surr2 or (ratio < 1.0 - self.clip_epsilon and advantages[idx] > 0) or (ratio > 1.0 + self.clip_epsilon and advantages[idx] < 0):
                grad_mean = (actions[idx] - mean_clipped) / variance
                dh = 1.0 - h ** 2
                g_w2 += -ratio * advantages[idx] * grad_mean * h.T
                g_b2 += -ratio * advantages[idx] * grad_mean
                g_w1 += -ratio * advantages[idx] * grad_mean * np.outer(state_col.ravel(), (dh * self.actor_w2.T).ravel())
                g_b1 += -ratio * advantages[idx] * grad_mean * (self.actor_w2.T * dh)
        self.actor_w1 -= self.lr * (g_w1 / n)
        self.actor_b1 -= self.lr * (g_b1 / n)
        self.actor_w2 -= self.lr * (g_w2 / n)
        self.actor_b2 -= self.lr * (g_b2 / n)
        return self

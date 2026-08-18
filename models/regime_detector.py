import numpy as np
import pandas as pd

class MarketRegimeDetector:
    """
    Hidden Markov Model (HMM) designed for financial market regime detection.
    Identifies 4 states:
      0: Bull (low vol, positive returns)
      1: Bear (high vol, negative returns)
      2: Range (very low vol, returns close to 0)
      3: High Volatility (erratic returns, spikes in spread)
    
    Uses an Expectation-Maximization (Baum-Welch) algorithm for training
    and the Viterbi algorithm for decoding states.
    """
    def __init__(self, n_states=4, max_iter=100, tol=1e-4):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        
        # Initialize probabilities transition matrix, emission parameters
        # State transitions: higher probability to stay in the same state
        self.transition_matrix = np.array([
            [0.85, 0.05, 0.07, 0.03], # From Bull
            [0.08, 0.80, 0.04, 0.08], # From Bear
            [0.10, 0.05, 0.80, 0.05], # From Range
            [0.05, 0.15, 0.05, 0.75]  # From High Vol
        ])
        
        # Prior distribution of states
        self.prior = np.array([0.3, 0.2, 0.4, 0.1])
        
        # Feature emission means (feature 1: return, feature 2: volatility)
        self.means = np.array([
            [0.0015, 0.008],  # State 0: Bull (positive return, low-med vol)
            [-0.0025, 0.025], # State 1: Bear (negative return, high vol)
            [0.0001, 0.004],  # State 2: Range (neutral return, very low vol)
            [0.0005, 0.035]   # State 3: High Vol (erratic return, massive vol)
        ])
        
        # Feature emission covariances (diagonal approximations for simplicity and stability)
        self.covariances = np.array([
            [1e-6, 1e-5],
            [5e-6, 5e-5],
            [5e-7, 5e-6],
            [1e-5, 1e-4]
        ])

    def _gaussian_probability(self, x, mean, cov):
        """
        Multivariate Gaussian density evaluation with diagonal covariance.
        """
        ndim = len(x)
        # Prevent division by zero
        cov = np.clip(cov, 1e-9, None)
        det = np.prod(cov)
        inv = 1.0 / cov
        diff = x - mean
        exponent = -0.5 * np.sum(diff * diff * inv)
        prob = (1.0 / np.sqrt(((2 * np.pi) ** ndim) * det)) * np.exp(exponent)
        return max(prob, 1e-15)

    def _compute_emissions(self, X):
        """
        Compute emission probability matrix for all observations X.
        X shape: (N_samples, N_features)
        Returns: (N_samples, n_states)
        """
        N = X.shape[0]
        emissions = np.zeros((N, self.n_states))
        for i in range(N):
            for s in range(self.n_states):
                emissions[i, s] = self._gaussian_probability(X[i], self.means[s], self.covariances[s])
        return emissions

    def fit(self, X):
        """
        Train the HMM on historical features X using Baum-Welch (simplified EM).
        X: numpy array of shape (N_samples, N_features)
        """
        if len(X) < 10:
            return self # Not enough data to fit, use priors
        
        N = X.shape[0]
        
        for iteration in range(self.max_iter):
            # 1. Emission probabilities
            emissions = self._compute_emissions(X)
            
            # Forward-Backward variables
            forward = np.zeros((N, self.n_states))
            backward = np.zeros((N, self.n_states))
            
            # Forward pass
            forward[0] = self.prior * emissions[0]
            forward[0] /= np.sum(forward[0]) + 1e-15
            
            for t in range(1, N):
                for s in range(self.n_states):
                    forward[t, s] = np.sum(forward[t-1] * self.transition_matrix[:, s]) * emissions[t, s]
                forward[t] /= np.sum(forward[t]) + 1e-15
            
            # Backward pass
            backward[N-1] = np.ones(self.n_states)
            for t in range(N-2, -1, -1):
                for s in range(self.n_states):
                    backward[t, s] = np.sum(self.transition_matrix[s, :] * emissions[t+1] * backward[t+1])
                backward[t] /= np.sum(backward[t]) + 1e-15
                
            # Compute Gammas (state probabilities) and Xis (transition probabilities)
            gamma = forward * backward
            gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-15
            
            # Check convergence
            old_means = self.means.copy()
            
            # M-step: Update Priors, Transition matrix, Means, and Covariances
            self.prior = gamma[0] / (np.sum(gamma[0]) + 1e-15)
            
            # Update transitions
            xi = np.zeros((N-1, self.n_states, self.n_states))
            for t in range(N-1):
                denom = np.sum(forward[t][:, None] * self.transition_matrix * emissions[t+1] * backward[t+1]) + 1e-15
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (forward[t, i] * self.transition_matrix[i, j] * emissions[t+1, j] * backward[t+1, j]) / denom
            
            self.transition_matrix = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0)[:, None] + 1e-15)
            self.transition_matrix /= np.sum(self.transition_matrix, axis=1, keepdims=True) + 1e-15
            
            # Update Means and Covariances
            for s in range(self.n_states):
                gamma_sum = np.sum(gamma[:, s]) + 1e-15
                self.means[s] = np.sum(gamma[:, s][:, None] * X, axis=0) / gamma_sum
                
                # Diagonal variance update
                diff = X - self.means[s]
                self.covariances[s] = np.sum(gamma[:, s][:, None] * (diff ** 2), axis=0) / gamma_sum
                self.covariances[s] = np.clip(self.covariances[s], 1e-9, None)
                
            # Tolerance threshold for convergence
            if np.max(np.abs(self.means - old_means)) < self.tol:
                break
                
        return self

    def predict_proba(self, X):
        """
        Predict probability distribution over states for each observation.
        """
        emissions = self._compute_emissions(X)
        N = X.shape[0]
        forward = np.zeros((N, self.n_states))
        
        forward[0] = self.prior * emissions[0]
        forward[0] /= np.sum(forward[0]) + 1e-15
        
        for t in range(1, N):
            for s in range(self.n_states):
                forward[t, s] = np.sum(forward[t-1] * self.transition_matrix[:, s]) * emissions[t, s]
            forward[t] /= np.sum(forward[t]) + 1e-15
            
        return forward

    def predict(self, X):
        """
        Viterbi algorithm to decode the most likely sequence of hidden states.
        """
        N = X.shape[0]
        emissions = self._compute_emissions(X)
        
        # DP tables
        viterbi_table = np.zeros((N, self.n_states))
        backpointer = np.zeros((N, self.n_states), dtype=int)
        
        # Init
        viterbi_table[0] = np.log(np.clip(self.prior, 1e-15, None)) + np.log(np.clip(emissions[0], 1e-15, None))
        
        for t in range(1, N):
            for s in range(self.n_states):
                trans_prob = viterbi_table[t-1] + np.log(np.clip(self.transition_matrix[:, s], 1e-15, None))
                backpointer[t, s] = np.argmax(trans_prob)
                viterbi_table[t, s] = trans_prob[backpointer[t, s]] + np.log(np.clip(emissions[t, s], 1e-15, None))
                
        # Backtrack
        state_seq = np.zeros(N, dtype=int)
        state_seq[N-1] = np.argmax(viterbi_table[N-1])
        
        for t in range(N-2, -1, -1):
            state_seq[t] = backpointer[t+1, state_seq[t+1]]
            
        return state_seq

    
    def regime_confidence(self, X):
        """
        LOT 4 (PDF Pilier B) : qualité de l'inférence de régime.

        Confiance = probabilité soft du régime dominant sur la DERNIÈRE
        observation, pénalisée par l'instabilité (nombre de changements d'état
        sur la séquence). Retourne (confiance 0..1, regime_id, prob_dominante).

        Principe (mentalité n°5/n°20) : on ne trade un régime que s'il est
        suffisamment CERTAIN — sinon « je ne sais pas » -> réduire.
        """
        proba = self.predict_proba(X)
        probs_last = proba[-1]
        regime_id = int(np.argmax(probs_last))
        conf = float(probs_last[regime_id])
        # Stabilité : 1 - (changements d'état / N) sur la séquence décodée
        try:
            seq = self.predict(X)
            changes = int(np.sum(np.abs(np.diff(seq)) > 0))
            stability = max(0.0, 1.0 - changes / max(len(seq) - 1, 1))
        except Exception:
            stability = 0.5
        return {
            "confidence": round(conf * (0.5 + 0.5 * stability), 4),
            "regime_id": regime_id,
            "prob_dominant": round(conf, 4),
            "stability": round(stability, 4),
        }

    def validate_on_asset(self, df, symbol: str = "?"):
        """
        LOT 4 (PDF Pilier B) : validation du HMM sur UN actif (les 7, pas
        seulement BTC). Retourne vraisemblance moyenne et stabilité, ou None
        si l'historique est insuffisant (honnêteté : pas de validation sur du
        vide).
        """
        try:
            if df is None or len(df) < 30:
                return None
            rets = df["close"].pct_change().dropna().values[-100:]
            if len(rets) < 20:
                return None
            vols = np.abs(rets).copy()
            X = np.column_stack([rets, vols])
            proba = self.predict_proba(X)
            # Log-vraisemblance moyenne (évite les underflow via log)
            eps = 1e-15
            loglik = float(np.mean(np.log(np.clip(np.max(proba, axis=1), eps, None))))
            seq = self.predict(X)
            changes = int(np.sum(np.abs(np.diff(seq)) > 0))
            stability = max(0.0, 1.0 - changes / max(len(seq) - 1, 1))
            return {
                "symbol": symbol,
                "n_samples": len(rets),
                "loglik_mean": round(loglik, 4),
                "stability": round(stability, 4),
                "regime_mix": [int((seq == s).sum()) for s in range(self.n_states)],
            }
        except Exception as e:
            logger = __import__("logging").getLogger("RegimeDetector")
            logger.debug(f"validate_on_asset failed for {symbol}: {e}")
            return None

    def get_regime_name(self, state_id):
        mapping = {
            0: "Bull Trend (Low Vol)",
            1: "Bear Trend (High Vol)",
            2: "Mean-Reverting Range",
            3: "Erratic High Volatility"
        }
        return mapping.get(state_id, "Unknown")


def compute_order_book_imbalance(bids, asks, depth=5):
    """
    Computes Order Book Imbalance (OBI) weighted by distance to the mid-price.
    bids: list of [price, volume]
    asks: list of [price, volume]
    depth: number of levels to use
    
    Returns a score between -1.0 (pure supply/selling pressure) and +1.0 (pure demand/buying pressure).
    """
    if not bids or not asks:
        return 0.0
        
    obi_bids = 0.0
    obi_asks = 0.0
    
    # Sort order books
    sorted_bids = sorted(bids, key=lambda x: x[0], reverse=True)[:depth]
    sorted_asks = sorted(asks, key=lambda x: x[0], reverse=False)[:depth]
    
    # Calculate weighted volume: weight is inverse to index (1/1, 1/2, 1/3, etc.)
    for idx, bid in enumerate(sorted_bids):
        weight = 1.0 / (idx + 1)
        obi_bids += bid[1] * weight
        
    for idx, ask in enumerate(sorted_asks):
        weight = 1.0 / (idx + 1)
        obi_asks += ask[1] * weight
        
    denominator = obi_bids + obi_asks
    if denominator == 0:
        return 0.0
        
    return (obi_bids - obi_asks) / denominator

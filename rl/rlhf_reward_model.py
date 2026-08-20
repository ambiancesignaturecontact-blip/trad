"""
LOT 55: RLHF (Reinforcement Learning from Human Feedback) for Trading
"""

import logging

import numpy as np

# PyTorch is a heavy OPTIONAL dependency. Without it, the RLHF reward model
# degrades gracefully (returns neutral scores) so the platform still boots.
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

logger = logging.getLogger("RLHFRewardModel")

if TORCH_AVAILABLE:
    class RewardModel(nn.Module):
        """Neural network that predicts human preference score for a trade"""
        def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Tanh()  # Output between -1 and 1
            )

        def forward(self, x):
            return self.net(x)


class RLHFRewardModel:
    """
    LOT 55: RLHF system for trading decisions.
    - Collects human (or simulated) feedback on trades
    - Trains a reward model
    - Can be used to score and adjust trading signals
    """

    def __init__(self, input_dim: int = 10):
        self.feedback_buffer: list[tuple[np.ndarray, float]] = []
        self.max_buffer_size = 1000
        self.is_trained = False
        self.input_dim = input_dim

        if not TORCH_AVAILABLE:
            logger.warning(
                "LOT 55: PyTorch not installed - RLHF reward model running in "
                "fallback mode (neutral scores). Install torch to enable it."
            )
            self.model = None
            self.optimizer = None
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RewardModel(input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def add_feedback(self, features: np.ndarray, human_preference_score: float):
        """
        Add human feedback.
        human_preference_score: float between -1 (bad) and +1 (good)
        """
        if len(features) != self.input_dim:
            logger.warning(f"Feature dimension mismatch. Expected {self.input_dim}, got {len(features)}")
            return

        self.feedback_buffer.append((features, human_preference_score))

        if len(self.feedback_buffer) > self.max_buffer_size:
            self.feedback_buffer.pop(0)

    def train_reward_model(self, epochs: int = 400):
        """Train the reward model on collected feedback"""
        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("LOT 55: Cannot train - PyTorch not available.")
            return False
        if len(self.feedback_buffer) < 30:
            logger.warning("Not enough feedback to train RLHF model")
            return False

        X = torch.tensor(np.array([f[0] for f in self.feedback_buffer]), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array([f[1] for f in self.feedback_buffer]), dtype=torch.float32).unsqueeze(1).to(self.device)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = nn.MSELoss()(pred, y)
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                logger.info(f"LOT 55: RLHF Reward Model | Epoch {epoch} | Loss: {loss.item():.4f}")

        self.is_trained = True
        logger.info("LOT 55: RLHF Reward Model training completed")
        return True

    def predict_reward(self, features: np.ndarray):
        """
        Predict how much a human would like this trade situation.

        LIMITES (LOT 4, PDF Pilier C) : RLHF est EXPÉRIMENTAL. Sans PyTorch
        ou sans entraînement, renvoie None = « je ne sais pas » — le main loop
        n'applique alors AUCUN modulateur (facteur 1.0). Correction d'un bug
        où le fallback 0.0 réduisait la taille de moitié sans raison
        (mentalité n°20 : savoir dire je ne sais pas).
        """
        if not TORCH_AVAILABLE or not self.is_trained:
            return None

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            reward = self.model(x).item()
        return float(np.clip(reward, -1.0, 1.0))

    def get_status(self) -> dict:
        return {
            "feedback_count": len(self.feedback_buffer),
            "is_trained": self.is_trained,
            "model_version": "v1.0"
        }

    def simulate_human_feedback(self, pnl: float, drawdown: float, regime_id: int) -> float:
        """
        Simulated human preference for testing.
        In real use, this would come from actual trader feedback.
        """
        score = 0.0
        score += np.tanh(pnl * 8) * 0.6
        score -= np.tanh(drawdown * 5) * 0.3
        if regime_id in [0, 2]:  # Bull or Range
            score += 0.1
        return float(np.clip(score, -1.0, 1.0))

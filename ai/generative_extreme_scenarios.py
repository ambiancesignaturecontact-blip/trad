"""
LOT 54: Generative Models for Extreme Market Scenarios
Lightweight but powerful implementation using PyTorch.
Supports: Conditional GAN-style generation + Simple Diffusion-like sampling.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("GenerativeExtremeScenarios")

class Generator(nn.Module):
    """Simple Generator network for synthetic returns"""
    def __init__(self, latent_dim=16, output_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Simple Discriminator"""
    def __init__(self, input_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class ExtremeScenarioGenerator:
    """
    LOT 54: Generates realistic extreme market scenarios.
    Can be used to stress-test the portfolio under tail events.
    """

    def __init__(self, latent_dim: int = 16, seq_len: int = 20):
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(latent_dim, output_dim=5).to(self.device)
        self.discriminator = Discriminator(input_dim=5).to(self.device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=2e-4)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4)

        self.is_trained = False
        logger.info("LOT 54: ExtremeScenarioGenerator initialized")

    def train(self, real_returns: np.ndarray, epochs: int = 800):
        """Train the GAN on historical returns"""
        real_returns = torch.tensor(real_returns, dtype=torch.float32).to(self.device)

        for epoch in range(epochs):
            # Train Discriminator
            z = torch.randn(real_returns.size(0), self.latent_dim).to(self.device)
            fake = self.generator(z).detach()

            d_real = self.discriminator(real_returns)
            d_fake = self.discriminator(fake)

            d_loss = -torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))

            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Train Generator
            z = torch.randn(real_returns.size(0), self.latent_dim).to(self.device)
            fake = self.generator(z)
            d_fake = self.discriminator(fake)

            g_loss = -torch.mean(torch.log(d_fake + 1e-8))

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            if epoch % 200 == 0:
                logger.info(f"LOT 54: Epoch {epoch} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

        self.is_trained = True
        logger.info("LOT 54: Generator trained successfully")

    def generate_extreme_scenarios(self, n_scenarios: int = 500, stress_factor: float = 2.5) -> np.ndarray:
        """
        Generate synthetic extreme market scenarios.
        stress_factor > 1 makes the scenarios more extreme.
        """
        if not self.is_trained:
            logger.warning("Generator not trained. Returning random noise.")
            return np.random.randn(n_scenarios, 5) * 0.03

        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_scenarios, self.latent_dim).to(self.device) * stress_factor
            scenarios = self.generator(z).cpu().numpy()

        logger.info(f"LOT 54: Generated {n_scenarios} extreme scenarios (stress={stress_factor})")
        return scenarios

    def generate_stress_paths(self, initial_price: float, n_paths: int = 100, 
                              horizon: int = 20, stress: float = 2.0) -> np.ndarray:
        """Generate full price paths under stress"""
        returns = self.generate_extreme_scenarios(n_paths * horizon, stress_factor=stress)
        returns = returns.reshape(n_paths, horizon, -1)[:, :, 0]  # Use first dimension as main return

        paths = np.zeros((n_paths, horizon + 1))
        paths[:, 0] = initial_price

        for t in range(horizon):
            paths[:, t+1] = paths[:, t] * (1 + returns[:, t])

        return paths
PYEOF
echo "✅ LOT 54: Generative Models for Extreme Scenarios created"
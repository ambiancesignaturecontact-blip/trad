"""
LOT 53: Advanced Causal Discovery Engine
Combines NOTEARS (if PyTorch available) + PC Algorithm + Causal Graph output.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("CausalDiscovery")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CausalDiscoveryEngine:
    """
    LOT 53: Discovers causal relationships between market variables.
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.causal_graph: Dict[str, List[str]] = {}
        self.causal_strength: Dict[Tuple[str, str], float] = {}

    def discover_causal_graph(self, 
                              returns_dict: Dict[str, np.ndarray],
                              external_signals: Optional[Dict[str, np.ndarray]] = None) -> Dict:
        """
        Main function. Returns a causal graph {variable: [parents]}.
        """
        variables = list(returns_dict.keys())
        if external_signals:
            variables += list(external_signals.keys())

        data = self._build_data_matrix(returns_dict, external_signals)

        if TORCH_AVAILABLE and len(variables) <= 12:
            try:
                graph = self._notears_discovery(data, variables)
                if graph:
                    self.causal_graph = graph
                    logger.info("LOT 53: Causal graph discovered using NOTEARS")
                    return graph
            except Exception as e:
                logger.warning(f"NOTEARS failed, falling back to PC: {e}")

        # Fallback to PC algorithm
        graph = self._pc_algorithm(data, variables)
        self.causal_graph = graph
        logger.info("LOT 53: Causal graph discovered using PC algorithm")
        return graph

    def _build_data_matrix(self, returns_dict, external_signals) -> pd.DataFrame:
        dfs = []
        for name, arr in returns_dict.items():
            dfs.append(pd.Series(arr[-250:], name=name))
        if external_signals:
            for name, arr in external_signals.items():
                dfs.append(pd.Series(arr[-250:], name=name))
        return pd.concat(dfs, axis=1).dropna()

    def _pc_algorithm(self, df: pd.DataFrame, variables: List[str]) -> Dict[str, List[str]]:
        """Simplified PC algorithm using partial correlation threshold"""
        graph = {var: [] for var in variables}
        corr = df.corr().abs()

        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                if corr.loc[var1, var2] > 0.35:  # strong correlation threshold
                    # Simple heuristic: the one with higher variance is more likely the cause
                    if df[var1].std() > df[var2].std():
                        graph[var1].append(var2)
                    else:
                        graph[var2].append(var1)

        return graph

    def _notears_discovery(self, df: pd.DataFrame, variables: List[str]) -> Dict[str, List[str]]:
        """Lightweight NOTEARS implementation (if PyTorch is available)"""
        if not TORCH_AVAILABLE:
            return {}

        data = torch.tensor(df.values, dtype=torch.float32)
        n, d = data.shape

        # Simple linear structural equation model
        W = torch.nn.Parameter(torch.zeros(d, d))

        optimizer = torch.optim.Adam([W], lr=0.01)

        for _ in range(300):
            optimizer.zero_grad()
            X_hat = data @ W
            loss = torch.mean((data - X_hat) ** 2)
            h = torch.trace(torch.matrix_exp(W * W)) - d
            loss = loss + 0.1 * h
            loss.backward()
            optimizer.step()

        W_np = W.detach().numpy()
        graph = {var: [] for var in variables}

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if abs(W_np[i, j]) > 0.15 and i != j:
                    graph[var1].append(var2)

        return graph

    def get_parents(self, variable: str) -> List[str]:
        return self.causal_graph.get(variable, [])

    def get_causal_strength(self, parent: str, child: str) -> float:
        return self.causal_strength.get((parent, child), 0.0)

    def filter_signal_by_causality(self, signal: float, variable: str, 
                                   external_causes: Dict[str, float]) -> float:
        """
        Adjusts a trading signal based on causal parents.
        If a strong causal parent has a conflicting signal, dampen the original signal.
        """
        parents = self.get_parents(variable)
        if not parents:
            return signal

        adjustment = 0.0
        for parent in parents:
            if parent in external_causes:
                parent_signal = external_causes[parent]
                # Simple conflict detection
                if np.sign(parent_signal) != np.sign(signal):
                    adjustment -= 0.25 * abs(parent_signal)

        return np.clip(signal + adjustment, -1.0, 1.0)

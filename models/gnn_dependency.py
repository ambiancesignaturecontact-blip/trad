"""
LOT 43: Dynamic Graph Neural Network (GNN) for Cross-Asset Dependency Modeling
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("GNN_Dependency")

class DynamicGraphBuilder:
    def __init__(self, asset_list: List[str], threshold: float = 0.35):
        self.assets = asset_list
        self.n_assets = len(asset_list)
        self.threshold = threshold
        self.asset_to_idx = {a: i for i, a in enumerate(asset_list)}
        
    def build_graph(self, returns_dict: Dict[str, np.ndarray], 
                    causal_scores: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        corr_matrix = np.eye(self.n_assets)
        
        for i, asset_i in enumerate(self.assets):
            if asset_i not in returns_dict or len(returns_dict[asset_i]) < 5:
                continue
            for j, asset_j in enumerate(self.assets):
                if i >= j or asset_j not in returns_dict or len(returns_dict[asset_j]) < 5:
                    continue
                    
                ri = returns_dict[asset_i][-30:]
                rj = returns_dict[asset_j][-30:]
                min_len = min(len(ri), len(rj))
                if min_len < 5:
                    continue
                    
                corr = np.corrcoef(ri[-min_len:], rj[-min_len:])[0, 1]
                if not np.isnan(corr):
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
        
        if causal_scores:
            for (a1, a2), score in causal_scores.items():
                if a1 in self.asset_to_idx and a2 in self.asset_to_idx:
                    i, j = self.asset_to_idx[a1], self.asset_to_idx[a2]
                    corr_matrix[i, j] = min(0.99, corr_matrix[i, j] + score * 0.3)
        
        edge_list = []
        edge_weights = []
        
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if i != j and abs(corr_matrix[i, j]) > self.threshold:
                    edge_list.append([i, j])
                    edge_weights.append(corr_matrix[i, j])
        
        if not edge_list:
            for i in range(self.n_assets):
                for j in range(self.n_assets):
                    if i != j:
                        edge_list.append([i, j])
                        edge_weights.append(0.1)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        
        return edge_index, edge_attr


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        messages = x[col] * edge_attr
        out = torch.zeros_like(x)
        out.index_add_(0, row, messages)
        out = out + x
        return self.activation(self.linear(out))


class DynamicGNN(nn.Module):
    def __init__(self, n_assets: int, input_dim: int = 4, hidden_dim: int = 32, output_dim: int = 3):
        super().__init__()
        self.n_assets = n_assets
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GraphConvLayer(hidden_dim, hidden_dim)
        self.conv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.dependency_head = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.node_encoder(node_features)
        x = self.dropout(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        
        risk = torch.sigmoid(self.risk_head(x)).squeeze(-1)
        alpha_adj = torch.tanh(self.alpha_head(x)).squeeze(-1)
        dep_logits = self.dependency_head(x)
        
        return {
            "risk_propagation": risk,
            "alpha_adjustment": alpha_adj,
            "dependency_logits": dep_logits
        }


class LiveGNNEngine:
    def __init__(self, asset_list: List[str], device: str = "cpu"):
        self.assets = asset_list
        self.device = torch.device(device)
        self.graph_builder = DynamicGraphBuilder(asset_list)
        self.model = DynamicGNN(n_assets=len(asset_list)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=8e-4)
        self.last_graph_update = 0
        self.graph_cache = None
        self.is_trained = False
        logger.info(f"LOT 43: LiveGNNEngine initialized for {len(asset_list)} assets")

    def _prepare_node_features(self, returns_dict: Dict[str, np.ndarray], prices: Dict[str, float]) -> torch.Tensor:
        features = []
        for asset in self.assets:
            if asset in returns_dict and len(returns_dict[asset]) >= 5:
                rets = returns_dict[asset][-20:]
                vol = float(np.std(rets))
                mom = float(np.mean(rets[-5:]))
                price = prices.get(asset, 100.0)
                log_price = np.log(price) if price > 0 else 0.0
                features.append([vol, mom, log_price, 1.0])
            else:
                features.append([0.02, 0.0, 4.0, 0.0])
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def update_graph_and_predict(self, returns_dict: Dict[str, np.ndarray],
                                  prices: Dict[str, float],
                                  causal_scores: Optional[Dict] = None) -> Dict[str, Dict]:
        edge_index, edge_attr = self.graph_builder.build_graph(returns_dict, causal_scores)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        node_feats = self._prepare_node_features(returns_dict, prices)
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(node_feats, edge_index, edge_attr)
        
        results = {}
        for i, asset in enumerate(self.assets):
            results[asset] = {
                "risk_multiplier": float(1.0 - out["risk_propagation"][i].item() * 0.4),
                "alpha_boost": float(out["alpha_adjustment"][i].item() * 0.25),
                "dependency_vector": out["dependency_logits"][i].cpu().numpy().tolist()
            }
        
        self.graph_cache = (edge_index, edge_attr, node_feats)
        return results

    def online_update(self, target_risk: Dict[str, float]):
        if self.graph_cache is None:
            return False
            
        edge_index, edge_attr, node_feats = self.graph_cache
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(node_feats, edge_index, edge_attr)
        target = torch.tensor([target_risk.get(a, 0.5) for a in self.assets], device=self.device, dtype=torch.float32)
        
        loss = nn.MSELoss()(out["risk_propagation"], target)
        loss.backward()
        self.optimizer.step()
        
        self.is_trained = True
        self.last_graph_update += 1
        
        if self.last_graph_update % 30 == 0:
            logger.info(f"LOT 43: GNN continual update #{self.last_graph_update} | loss={loss.item():.4f}")
        
        return True

    def get_graph_summary(self) -> dict:
        return {
            "assets": self.assets,
            "last_update": self.last_graph_update,
            "trained": self.is_trained
        }

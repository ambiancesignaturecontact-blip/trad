"""
Dynamic Hierarchical Risk Parity (HRP) Allocator (LOT 39)
Modern risk parity with hierarchical clustering.
Much more robust than traditional mean-variance optimization.
"""
import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger("HRPAllocator")

class DynamicHRPAllocator:
    """
    Dynamic Hierarchical Risk Parity.
    - Performs hierarchical clustering on asset/strategy returns
    - Allocates risk equally within clusters (inverse variance)
    - Allocates between clusters proportionally to inverse cluster variance
    """

    def __init__(self, max_clusters: int = 4):
        self.max_clusters = max_clusters
        self.last_weights = None

    def _get_quasi_diagonal(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """Reorder correlation matrix according to hierarchical clustering"""
        dist = np.sqrt((1 - corr_matrix) / 2)
        link = linkage(squareform(dist), method='ward')

        # Get order from dendrogram
        from scipy.cluster.hierarchy import leaves_list
        order = leaves_list(link)

        return corr_matrix.iloc[order, order]

    def _get_cluster_variances(self, corr_matrix: pd.DataFrame,
                               cluster_labels: np.ndarray) -> dict[int, float]:
        """Calculate variance of each cluster"""
        cluster_vars = {}
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            sub_corr = corr_matrix.iloc[mask, mask]
            # Approximate cluster variance
            cluster_vars[cluster_id] = sub_corr.values.mean()
        return cluster_vars

    def compute_hrp_weights(self, returns_df: pd.DataFrame) -> pd.Series:
        """
        Main HRP allocation.
        returns_df: DataFrame with assets/strategies as columns
        """
        if returns_df.shape[1] < 2:
            return pd.Series(1.0 / returns_df.shape[1], index=returns_df.columns)

        # Correlation matrix
        corr = returns_df.corr()

        # Hierarchical clustering
        dist = np.sqrt((1 - corr) / 2)
        link = linkage(squareform(dist), method='ward')
        cluster_labels = fcluster(link, t=self.max_clusters, criterion='maxclust')

        # Quasi-diagonal reordering
        corr_ordered = self._get_quasi_diagonal(corr)

        # Inverse variance within clusters
        inv_var = 1 / returns_df.var()

        # Cluster-level allocation
        cluster_vars = self._get_cluster_variances(corr_ordered, cluster_labels)
        cluster_weights = {c: 1/v for c, v in cluster_vars.items()}
        total = sum(cluster_weights.values())
        cluster_weights = {c: w/total for c, w in cluster_weights.items()}

        # Final weights
        weights = pd.Series(0.0, index=returns_df.columns)
        for i, col in enumerate(returns_df.columns):
            cluster = cluster_labels[i]
            weights[col] = inv_var[col] * cluster_weights[cluster]

        # Normalize
        weights = weights / weights.sum()
        self.last_weights = weights

        return weights

    def get_allocation(self, returns_dict: dict[str, np.ndarray]) -> dict[str, float]:
        """Convenience method"""
        df = pd.DataFrame(returns_dict)
        weights = self.compute_hrp_weights(df)
        return weights.to_dict()

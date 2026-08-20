"""
LOT 48+: Robust Feature Store with Versioning
Professional-grade feature management for trading.
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime

import pandas as pd

logger = logging.getLogger("RobustFeatureStore")

class FeatureStore:
    """
    LOT 48+: Robust Feature Store with versioning, metadata, and caching.
    """

    def __init__(self, storage_path: str = "feature_store.json"):
        self.storage_path = storage_path
        self.features: dict[str, dict] = {}           # symbol -> {version: features}
        self.metadata: dict[str, dict] = {}           # symbol -> {version: metadata}
        self.versions: dict[str, list[str]] = defaultdict(list)
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    self.features = data.get("features", {})
                    self.metadata = data.get("metadata", {})
                    self.versions = defaultdict(list, data.get("versions", {}))
                logger.info(f"Feature Store loaded: {len(self.features)} symbols")
            except Exception as e:
                logger.error(f"Failed to load Feature Store: {e}")

    def _save(self):
        try:
            data = {
                "features": self.features,
                "metadata": self.metadata,
                "versions": dict(self.versions),
                "last_updated": datetime.now().isoformat()
            }
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save Feature Store: {e}")

    def compute_features(self, symbol: str, df: pd.DataFrame, version: str = "v1.0") -> dict:
        """Compute and store features with versioning."""
        if df.empty or len(df) < 20:
            return {}

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        features = {
            "returns_1": float(close.pct_change(1).iloc[-1]),
            "returns_5": float(close.pct_change(5).iloc[-1]),
            "volatility_20": float(close.pct_change().rolling(20).std().iloc[-1]),
            "rsi_14": float(self._rsi(close, 14)),
            "atr_14": float(self._atr(high, low, close, 14)),
            "volume_zscore": float(((volume - volume.rolling(20).mean()) / volume.rolling(20).std()).iloc[-1]),
            "price_vs_sma20": float((close.iloc[-1] / close.rolling(20).mean().iloc[-1]) - 1),
            "momentum_10": float(close.iloc[-1] / close.iloc[-10] - 1),
            "computed_at": datetime.now().isoformat()
        }

        if symbol not in self.features:
            self.features[symbol] = {}
            self.metadata[symbol] = {}
            self.versions[symbol] = []

        self.features[symbol][version] = features
        self.metadata[symbol][version] = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "rows_used": len(df),
            "source": "live"
        }

        if version not in self.versions[symbol]:
            self.versions[symbol].append(version)

        self._save()
        logger.info(f"Feature Store: Computed {version} for {symbol}")

        return features

    def _rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]

    def _atr(self, high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def get_features(self, symbol: str, version: str | None = None) -> dict | None:
        """Get features. If version is None, returns latest version."""
        if symbol not in self.features:
            return None

        if version is None:
            # Return latest version
            versions = self.versions.get(symbol, [])
            if not versions:
                return None
            version = versions[-1]

        return self.features.get(symbol, {}).get(version)

    def get_metadata(self, symbol: str, version: str | None = None) -> dict | None:
        if symbol not in self.metadata:
            return None
        if version is None:
            versions = self.versions.get(symbol, [])
            if not versions:
                return None
            version = versions[-1]
        return self.metadata.get(symbol, {}).get(version)

    def list_versions(self, symbol: str) -> list[str]:
        return self.versions.get(symbol, [])

    def get_status(self) -> dict:
        return {
            "symbols": list(self.features.keys()),
            "total_symbols": len(self.features),
            "storage_path": self.storage_path
        }

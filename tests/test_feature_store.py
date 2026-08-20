import pandas as pd

from core.feature_store import FeatureStore


def test_feature_store_computes_features(tmp_path):
    fs = FeatureStore(storage_path=str(tmp_path / "fs.json"))

    df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105] * 10,
        'high': [101, 102, 103, 104, 105, 106] * 10,
        'low': [99, 100, 101, 102, 103, 104] * 10,
        'volume': [1000] * 60
    })

    features = fs.compute_features("BTCUSDT", df, version="v1.0")

    assert "rsi_14" in features
    assert "volatility_20" in features
    assert features["rsi_14"] > 0
    assert 0 < features["volatility_20"] < 1

def test_feature_store_versioning(tmp_path):
    fs = FeatureStore(storage_path=str(tmp_path / "fs.json"))
    df = pd.DataFrame({
        'close': [100] * 30,
        'high': [101] * 30,
        'low': [99] * 30,
        'volume': [1000] * 30
    })

    fs.compute_features("ETHUSDT", df, version="v1.0")
    fs.compute_features("ETHUSDT", df, version="v2.0")

    versions = fs.list_versions("ETHUSDT")
    assert "v1.0" in versions
    assert "v2.0" in versions

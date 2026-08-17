import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_returns():
    import numpy as np
    return {
        "BTCUSDT": np.random.randn(100) * 0.02,
        "ETHUSDT": np.random.randn(100) * 0.025
    }

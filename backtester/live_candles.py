"""
Données de marché RÉELLES pour les scripts CLI de backtest (P0-5, audit §4.9).

Binance REST est géobloqué depuis la France (HTTP 451) : ces scripts utilisaient
un fallback SYNTHÉTIQUE silencieux — leurs « preuves » de rentabilité étaient
donc construites sur des données fabriquées. Ce module remplace cela par de
vraies bougies horaires publiques accessibles depuis la France :

    OKX -> Coinbase -> Kraken -> Binance (dernier recours)

Règle d'honnêteté : PAS de fallback synthétique. Si aucune source réelle ne
répond, fetch_real_candles() retourne (None, "") et le script CLI s'arrête
avec un message clair — « AUCUNE DONNÉE RÉELLE -> AUCUNE PREUVE ».
"""
from typing import Optional, Tuple

import httpx
import pandas as pd

# Ordre de tentative des sources (toutes publiques, sans clé)
SOURCES_ORDER = ["okx", "coinbase", "kraken", "binance"]

# Mapping des symboles internes (BTCUSDT...) vers chaque source
SYMBOL_MAP = {
    "BTCUSDT": {"okx": "BTC-USDT", "coinbase": "BTC-USD",
                "kraken": "XBTUSD", "binance": "BTCUSDT"},
    "ETHUSDT": {"okx": "ETH-USDT", "coinbase": "ETH-USD",
                "kraken": "ETHUSD", "binance": "ETHUSDT"},
    "SOLUSDT": {"okx": "SOL-USDT", "coinbase": "SOL-USD",
                "kraken": "SOLUSD", "binance": "SOLUSDT"},
}

# OKX et Coinbase plafonnent les barres par requête (300)
_MAX_BARS = 300


def _fetch_okx(symbol: str, limit: int) -> Optional[pd.DataFrame]:
    inst = SYMBOL_MAP[symbol]["okx"]
    url = f"https://www.okx.com/api/v5/market/candles?instId={inst}&bar=1H&limit={min(limit, _MAX_BARS)}"
    r = httpx.get(url, timeout=10.0)
    data = r.json().get("data", [])
    rows = []
    for b in data:  # OKX renvoie l'ordre DESC (plus récent en premier)
        rows.append({"timestamp": pd.to_datetime(int(b[0]), unit="ms"),
                     "open": float(b[1]), "high": float(b[2]), "low": float(b[3]),
                     "close": float(b[4]), "volume": float(b[5])})
    rows.reverse()
    return pd.DataFrame(rows).set_index("timestamp") if rows else None


def _fetch_coinbase(symbol: str, limit: int) -> Optional[pd.DataFrame]:
    prod = SYMBOL_MAP[symbol]["coinbase"]
    url = f"https://api.exchange.coinbase.com/products/{prod}/candles?granularity=3600"
    r = httpx.get(url, timeout=10.0)
    data = r.json()  # [ts_s, low, high, open, close, volume] — ordre ASC
    rows = [{"timestamp": pd.to_datetime(int(b[0]), unit="s"),
             "open": float(b[3]), "high": float(b[2]), "low": float(b[1]),
             "close": float(b[4]), "volume": float(b[5])} for b in data]
    rows.sort(key=lambda x: x["timestamp"])
    return pd.DataFrame(rows).set_index("timestamp")[-limit:] if rows else None


def _fetch_kraken(symbol: str, limit: int) -> Optional[pd.DataFrame]:
    pair = SYMBOL_MAP[symbol]["kraken"]
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=60"
    r = httpx.get(url, timeout=10.0)
    result = r.json().get("result", {})
    # Kraken renvoie la paire NORMALISÉE (ex. XXBTZUSD) — on la cherche
    data = []
    for k, v in result.items():
        if k != "last" and isinstance(v, list) and v:
            data = v
            break
    rows = [{"timestamp": pd.to_datetime(int(b[0]), unit="s"),
             "open": float(b[1]), "high": float(b[2]), "low": float(b[3]),
             "close": float(b[4]), "volume": float(b[6])} for b in data]
    rows.sort(key=lambda x: x["timestamp"])
    return pd.DataFrame(rows).set_index("timestamp")[-limit:] if rows else None


def _fetch_binance(symbol: str, limit: int) -> Optional[pd.DataFrame]:
    s = SYMBOL_MAP[symbol]["binance"]
    url = f"https://api.binance.com/api/v3/klines?symbol={s}&interval=1h&limit={min(limit, _MAX_BARS)}"
    r = httpx.get(url, timeout=10.0)
    data = r.json()
    rows = [{"timestamp": pd.to_datetime(int(b[0]), unit="ms"),
             "open": float(b[1]), "high": float(b[2]), "low": float(b[3]),
             "close": float(b[4]), "volume": float(b[5])} for b in data]
    return pd.DataFrame(rows).set_index("timestamp") if rows else None


_FETCHERS = {"okx": _fetch_okx, "coinbase": _fetch_coinbase,
             "kraken": _fetch_kraken, "binance": _fetch_binance}


def fetch_real_candles(symbol: str = "BTCUSDT", limit: int = 500,
                       verbose: bool = True) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Récupère de VRAIES bougies horaires (open/high/low/close/volume) pour le
    symbole demandé. Essaie OKX -> Coinbase -> Kraken -> Binance.

    Retourne (df, source) avec df=None si AUCUNE source réelle ne répond —
    jamais de données synthétiques. Le DataFrame a un DatetimeIndex croissant
    et les colonnes open/high/low/close/volume.
    """
    if symbol not in SYMBOL_MAP:
        if verbose:
            print(f"❌ {symbol}: symbole non supporté par live_candles ("
                  f"supporté: {list(SYMBOL_MAP.keys())})")
        return None, ""
    for src in SOURCES_ORDER:
        try:
            df = _FETCHERS[src](symbol, limit)
            if df is not None and len(df) >= 100:
                if verbose:
                    print(
                        f"✅ Données RÉELLES : {len(df)} barres 1h {symbol} via "
                        f"{src.upper()} ({df.index[0]} -> {df.index[-1]})"
                    )
                return df, src
        except Exception as e:
            if verbose:
                print(f"   {src}: indisponible ({e.__class__.__name__})")
    if verbose:
        print("❌ AUCUNE source de données réelles disponible -> pas de backtest "
              "(une preuve sur données synthétiques ne prouve rien).")
    return None, ""

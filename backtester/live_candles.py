"""
Données de marché RÉELLES pour les scripts CLI de backtest (P0-5, audit §4.9).

Binance REST est géobloqué depuis la France (HTTP 451) : ces scripts utilisaient
un fallback SYNTHÉTIQUE silencieux — leurs « preuves » de rentabilité étaient
donc construites sur des données fabriquées. Ce module remplace cela par de
vraies bougies horaires publiques accessibles depuis la France :

    OKX -> Coinbase -> Kraken -> Binance (dernier recours)

OKX et Coinbase plafonnent à 300 barres par requête : ce module PAGINE pour
atteindre l'historique demandé (jusqu'à MAX_BARS=1200 barres ≈ 50 jours 1h).

Règle d'honnêteté : PAS de fallback synthétique. Si aucune source réelle ne
répond, fetch_real_candles() retourne (None, "") et le script CLI s'arrête
avec un message clair — « AUCUNE DONNÉE RÉELLE -> AUCUNE PREUVE ».
"""
from typing import List, Optional, Tuple

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

# Borne dure par source et globale (barres 1h)
_PAGE = 300                 # OKX / Coinbase : max barres par requête
KRAKEN_NATIVE = 720         # Kraken renvoie 720 bougies 1h par appel
MAX_BARS = 1200             # borne dure globale (≈ 50 jours de données 1h)


def _build_frame(rows: List[dict], limit: int) -> Optional[pd.DataFrame]:
    if not rows:
        return None
    rows.sort(key=lambda x: x["timestamp"])
    df = pd.DataFrame(rows).set_index("timestamp")
    # colonnes canoniques
    df = df[["open", "high", "low", "close", "volume"]]
    return df.tail(limit)


def _fetch_okx(symbol: str, limit: int) -> Optional[pd.DataFrame]:
    """OKX : pagination via `after` (ts ms de la bougie la plus ancienne)."""
    inst = SYMBOL_MAP[symbol]["okx"]
    rows, after, last_after = [], None, None
    while len(rows) < limit and len(rows) < MAX_BARS:
        url = (f"https://www.okx.com/api/v5/market/candles?instId={inst}"
               f"&bar=1H&limit={min(_PAGE, limit)}")
        if after is not None:
            url += f"&after={after}"
        r = httpx.get(url, timeout=10.0)
        data = r.json().get("data", [])  # ordre DESC (récent en premier)
        if not data:
            break
        for b in data:
            rows.append({"timestamp": pd.to_datetime(int(b[0]), unit="ms"),
                         "open": float(b[1]), "high": float(b[2]), "low": float(b[3]),
                         "close": float(b[4]), "volume": float(b[5])})
        last_after = after
        after = int(data[-1][0])
        if after == last_after:      # page identique -> arrêt (évite la boucle)
            break
    return _build_frame(rows, limit)


def _fetch_coinbase(symbol: str, limit: int) -> Optional[pd.DataFrame]:
    """Coinbase : pagination par fenêtres start/end (max 300 par requête)."""
    prod = SYMBOL_MAP[symbol]["coinbase"]
    rows, end = [], None
    while len(rows) < limit and len(rows) < MAX_BARS:
        url = f"https://api.exchange.coinbase.com/products/{prod}/candles?granularity=3600"
        if end is not None:
            start = end - _PAGE * 3600
            url += f"&start={start}&end={end}"
        r = httpx.get(url, timeout=10.0)
        data = r.json()  # [ts_s, low, high, open, close, volume] — ordre ASC
        if not isinstance(data, list) or not data:
            break
        for b in data:
            rows.append({"timestamp": pd.to_datetime(int(b[0]), unit="s"),
                         "open": float(b[3]), "high": float(b[2]), "low": float(b[1]),
                         "close": float(b[4]), "volume": float(b[5])})
        first_ts = min(int(b[0]) for b in data)
        if end is not None and first_ts >= end:
            break
        end = first_ts - 3600
    return _build_frame(rows, limit)


def _fetch_kraken(symbol: str, limit: int) -> Optional[pd.DataFrame]:
    """Kraken : 720 bougies 1h natives (suffisant jusqu'à 720)."""
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
    return _build_frame(rows, limit)


def _fetch_binance(symbol: str, limit: int) -> Optional[pd.DataFrame]:
    s = SYMBOL_MAP[symbol]["binance"]
    url = (f"https://api.binance.com/api/v3/klines?symbol={s}"
           f"&interval=1h&limit={min(limit, MAX_BARS)}")
    r = httpx.get(url, timeout=10.0)
    data = r.json()
    rows = [{"timestamp": pd.to_datetime(int(b[0]), unit="ms"),
             "open": float(b[1]), "high": float(b[2]), "low": float(b[3]),
             "close": float(b[4]), "volume": float(b[5])} for b in data]
    return _build_frame(rows, limit)


_FETCHERS = {"okx": _fetch_okx, "coinbase": _fetch_coinbase,
             "kraken": _fetch_kraken, "binance": _fetch_binance}


def fetch_real_candles(symbol: str = "BTCUSDT", limit: int = 500,
                       verbose: bool = True) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Récupère de VRAIES bougies horaires (open/high/low/close/volume) pour le
    symbole demandé (pagination incluse). Essaie OKX -> Coinbase -> Kraken ->
    Binance.

    Retourne (df, source) avec df=None si AUCUNE source réelle ne répond —
    jamais de données synthétiques. Le DataFrame a un DatetimeIndex croissant
    et les colonnes open/high/low/close/volume.
    """
    limit = min(int(limit), MAX_BARS)
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

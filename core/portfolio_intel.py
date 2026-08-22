"""
PORTFOLIO INTELLIGENCE (PHASE 4 — P4-A, axe 2 du mandat).

« Le cerveau ne regarde plus seulement les trades individuels : il regarde le
portefeuille entier comme un système unique. Un excellent trade peut être
refusé simplement parce que le portefeuille est déjà trop exposé au même
facteur de risque. »

Ce module fournit :
  1. `estimate_beta`        : bêta causal (régression OLS sur fenêtre glissante)
     d'un actif vs BTC — l'exposition « crypto regime factor » ;
  2. `refresh_beta_cache`   : cache STATE {symbole: bêta} rafraîchi ≤ 6 h à
     partir des candles RÉELLES (jamais de chiffre inventé) ;
  3. `portfolio_exposures`  : exposition nette du portefeuille (positions
     ouvertes valorisées) au facteur BTC, en % de l'équité ;
  4. `exposure_gate_blocks` : décision PURE — bloque le trade candidat si :
       (a) l'exposition nette au facteur BTC APRÈS ajout dépasse une limite
           (config portfolio.max_btc_beta_exposure_pct, défaut 50 % équité) ;
       (b) la corrélation de l'actif candidat avec une position existante
           dépasse portfolio.redundant_corr (0.85) DANS LE MÊME SENS
           (concentration corrélée : « déjà trop exposé au même facteur »).

Principes :
  - JAMAIS bloquant : sans bêta/corrélation mesurés (échantillon insuffisant),
    AUCUN blocage — pas de chiffre inventé.
  - DÉMO == RÉAL : aucun flag de mode.
  - La décision est une fonction PURE (testable) ; le gate est câblé dans
    main.py après le gate de friction, avant le journal de décision.
"""
import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger("InstitutionalTradingBot")

BETA_MIN_SAMPLES = 240            # ~10 jours de barres 1h pour un bêta stable
BETA_CACHE_MAX_AGE = 6 * 3600     # le bêta est lent : cache 6 h


# --------------------------------------------------------------------------- #
# Bêta causal (régression OLS : cov/var sur fenêtre glissante)
# --------------------------------------------------------------------------- #
def estimate_beta(asset_ret: pd.Series, btc_ret: pd.Series,
                  window: int = 720) -> float | None:
    """Bêta OLS de l'actif vs BTC sur la fenêtre glissante (causale : seules
    les données passées jusqu'à la dernière barre). None si échantillon
    insuffisant ou variance BTC nulle — jamais de valeur inventée."""
    if asset_ret is None or btc_ret is None:
        return None
    a = asset_ret.astype(float).dropna().tail(window)
    b = btc_ret.astype(float).dropna().tail(window)
    common = a.index.intersection(b.index)
    if len(common) < BETA_MIN_SAMPLES:
        return None
    a, b = a.loc[common], b.loc[common]
    var_b = float(b.var())
    if not var_b > 0 or not np.isfinite(var_b):
        return None
    beta = float(a.cov(b) / var_b)
    return beta if np.isfinite(beta) else None


def refresh_beta_cache(state: dict, db, max_age_sec: float = BETA_CACHE_MAX_AGE,
                       symbols=None) -> dict:
    """Cache STATE['asset_btc_beta'] = {symbole: bêta} rafraîchi au plus tous
    les max_age_sec. Nécessite les candles réelles (BTC + actifs). Jamais
    bloquant ; sans données -> cache vide (pas de blocage)."""
    try:
        cache = state.get("asset_btc_beta") or {}
        ts = state.get("asset_btc_beta_ts") or 0.0
        if time.time() - ts < max_age_sec:
            return cache
        if db is None or not hasattr(db, "load_candles"):
            return cache
        syms = symbols or ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSD",
                           "EURUSD", "AAPL", "TSLA")
        btc = db.load_candles("BTCUSDT", limit=2000)
        if btc is None or btc.empty:
            return cache
        btc_ret = btc["close"].astype(float).pct_change()
        out = {}
        for s in syms:
            try:
                df = db.load_candles(s, limit=2000)
                if df is None or df.empty:
                    continue
                ret = df["close"].astype(float).pct_change()
                beta = estimate_beta(ret, btc_ret)
                if beta is not None:
                    out[s] = round(float(beta), 3)
            except Exception:
                continue
        state["asset_btc_beta"] = out
        state["asset_btc_beta_ts"] = time.time()
        return out
    except Exception as e:
        logger.debug(f"refresh_beta_cache failed: {e}")
        return state.get("asset_btc_beta") or {}


# --------------------------------------------------------------------------- #
# Exposition du portefeuille + corrélations
# --------------------------------------------------------------------------- #
def _returns_series(db, symbol: str, limit: int = 2000) -> pd.Series | None:
    try:
        df = db.load_candles(symbol, limit=limit)
        if df is None or df.empty:
            return None
        return df["close"].astype(float).pct_change().dropna()
    except Exception:
        return None


def portfolio_exposures(state: dict, db, betas: dict | None = None,
                        corr_window: int = 720) -> dict:
    """Exposition nette du portefeuille au facteur BTC (% équité) et carte
    des positions. betas : cache STATE['asset_btc_beta']. Sans prix/quantité
    -> expositions vides."""
    out = {"equity": None, "positions": [], "btc_beta_exposure_pct": 0.0,
           "n_positions": 0, "correlations": {}}
    try:
        equity = float(state.get("current_equity") or 0.0)
        if equity <= 0:
            return out
        out["equity"] = round(equity, 2)
        positions = db.get_positions() if hasattr(db, "get_positions") else []
        betas = betas or {}
        prices = state.get("last_known_prices") or {}
        net = 0.0
        corr = {}
        for p in positions:
            sym = str(p.get("symbol", ""))
            qty = float(p.get("qty") or 0.0)
            if abs(qty) < 1e-12:
                continue
            price = prices.get(sym) or float(p.get("avg_price") or 0.0)
            if price <= 0:
                continue
            notional = qty * price
            beta = betas.get(sym)
            btc_exp = notional * (beta if beta is not None else 0.0)
            net += btc_exp
            out["positions"].append({
                "symbol": sym, "qty": qty, "price": round(price, 6),
                "notional": round(notional, 4),
                "btc_beta": beta, "btc_exposure": round(btc_exp, 4),
            })
            # corrélation avec les AUTRES positions ouvertes (causale)
            if db is not None and sym not in corr:
                ret_s = _returns_series(db, sym)
                if ret_s is not None:
                    for p2 in positions:
                        s2 = str(p2.get("symbol", ""))
                        if s2 == sym or s2 in corr.get(sym, {}):
                            continue
                        ret_o = _returns_series(db, s2)
                        if ret_o is None:
                            continue
                        common = ret_s.index.intersection(ret_o.index)
                        if len(common) < BETA_MIN_SAMPLES:
                            continue
                        c = float(ret_s.loc[common].corr(ret_o.loc[common]))
                        if np.isfinite(c):
                            corr.setdefault(sym, {})[s2] = round(c, 3)
        out["n_positions"] = len(out["positions"])
        out["btc_beta_exposure_pct"] = round(net / equity * 100.0, 2)
        out["correlations"] = corr
    except Exception as e:
        logger.debug(f"portfolio_exposures failed: {e}")
    return out


# --------------------------------------------------------------------------- #
# Décision PURE : le trade candidat est-il refusé ?
# --------------------------------------------------------------------------- #
def exposure_gate_blocks(symbol: str, side: str, qty: float, price: float,
                         equity: float, betas: dict,
                         positions: list[dict],
                         max_btc_beta_pct: float = 50.0,
                         redundant_corr: float = 0.85,
                         correlations: dict | None = None) -> tuple:
    """
    (block, reason, detail) — PURE et testable.
      (a) Facteur BTC : net actuel + exposition du candidat ; si |net| dépasse
          max_btc_beta_pct % de l'équité -> blocage.
      (b) Concentration corrélée : si la corrélation du candidat avec une
          position existante > redundant_corr DANS LE MÊME SENS -> blocage
          (« déjà trop exposé au même facteur »).
    Sans bêta ni corrélation mesurés -> (False, None, None) : pas de blocage.
    """
    if not equity or equity <= 0 or not price or price <= 0:
        return False, None, None
    beta = betas.get(symbol)
    # (a) facteur BTC
    if beta is not None:
        net_now = 0.0
        for p in positions:
            b = betas.get(str(p.get("symbol", "")))
            if b is None:
                continue
            notional = float(p.get("qty") or 0.0) * \
                (float(p.get("price") or float(p.get("avg_price") or 0.0)))
            net_now += notional * b
        cand_notional = qty * price
        direction = 1.0 if str(side).upper() == "BUY" else -1.0
        net_after = net_now + direction * cand_notional * beta
        exp_pct = net_after / equity * 100.0
        if abs(exp_pct) > max_btc_beta_pct:
            return (True,
                    "portfolio_exposure: facteur BTC",
                    (f"exposition nette au facteur BTC après ajout "
                     f"{exp_pct:.1f}% de l'équité > limite "
                     f"{max_btc_beta_pct:.0f}% (bêta {beta:.2f}, "
                     f"notionnel {cand_notional:.2f}$) — le portefeuille est "
                     f"déjà trop exposé au même facteur"))
    # (b) concentration corrélée (même sens)
    corr_map = correlations or {}
    for p in positions:
        s2 = str(p.get("symbol", ""))
        if s2 == symbol or abs(float(p.get("qty") or 0.0)) < 1e-12:
            continue
        c = corr_map.get(symbol, {}).get(s2)
        if c is None:
            c = corr_map.get(s2, {}).get(symbol)
        if c is not None and abs(c) > redundant_corr:
            same_dir = (float(p.get("qty")) > 0) == (str(side).upper() == "BUY")
            if same_dir:
                return (True,
                        "portfolio_exposure: concentration corrélée",
                        (f"corr({symbol},{s2}) = {c:.2f} > "
                         f"{redundant_corr:.2f} dans le même sens — le "
                         f"portefeuille est déjà exposé au même facteur"))
    return False, None, None

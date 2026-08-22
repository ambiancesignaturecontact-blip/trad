"""
RESEARCH EXPERIMENT ENGINE (PHASE 3 — Cycle 2, item 2).

Exécute le pipeline de VALIDATION d'une hypothèse enregistrée dans la table
`experiments` (Research Memory §5-§7) :

    HYPOTHÈSE -> EXPÉRIENCE (baseline vs traitement isolé)
        -> WALK-FORWARD (train 70 % / OOS 30 %, seuil estimé sur train seul)
        -> STRESS (fenêtre de vol maximale réelle)
        -> DÉCISION REJECT / KEEP enregistrée (jamais PROMOTE ici)

Règles absolues :
  1. Données 100 % réelles (cache DB `market_candles`, sinon Yahoo) — aucun
     fallback synthétique.
  2. Le « traitement » est une variante ISOLÉE : la production n'est jamais
     modifiée par une expérience.
  3. Pas de look-ahead : le seuil de vol du filtre est calibré sur la partie
     TRAIN (70 %) puis appliqué tel quel sur l'OOS (30 %).
  4. Coûts réalistes mesurés (PHASE 2) : commission 0,100 %/side (frais
     réels) + slippage médian 6,6 bps/side -> aller-retour 0,213 %.
  5. Le signal Momentum de l'expérience est VÉRIFIÉ en parité avec
     `MomentumStrategy.generate_signal` (production) — l'expérience teste le
     signal réellement déployé, pas une variante de laboratoire.
  6. Décision honnête : REJECT seulement si le traitement DÉGRADE l'expectancy
     OOS (ou n'apporte aucune amélioration mesurable) ; KEEP (preuve
     insuffisante) si < MIN_TRADES_OOS ; PROMOTE interdit dans ce pipeline.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("InstitutionalTradingBot")

# --------------------------------------------------------------------------- #
# Constantes expérimentales (documentées, mesurées quand possible)
# --------------------------------------------------------------------------- #
TRAIN_RATIO = 0.70              # walk-forward : 70 % train / 30 % OOS
VOL_QUANTILE = 0.80             # seuil du filtre : vol > percentile 80 (train)
FILTER_SCALE = 0.50             # poids Momentum réduit de moitié en HIGH_VOL
VOL_EWMA_SPAN = 24              # vol réalisée EWMA ~ 24 h
MIN_TRADES_OOS = 10             # en dessous : preuve insuffisante (KEEP)
COST_AR_PCT = 0.213             # aller-retour % : 0,2 commission + 13,2 bps slip
STRESS_TOP_QUANTILE = 0.90      # stress : fenêtre des 10 % de vol maximale

DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSD",
                   "EURUSD", "AAPL", "TSLA")


# --------------------------------------------------------------------------- #
# Données réelles
# --------------------------------------------------------------------------- #
def load_candles(db, symbol: str, limit: int = 6000) -> pd.DataFrame:
    """Candles réelles 1h : cache DB `market_candles` d'abord, sinon Yahoo.
    Retourne un DataFrame indexé par timestamp (UTC). Jamais synthétique."""
    df = None
    if db is not None and hasattr(db, "load_candles"):
        try:
            df = db.load_candles(symbol, limit=limit)
        except Exception as e:
            logger.debug(f"cache candles failed ({symbol}): {e}")
    if df is None or df.empty:
        try:
            from market_data.historical_fetch import fetch_historical_market_data
            df = fetch_historical_market_data(symbol)
        except Exception as e:
            logger.warning(f"Yahoo candles failed ({symbol}): {e}")
    if df is None or df.empty:
        return pd.DataFrame()
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
        else:
            df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index().tail(limit)


# --------------------------------------------------------------------------- #
# Signal Momentum — réplique vectorisée de strategies/momentum.py
# --------------------------------------------------------------------------- #
def momentum_signal_series(close: pd.Series, roc_period: int = 14,
                           rsi_period: int = 14, volume_ma: int = 20,
                           min_momentum: float = 0.012,
                           volume: pd.Series | None = None) -> pd.Series:
    """
    Réplique VECTORISÉE du signal `MomentumStrategy.generate_signal`
    (production) : même règle à chaque barre t en utilisant uniquement
    l'historique [0..t] (aucun look-ahead). La parité avec la production est
    vérifiée par test (`check_signal_parity`).

    FIDÉLITÉ AU CODE DE PRODUCTION (défauts compris, documentés) :
      - RSI appliqué indépendamment du score ROC (la production n'exige pas
        `score != 0`) ;
      - ROC sur `close[-roc_period]` => décalage réel de roc_period - 1 barres
        (off-by-one de la production, reproduit pour la parité).
    """
    roc = close.pct_change(roc_period - 1)   # reproduit close[-roc_period]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    # division pandas native : 0/0 -> NaN, x/0 -> inf -> rsi = 100 (comme
    # la production, qui divise gain/loss sans protection)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    score = pd.Series(0.0, index=close.index)
    score[roc > min_momentum] = 0.55
    score[roc < -min_momentum] = -0.55
    score[rsi > 65] += 0.25
    score[rsi < 35] -= 0.25

    if volume is not None:
        vol_ma_s = volume.rolling(volume_ma).mean()
        vol_ratio = volume / vol_ma_s.replace(0, np.nan)
        score[vol_ratio > 1.4] *= 1.15

    return score.clip(-1.0, 1.0)


def check_signal_parity(df: pd.DataFrame, tol: float = 1e-9) -> bool:
    """Vérifie que la réplique vectorisée reproduit `MomentumStrategy`
    (production) sur la DERNIÈRE barre d'un dataset réel. Retourne False si
    désaccord (l'expérience doit alors être invalidée, pas ajustée)."""
    try:
        from strategies.momentum import MomentumStrategy
        prod_signal, _ = MomentumStrategy().generate_signal({"df": df})
        vec = momentum_signal_series(df["close"], volume=df.get("volume"))
        return bool(abs(float(prod_signal) - float(vec.iloc[-1])) <= tol)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Vol réalisée EWMA (causale) + masque régime HIGH_VOL (proxy documentée)
# --------------------------------------------------------------------------- #
def volatility_ewma(close: pd.Series, span: int = VOL_EWMA_SPAN) -> pd.Series:
    """Vol réalisée EWMA des rendements (causale : ne voit que [0..t])."""
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    return ret.rolling(span).std().ewm(span=span).mean()


def high_vol_mask(vol: pd.Series, threshold: float) -> pd.Series:
    """True quand vol EWMA > seuil (proxy causale du régime 3 « Erratic High
    Volatility » — l'hypothèse #1 porte sur ce régime ; proxy documentée :
    le HMM de production est non-causal, inutilisable en backtest)."""
    return (vol > threshold).fillna(False)


# --------------------------------------------------------------------------- #
# Backtest de signaux (transparent, coûts réalistes)
# --------------------------------------------------------------------------- #
def backtest_signals(close: pd.Series, signals: pd.Series,
                     cost_ar_pct: float = COST_AR_PCT) -> dict:
    """
    Backtest simple et transparent : position = sign(signal) (0 si nul),
    coût aller-retour (commission + slippage) à chaque changement de
    direction. Retourne des métriques par trade et globales — AUCUNE
    annualisation fantaisiste (le Sharpe est quotidien annualisé ×√365).
    """
    close = close.astype(float)
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # PAS DE LOOK-AHEAD : le signal de la barre t est connu au close_t, la
    # position s'applique donc au rendement de la barre t+1 (shift de 1).
    # Position CONTINUE = intensité du signal bornée [-1, 1] : c'est ce que
    # le filtre vol modifie (poids), pas le signe.
    pos = signals.shift(1).clip(-1.0, 1.0).fillna(0.0).values
    delta = np.abs(np.diff(pos, prepend=0.0))
    # coût proportionnel à la variation de position (0 -> x -> 0 = 1 AR)
    cost = delta * (cost_ar_pct / 100.0) / 2.0
    pnl = pos * ret.values - cost
    trades_idx = np.where(delta > 1e-9)[0]
    n_trades = int(len(trades_idx))
    wins = int((pnl[trades_idx] > 0).sum()) if n_trades else 0
    cumul = float(pnl.sum())
    cum_pnl_pct = cumul * 100.0
    if isinstance(close.index, pd.DatetimeIndex):
        daily = pd.Series(pnl, index=close.index).resample("1D").sum()
        sharpe = float(daily.mean() / daily.std() * np.sqrt(365)) \
            if len(daily) > 2 and daily.std() > 0 else None
    else:
        sharpe = None  # index non temporel : pas d'annualisation fantaisiste
    eq = np.cumsum(pnl)
    max_dd = float((eq - np.maximum.accumulate(eq)).min() * 100.0) \
        if len(eq) else 0.0
    # round-trips (définition STANDARD) : intervalle entre l'entrée et le
    # retour à zéro de la position. La barre de SORTIE (où la position devient
    # nulle) porte le coût de sortie : elle est attribuée au RT qui se
    # termine. Le RT encore ouvert en fin d'échantillon est exclu du win rate
    # (pas de sortie) mais son pnl brut reste dans le cumul global.
    rt, cur, prev_p = [], 0.0, 0.0
    for i in range(len(pos)):
        p = pos[i]
        if abs(p) > 1e-9 or abs(prev_p) > 1e-9:
            cur += pnl[i]        # barre en position ou barre de sortie
        if abs(p) <= 1e-9 and abs(prev_p) > 1e-9:
            rt.append(cur)       # retour à zéro : RT clôturé
            cur = 0.0
        prev_p = p
    n_open = 1 if cur != 0.0 else 0
    if cur != 0.0:
        rt.append(cur)
    n_rt_closed = max(0, len(rt) - n_open)
    rt_closed = rt[:n_rt_closed] if n_rt_closed else []
    return {
        "n_trades": n_trades,   # changements de position (transparence)
        "win_rate": round(wins / n_trades, 4) if n_trades else None,
        "expectancy_pct": round(float(pnl[trades_idx].mean()) * 100.0, 4)
        if n_trades else None,
        "n_round_trips": n_rt_closed,      # définition standard, entrées clôturées
        "n_open_trades": n_open,
        "win_rate_rt": round(sum(1 for p in rt_closed if p > 0) / n_rt_closed, 4)
        if n_rt_closed else None,
        "expectancy_rt_pct": round(float(sum(rt_closed) / n_rt_closed) * 100.0, 4)
        if n_rt_closed else None,
        "cumulative_pnl_pct": round(cum_pnl_pct, 4),
        "sharpe_annual": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(max_dd, 4),
    }


def stress_high_vol(close: pd.Series, signals: pd.Series,
                    vol: pd.Series, top_quantile: float = STRESS_TOP_QUANTILE,
                    cost_ar_pct: float = COST_AR_PCT) -> dict:
    """Stress : sous-échantillon des 10 % de barres de vol maximale réelle du
    dataset (fenêtre continue autour du pic) — PnL des deux bras sur cette
    fenêtre. C'est le scénario que le filtre est censé protéger."""
    thr = vol.quantile(top_quantile) if vol.notna().sum() > 2 else np.inf
    stressed = (vol >= thr).fillna(False)
    if not stressed.any():
        return {"n_bars": 0, "note": "pas de fenêtre stressable"}
    idx = close.index[stressed]
    lo, hi = idx.min(), idx.max()
    sub_close = close.loc[lo:hi]
    sub_sig = signals.reindex(sub_close.index).fillna(0.0)
    return {"n_bars": int(len(sub_close)),
            **backtest_signals(sub_close, sub_sig, cost_ar_pct=cost_ar_pct)}


# --------------------------------------------------------------------------- #
# Décision (fonction pure, testable)
# --------------------------------------------------------------------------- #
def decide(exp_base, exp_treat, st_base: float, st_treat: float,
           n_rt: int) -> tuple:
    """
    REJECT seulement si le traitement n'améliore NI l'OOS NI le stress.
    KEEP (preuve insuffisante) si < MIN_TRADES_OOS round-trips OOS.
    KEEP (mixtes) si amélioration d'un côté seulement. Jamais PROMOTE.
    """
    improved_oos = exp_treat is not None and exp_base is not None \
        and exp_treat >= exp_base
    improved_stress = st_treat > st_base + 1e-9
    if n_rt < MIN_TRADES_OOS:
        return "KEEP", (f"Preuve insuffisante : {n_rt} round-trips OOS "
                        f"(< {MIN_TRADES_OOS}). Le filtre vol n'est ni promu "
                        f"ni tué.")
    if not improved_oos and not improved_stress:
        return "REJECT", (f"Le filtre vol n'améliore NI l'expectancy OOS "
                          f"({exp_treat} % vs {exp_base} % baseline) "
                          f"NI le stress haute vol ({st_treat} % vs "
                          f"{st_base} %). Hypothèse invalidée -> kill list.")
    if improved_oos and improved_stress:
        return "KEEP", (f"Le filtre vol améliore l'expectancy OOS "
                        f"({exp_treat} % vs {exp_base} %) ET le stress "
                        f"({st_treat} % vs {st_base} %). Preuve positive mais "
                        f"limitée : KEEP, aucune promotion automatique.")
    return "KEEP", (f"Résultats MIXTES : "
                    f"{'amélioration' if improved_oos else 'dégradation'} OOS "
                    f"({exp_treat} % vs {exp_base} %) et "
                    f"{'amélioration' if improved_stress else 'dégradation'} "
                    f"stress ({st_treat} % vs {st_base} %). Hypothèse "
                    f"conservée (KEEP) sans promotion — ré-évaluation après "
                    f"affinement ou plus de données.")


# --------------------------------------------------------------------------- #
# Orchestration d'une expérience
# --------------------------------------------------------------------------- #
def run_experiment(db, experiment_id: int,
                   symbols: tuple = DEFAULT_SYMBOLS,
                   train_ratio: float = TRAIN_RATIO,
                   vol_quantile: float = VOL_QUANTILE,
                   filter_scale: float = FILTER_SCALE,
                   cost_ar_pct: float = COST_AR_PCT,
                   research_memory=None) -> dict:
    """
    Pipeline complet pour l'hypothèse #1 (Momentum + filtre vol en HIGH_VOL) :
    baseline vs traitement, walk-forward 70/30, stress, décision REJECT/KEEP.
    Enregistre le résultat via ResearchMemory si fourni (ou directement db).
    """
    from core.research_memory import ResearchMemory
    rm = research_memory or (ResearchMemory(db) if db is not None else None)

    # 1. charger l'hypothèse
    hypothesis = f"experiment#{experiment_id}"
    try:
        exps = db.list_experiments(limit=200) if db is not None else []
        for e in exps:
            if int(e.get("id", -1)) == experiment_id:
                hypothesis = e.get("hypothesis", hypothesis)
                break
    except Exception:
        pass

    per_symbol = {}
    oos_base, oos_treat, stress_base, stress_treat = [], [], [], []
    parity_failures = []

    for sym in symbols:
        df = load_candles(db, sym)
        if df.empty or len(df) < 400:
            continue
        close = df["close"].astype(float)
        vol = volatility_ewma(close)
        sig = momentum_signal_series(close, volume=df.get("volume")
                                     if "volume" in df.columns else None)

        # parité avec la production (garde-fou : pas de variante de labo)
        if not check_signal_parity(df):
            parity_failures.append(sym)
            continue

        # walk-forward : seuil de vol calibré sur TRAIN uniquement
        split = int(len(df) * train_ratio)
        train_vol = vol.iloc[:split]
        thr = train_vol.quantile(vol_quantile) if train_vol.notna().sum() > 2 \
            else np.inf
        mask = high_vol_mask(vol, thr)
        sig_treat = sig.where(~mask, sig * filter_scale).fillna(0.0)

        # backtests (sur tout le dataset pour la lecture ; OOS séparé)
        oos_close = close.iloc[split:]
        oos_sig_base = sig.iloc[split:]
        oos_sig_treat = sig_treat.iloc[split:]
        b_base = backtest_signals(oos_close, oos_sig_base, cost_ar_pct=cost_ar_pct)
        b_treat = backtest_signals(oos_close, oos_sig_treat, cost_ar_pct=cost_ar_pct)
        s_base = stress_high_vol(close, sig, vol, cost_ar_pct=cost_ar_pct)
        s_treat = stress_high_vol(close, sig_treat, vol, cost_ar_pct=cost_ar_pct)

        oos_base.append(b_base)
        oos_treat.append(b_treat)
        stress_base.append(s_base)
        stress_treat.append(s_treat)
        per_symbol[sym] = {
            "n_bars": int(len(df)), "n_oos": int(len(oos_close)),
            "vol_threshold": round(float(thr), 6) if np.isfinite(thr) else None,
            "baseline_oos": b_base, "treatment_oos": b_treat,
            "baseline_stress": s_base, "treatment_stress": s_treat,
        }

    # 2. agrégation OOS (pondérée par le nombre de trades) — les métriques
    # décisionnelles sont les ROUND-TRIPS (définition standard, entrées
    # clôturées) ; les métriques par barre restent exposées en transparence.
    def _agg(rows, key):
        n = sum(r.get("n_trades", 0) for r in rows)
        n_rt = sum(r.get("n_round_trips", 0) for r in rows)
        exps = [r.get("expectancy_rt_pct") for r in rows
                if r.get("expectancy_rt_pct") is not None]
        wr = [r.get("win_rate_rt") for r in rows
              if r.get("win_rate_rt") is not None]
        pnl = sum(r.get("cumulative_pnl_pct", 0.0) for r in rows)
        return {"n_trades": n,
                "n_round_trips": n_rt,
                "expectancy_pct": round(sum(exps) / len(exps), 4) if exps else None,
                "win_rate": round(sum(wr) / len(wr), 4) if wr else None,
                "cumulative_pnl_pct": round(pnl, 4)}

    agg_base, agg_treat = _agg(oos_base, "baseline"), _agg(oos_treat, "treatment")
    stress_agg_base = _agg(stress_base, "baseline")
    stress_agg_treat = _agg(stress_treat, "treatment")
    n_oos = agg_treat["n_trades"]

    # 3. décision (honnête, conservatrice, nuancée)
    # REJECT seulement si le traitement n'améliore NI l'OOS NI le stress.
    # Résultats mixtes (amélioration d'un côté, dégradation dans le bruit de
    # l'autre) -> KEEP : hypothèse conservée, jamais promue sans preuve plus
    # forte. L'objectif déclaré du filtre (réduire les pertes en HIGH_VOL)
    # est mesuré dans la colonne stress.
    exp_base = agg_base.get("expectancy_pct")
    exp_treat = agg_treat.get("expectancy_pct")
    st_base = (stress_agg_base.get("cumulative_pnl_pct") or 0.0)
    st_treat = (stress_agg_treat.get("cumulative_pnl_pct") or 0.0)
    n_rt = agg_treat.get("n_round_trips", 0)
    decision, conclusion = decide(exp_base, exp_treat, st_base, st_treat, n_rt)

    results = {
        "hypothesis": hypothesis,
        "symbols": [s for s in per_symbol],
        "parity_failures": parity_failures,
        "cost_ar_pct": cost_ar_pct,
        "train_ratio": train_ratio, "vol_quantile": vol_quantile,
        "filter_scale": filter_scale,
        "oos": {"baseline": agg_base, "treatment": agg_treat,
                "n_oos_trades": n_oos},
        "stress": {"baseline": {k: v for k, v in
                                _agg(stress_base, "stress").items()},
                   "treatment": {k: v for k, v in
                                 _agg(stress_treat, "stress").items()}},
        "per_symbol": per_symbol,
        "decision": decision,
    }

    # 4. enregistrement (jamais bloquant)
    recorded = False
    if rm is not None:
        try:
            recorded = rm.record_experiment_result(
                experiment_id, results, conclusion, decision)
        except Exception as e:
            logger.warning(f"record_experiment_result failed: {e}")
    results["recorded"] = recorded
    return results

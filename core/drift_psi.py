"""
DRIFT DETECTION — PSI (Population Stability Index) sur les features clés
(LOT D / F4 — online learning).

Faiblesse F4 de l'audit : le bot a des bases d'apprentissage continu
(MLOps, meta-labeling, bandits) mais pas de détection de drift DISTRIBUTION
mature. Le CUSUM existant (models/mlops_pipeline.py) surveille l'erreur de
PRÉDICTION du modèle ; ce module surveille la DISTRIBUTION des features
d'entrée — les deux sont complémentaires :

  - CUSUM : « le modèle se trompe plus que d'habitude » (output drift)
  - PSI   : « le marché n'a plus la même distribution » (input drift)

PSI (Population Stability Index) — standard institutionnel (crédit risk,
quant finance) : on compare la distribution de référence (fenêtre longue)
à la distribution récente (fenêtre courte), feature par feature, par
bucketing en percentiles. Seuils d'usage courant :
    PSI < 0.10  -> STABLE (distribution inchangée)
    0.10-0.25   -> MODERATE (drift à surveiller)
    PSI > 0.25  -> SEVERE (drift majeur — l'edge des stratégies peut
                   avoir changé de nature)

Utilisation dans QUANT-PORTAL (LOT D) :
  - le PSI est calculé sur les features RÉELLES dérivées des candles
    (returns_1, |returns|, momentum_10, log(volume) — des séries quasi-iid :
    les features lissées par rolling y produisent des faux positifs, voir
    PSI_FEATURES) ;
  - en drift SEVERE, l'oubli du bandit Thompson est ACCÉLÉRÉ (decay plus
    fort : demi-vie plus courte) pour que l'allocateur stratégique cesse
    de récompenser un edge mort et explore à nouveau ;
  - le tout est exposé en télémétrie (`drift_psi`).

DÉMO == RÉAL : aucun paramètre de mode ici — même chemin de code partout.
"""
import logging
import time

import numpy as np
import pandas as pd

from core.config import settings

logger = logging.getLogger("InstitutionalTradingBot")  # même canal que main

# --------------------------------------------------------------------------- #
# CONFIG (LOT 3 : tout tunable passe par core/config.py / config.yaml)
# --------------------------------------------------------------------------- #
PSI_N_BINS: int = settings.get_int("drift", "psi_n_bins", 10)
# Seuils MARCHÉ (calibrés sur données réelles, LOT D) : les seuils du crédit
# (0.10/0.25) déclenchent SEVERE en permanence sur des rendements de marché
# (mesuré : 7/7 actifs SEVERE avec 0.25). PSI > 0.60 sur ~3 mois de
# référence = changement de régime majeur ; 0.30-0.60 = à surveiller.
PSI_STABLE_THRESHOLD: float = settings.get_float("drift", "psi_stable_threshold", 0.30)
PSI_SEVERE_THRESHOLD: float = settings.get_float("drift", "psi_severe_threshold", 0.60)
# Fenêtres (en barres) : référence ~3 mois (2000 h), récente ~17 jours
# (400 h) — alignées sur l'horizon de régime du bandit (demi-vie ~34 MAJ).
# Une référence trop courte (400) capte les régimes normaux comme du drift.
PSI_WINDOW_REFERENCE: int = settings.get_int("drift", "psi_window_reference", 2000)
PSI_WINDOW_RECENT: int = settings.get_int("drift", "psi_window_recent", 400)
# Decay du bandit : stable (0.98 = demi-vie ~34 MAJ) vs drift sévère
# (0.92 = demi-vie ~8 MAJ — l'edge est présumé mort, on oublie vite).
BANDIT_DECAY_STABLE: float = settings.get_float("drift", "bandit_decay_stable", 0.98)
BANDIT_DECAY_DRIFT: float = settings.get_float("drift", "bandit_decay_drift", 0.92)
# Bornes dures du decay appliqué au bandit (jamais de décroissance
# instantanée — un oubli total = un bandit vierge qui repart à l'exploration
# pure, perte de l'apprentissage déjà acquis).
BANDIT_DECAY_MIN: float = settings.get_float("drift", "bandit_decay_min", 0.85)
BANDIT_DECAY_MAX: float = settings.get_float("drift", "bandit_decay_max", 0.995)
# Intervalle de recalcul dans la boucle live
PSI_INTERVAL_SECONDS: float = settings.get_float("drift", "psi_interval_seconds", 900.0)

# Features surveillées. Leçons de calibrage (LOT D, MESURÉES sur données
# réelles — pas de la théorie) :
#   1. Les features LISSÉES par rolling (volatility_20, volume_zscore) sont
#      autocorrélées : PSI > 10 sur marché homogène (faux positifs).
#   2. momentum_10 (tendance 10 barres) est un détecteur de TENDANCE, pas de
#      distribution : un marché normal en tendance produit des PSI 2-5x plus
#      élevés que les rendements (mesuré : SOL 2.89, EURUSD 6.6 sur fenêtres
#      courtes) -> RETIRÉ.
#   3. Les seuils du crédit (0.10/0.25) sont trop stricts pour des données de
#      marché : avec des fenêtres courtes (17j vs 6j), TOUT le monde est
#      SEVERE en permanence (mesuré : 7/7 actifs) -> fenêtres alignées sur
#      l'horizon de régime (référence ~3 mois) et seuils marché 0.30/0.60.
# On surveille donc des features quasi-iid robustes :
#   - returns_1       : rendements bruts (drift de moyenne ET de vol)
#   - returns_abs     : |rendements| (détecteur de vol direct)
#   - volume_log      : log(volume brut) (participation)
PSI_FEATURES: list[str] = ["returns_1", "returns_abs", "volume_log"]

# Features qui ALIMENTENT le statut/decay (le signal). Le volume log est
# calculé et exposé (information) mais n'est PAS déclencheur : les données
# de volume Yahoo sont inconstantes (fallback 10.0, trous) et un changement
# de volume peut être structurel (participation) sans affecter l'edge des
# stratégies OHLC. Le drift qui compte pour l'edge = celui des RENDEMENTS
# (moyenne et vol) — mesuré par returns_1 et returns_abs.
PSI_SIGNAL_FEATURES: list[str] = ["returns_1", "returns_abs"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_psi(reference: np.ndarray, recent: np.ndarray,
                n_bins: int = PSI_N_BINS) -> float:
    """
    Population Stability Index entre deux distributions 1D.

    PSI = Σ (A_i - B_i) × ln(A_i / B_i), avec A = proportions de la
    distribution RÉCENTE et B = proportions de la RÉFÉRENCE, sur des bins
    définis par les percentiles de la RÉFÉRENCE (méthode standard).

    NB (leçon de calibrage LOT D) : ce PSI est un test de RANG — invariant
    par toute transformation monotone (log, affine, exp). Il ne doit être
    appliqué qu'à des séries quasi-iid : sur des features lissées par
    rolling (vol 20 barres, z-score de volume), la fenêtre récente corrélée
    tombe en bloc dans les bins extrêmes et produit des FAUX POSITIFS
    massifs (PSI > 10 sur marché homogène, mesuré). Les features surveillées
    sont donc des séries brutes quasi-iid (returns, |returns|, momentum, log
    volume) — voir PSI_FEATURES.

    Cas dégénérés gérés :
      - données insuffisantes (< 5 points) -> 0.0 (pas de drift mesurable)
      - bin vide dans la référence -> epsilon (pas de division par zéro)
      - variance de référence nulle -> 0.0 (série constante = pas de
        changement mesurable fiablement)
    """
    ref = np.asarray(reference, dtype=float)
    rec = np.asarray(recent, dtype=float)
    if len(ref) < 5 or len(rec) < 5:
        return 0.0
    ref = ref[np.isfinite(ref)]
    rec = rec[np.isfinite(rec)]
    if len(ref) < 5 or len(rec) < 5:
        return 0.0
    if float(np.std(ref)) < 1e-12:
        # série de référence constante : le moindre écart récent est un drift,
        # mais un PSI sur variance nulle est instable -> 0 (aucun signal fiable)
        return 0.0

    n_bins = max(2, int(n_bins))
    # bornes des bins = percentiles de la RÉFÉRENCE
    edges = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    def _proportions(x: np.ndarray) -> np.ndarray:
        counts = np.zeros(n_bins)
        idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
        for i in idx:
            counts[i] += 1.0
        return counts / max(len(x), 1)

    p_ref = _proportions(ref)
    p_rec = _proportions(rec)
    psi = 0.0
    for a, b in zip(p_rec, p_ref):
        if b <= 1e-12:
            b = 1e-12
        if a <= 1e-12:
            a = 1e-12
        psi += (a - b) * np.log(a / b)
    return float(max(0.0, psi))


def psi_status(psi: float) -> str:
    """STABLE (< 0.10) / MODERATE (0.10-0.25) / SEVERE (> 0.25)."""
    if psi >= PSI_SEVERE_THRESHOLD:
        return "SEVERE"
    if psi >= PSI_STABLE_THRESHOLD:
        return "MODERATE"
    return "STABLE"


def bandit_decay_for_psi(psi: float) -> float:
    """
    Decay du bandit recommandé pour un PSI donné, BORNÉ :
      - PSI <= stable  -> BANDIT_DECAY_STABLE (comportement nominal)
      - PSI >= sévère  -> BANDIT_DECAY_DRIFT (oubli accéléré)
      - entre les deux -> interpolation linéaire
    Bornes dures [BANDIT_DECAY_MIN, BANDIT_DECAY_MAX] (jamais d'oubli
    instantané ni de mémoire infinie).
    """
    if psi <= PSI_STABLE_THRESHOLD:
        return float(_clamp(BANDIT_DECAY_STABLE, BANDIT_DECAY_MIN, BANDIT_DECAY_MAX))
    if psi >= PSI_SEVERE_THRESHOLD:
        return float(_clamp(BANDIT_DECAY_DRIFT, BANDIT_DECAY_MIN, BANDIT_DECAY_MAX))
    span = max(PSI_SEVERE_THRESHOLD - PSI_STABLE_THRESHOLD, 1e-9)
    t = (psi - PSI_STABLE_THRESHOLD) / span
    decay = BANDIT_DECAY_STABLE + t * (BANDIT_DECAY_DRIFT - BANDIT_DECAY_STABLE)
    return float(_clamp(decay, BANDIT_DECAY_MIN, BANDIT_DECAY_MAX))


def extract_psi_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Séries de features RÉELLES depuis des candles OHLCV (cohérentes avec le
    feature store LOT 48). Retourne {feature: np.ndarray} ; une feature
    incalculable est omise (jamais de valeur inventée).
    """
    out: dict[str, np.ndarray] = {}
    try:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        returns_1 = close.pct_change(1).replace([np.inf, -np.inf], np.nan).dropna().values
        if len(returns_1) >= 10:
            out["returns_1"] = returns_1
            out["returns_abs"] = np.abs(returns_1)

        # volume en log (normalise la distribution asymétrique ; le niveau
        # absolu détecte les changements de participation). NB : momentum_10
        # a été RETIRÉ (leçon de calibrage LOT D) — c'est un détecteur de
        # TENDANCE, pas de distribution : un marché normal en tendance
        # produit des PSI 2-5x plus élevés que les rendements.
        vol_pos = volume[volume > 0.0]
        if len(vol_pos) >= 10:
            out["volume_log"] = np.log(vol_pos).values
    except Exception as e:  # jamais bloquant
        logger.warning(f"extract_psi_features failed: {e}")
    return out


class DriftMonitor:
    """
    Surveille le PSI des features clés entre une fenêtre de RÉFÉRENCE et une
    fenêtre RÉCENTE des candles réelles. Expose :
      - psi par feature + max_psi + statut global
      - bandit_decay recommandé (appliqué au bandit Thompson par main.py)
      - n_observations, dernière mise à jour (transparence)
    """

    def __init__(self, reference_window: int = PSI_WINDOW_REFERENCE,
                 recent_window: int = PSI_WINDOW_RECENT):
        self.reference_window = max(30, int(reference_window))
        self.recent_window = max(10, int(recent_window))
        self.last_psi: dict[str, float] = {}
        self.last_max_psi: float = 0.0
        self.last_status: str = "STABLE"
        self.last_bandit_decay: float = float(BANDIT_DECAY_STABLE)
        self.n_updates: int = 0
        self.last_update_ts: float = 0.0
        self.last_n_bars: int = 0

    def update(self, df: pd.DataFrame, now: float | None = None) -> dict:
        """
        Recalcule le PSI sur les candles fournies. Le DataFrame doit contenir
        au moins reference_window + recent_window + 25 barres pour un calcul
        significatif (sinon -> statut STABLE, aucun signal fabriqué).
        Retourne le dict d'état (identique à to_dict()).
        """
        ts = now if now is not None else time.time()
        if df is None or df.empty:
            return self.to_dict()

        features = extract_psi_features(df)
        if not features:
            return self.to_dict()

        total_needed = self.reference_window + self.recent_window
        if len(df) < total_needed:
            # échantillon insuffisant : pas de signal fiable (honnêteté)
            self.last_n_bars = len(df)
            return self.to_dict()

        psi_map: dict[str, float] = {}
        for name, series in features.items():
            if len(series) < total_needed + 5:
                continue
            ref = series[-self.reference_window - self.recent_window:-self.recent_window]
            rec = series[-self.recent_window:]
            if len(ref) >= 20 and len(rec) >= 10:
                psi_map[name] = round(compute_psi(ref, rec), 4)

        if not psi_map:
            return self.to_dict()

        self.last_psi = psi_map
        # LOT D (calibré sur données réelles) : le statut/decay ne porte que
        # sur les features de RENDEMENTS (PSI_SIGNAL_FEATURES) — le volume
        # log est exposé mais ne déclenche pas (données inconstantes).
        signal_psis = [v for k, v in psi_map.items() if k in PSI_SIGNAL_FEATURES]
        self.last_max_psi = max(signal_psis) if signal_psis else 0.0
        self.last_status = psi_status(self.last_max_psi)
        self.last_bandit_decay = round(bandit_decay_for_psi(self.last_max_psi), 4)
        self.n_updates += 1
        self.last_update_ts = ts
        self.last_n_bars = len(df)
        return self.to_dict()

    def to_dict(self) -> dict:
        return {
            "psi_per_feature": {k: round(v, 4) for k, v in self.last_psi.items()},
            "max_psi": round(self.last_max_psi, 4),
            "status": self.last_status,
            "bandit_decay_recommended": round(self.last_bandit_decay, 4),
            "bandit_decay_bounds": [round(BANDIT_DECAY_MIN, 4),
                                    round(BANDIT_DECAY_MAX, 4)],
            "thresholds": {"stable": PSI_STABLE_THRESHOLD,
                           "severe": PSI_SEVERE_THRESHOLD},
            "windows": {"reference": self.reference_window,
                        "recent": self.recent_window},
            "n_updates": self.n_updates,
            "last_update_ts": round(self.last_update_ts, 2),
            "n_bars": self.last_n_bars,
            "note": "PSI > 0.25 = drift sévère : l'oubli du bandit est accéléré.",
        }


class MultiAssetDriftMonitor:
    """
    LOT D (F4, corrigé) : surveillance du drift PSI sur PLUSIEURS actifs.

    Chaque actif a son propre DriftMonitor (fenêtres de référence/récente sur
    SES candles réelles). L'état agrégé expose :
      - per_asset : {symbol: état individuel} (psi par feature, statut...)
      - max_psi    : le maximum sur les actifs calculés (le pire cas pilote
                     l'oubli du bandit — conservateur : si UN marché a
                     drastiquement changé de distribution, on oublie plus
                     vite, même si les autres sont stables)
      - status / bandit_decay_recommended : dérivés du max_psi global
    """

    def __init__(self, symbols: list[str] | None = None,
                 reference_window: int = PSI_WINDOW_REFERENCE,
                 recent_window: int = PSI_WINDOW_RECENT):
        self.symbols = list(symbols or [])
        self.monitors: dict[str, DriftMonitor] = {
            s: DriftMonitor(reference_window=reference_window,
                            recent_window=recent_window)
            for s in self.symbols}
        self.last_state: dict = {}

    def update_all(self, candles_by_symbol: dict[str, pd.DataFrame],
                   now: float | None = None) -> dict:
        """Met à jour chaque actif fourni ; agrège l'état (max_psi global)."""
        per_asset: dict[str, dict] = {}
        max_psi = 0.0
        for sym, df in candles_by_symbol.items():
            mon = self.monitors.setdefault(
                sym, DriftMonitor())
            st = mon.update(df, now=now)
            per_asset[sym] = st
            if st["n_updates"] > 0:
                max_psi = max(max_psi, st["max_psi"])
        if not per_asset:
            return self.last_state
        self.last_state = {
            "per_asset": per_asset,
            "max_psi": round(max_psi, 4),
            "status": psi_status(max_psi),
            "bandit_decay_recommended": round(bandit_decay_for_psi(max_psi), 4),
            "bandit_decay_bounds": [round(BANDIT_DECAY_MIN, 4),
                                    round(BANDIT_DECAY_MAX, 4)],
            "thresholds": {"stable": PSI_STABLE_THRESHOLD,
                           "severe": PSI_SEVERE_THRESHOLD},
            "n_updates": sum(1 for s in per_asset.values() if s["n_updates"] > 0),
            "last_update_ts": round(time.time() if now is None else now, 2),
            "note": "PSI multi-actifs : max_psi = pire cas sur les actifs calculés.",
        }
        return dict(self.last_state)

    def to_dict(self) -> dict:
        return dict(self.last_state)


# Durée pendant laquelle un drift CUSUM (erreur de prédiction) reste actif
# dans l'état fusionné avant de retomber à False (le retraining automatique
# a eu le temps de se faire — le flag est un signal TEMPORAIRE, pas une
# étiquette permanente).
DRIFT_CUSUM_HOLD_SECONDS: float = settings.get_float(
    "drift", "cusum_hold_seconds", 3600.0)


def unified_drift_state(psi_state: dict, cusum_state: dict | None) -> dict:
    """
    FUSION CUSUM + PSI (LOT D, corrigé) : un seul état de drift pour le bot.

      - CUSUM  (models/mlops_pipeline.py) : erreur de PRÉDICTION du modèle
        anormalement élevée (output drift) — déclenche le retraining.
      - PSI    (ce module)                : distribution des features changée
        (input drift) — accélère l'oubli du bandit.

    Règle de fusion (conservatrice, jamais de faux "stable") :
      - status = SEVERE si PSI SEVERE OU CUSUM détecté (un seul suffit :
        le système perd son edge dès qu'UNE des deux faces drift)
      - status = MODERATE si PSI MODERATE (CUSUM est binaire)
      - sinon STABLE
      - decay recommandé = le PLUS AGRESSIF (min) des deux recommandations,
        borné [BANDIT_DECAY_MIN, BANDIT_DECAY_MAX].

    cusum_state attendu : {"detected": bool, "ts": float} ou None.
    """
    psi_status_ = psi_state.get("status", "STABLE") if psi_state else "STABLE"
    cusum_detected = bool((cusum_state or {}).get("detected", False))

    if psi_status_ == "SEVERE" or cusum_detected:
        status = "SEVERE"
    elif psi_status_ == "MODERATE":
        status = "MODERATE"
    else:
        status = "STABLE"

    psi_decay = float(psi_state.get("bandit_decay_recommended",
                                    BANDIT_DECAY_STABLE)) if psi_state else BANDIT_DECAY_STABLE
    cusum_decay = float(BANDIT_DECAY_DRIFT) if cusum_detected else BANDIT_DECAY_STABLE
    decay = min(psi_decay, cusum_decay)
    decay = float(_clamp(decay, BANDIT_DECAY_MIN, BANDIT_DECAY_MAX))

    return {
        "status": status,
        "max_psi": float(psi_state.get("max_psi", 0.0)) if psi_state else 0.0,
        "bandit_decay_recommended": round(decay, 4),
        "sources": {
            "psi": psi_status_,
            "cusum": "DETECTED" if cusum_detected else "OK",
        },
        "note": "Fusion CUSUM (erreur de prédiction) + PSI (distribution des features) — SEVERE si l'un des deux est sévère.",
    }


def run_drift_check(state: dict, db, drift_monitor: MultiAssetDriftMonitor,
                    meta_engine, audit_ip, logger=None,
                    symbols: list[str] | None = None,
                    load_limit: int = 2500) -> dict:
    """
    Tick périodique du drift PSI MULTI-ACTIFS + fusion CUSUM (appelé par la
    boucle live de main.py) :
      1. toutes les PSI_INTERVAL_SECONDS (sinon renvoie l'état courant) ;
      2. charge les candles RÉELLES du cache DB pour CHAQUE actif surveillé ;
      3. met à jour le MultiAssetDriftMonitor (pire cas = max_psi global) ;
      4. fusionne avec l'état CUSUM (state["drift_cusum"]) ;
      5. applique le decay recommandé au bandit (jamais < BANDIT_DECAY_MIN) ;
      6. audit log si drift SÉVÈRE (source PSI ou CUSUM).

    Jamais bloquant : toute erreur renvoie l'état courant inchangé (le bot
    continue avec le comportement nominal).
    """
    log = logger if logger is not None else logging.getLogger("InstitutionalTradingBot")
    try:
        now = time.time()
        if now - float(state.get("drift_psi_last_ts", 0.0)) < PSI_INTERVAL_SECONDS:
            return dict(state.get("drift_psi", {}))
        state["drift_psi_last_ts"] = now

        # CUSUM : le flag reste actif DRIFT_CUSUM_HOLD_SECONDS (le retraining
        # automatique a eu le temps de se faire) puis retombe à False.
        cusum_state = state.get("drift_cusum") or {}
        if cusum_state.get("detected") and \
                now - float(cusum_state.get("ts", 0.0)) > DRIFT_CUSUM_HOLD_SECONDS:
            cusum_state = {"detected": False}
            state["drift_cusum"] = cusum_state

        # PSI multi-actifs : candles RÉELLES pour chaque actif surveillé.
        syms = symbols if symbols else list(getattr(drift_monitor, "symbols", []) or ["BTCUSDT"])
        candles_by_symbol: dict = {}
        for sym in syms:
            try:
                df_s = db.load_candles(sym, limit=load_limit)
                if df_s is not None and len(df_s) >= 50:
                    candles_by_symbol[sym] = df_s
            except Exception:
                continue
        if not candles_by_symbol:
            return dict(state.get("drift_psi", {}))

        dd = drift_monitor.update_all(candles_by_symbol, now=now)
        if not dd:
            return dict(state.get("drift_psi", {}))

        # Fusion CUSUM + PSI -> état unifié + decay final.
        unified = unified_drift_state(dd, cusum_state)
        dd["unified"] = unified
        dd["cusum"] = {
            "detected": bool(cusum_state.get("detected", False)),
            "ts": cusum_state.get("ts", 0.0),
            "hold_seconds": DRIFT_CUSUM_HOLD_SECONDS,
        }
        state["drift_psi"] = dd

        decay = float(unified["bandit_decay_recommended"])
        if abs(decay - float(getattr(meta_engine, "bandit_decay", decay))) > 1e-6:
            meta_engine.set_bandit_decay(decay)
        state["drift_psi"]["bandit_decay_applied"] = float(
            getattr(meta_engine, "bandit_decay", decay))

        if unified["status"] != "STABLE":
            log.info(
                f"📈 DRIFT {unified['status']} : PSI max={dd['max_psi']:.3f} "
                f"({dd['unified']['sources']}) -> oubli bandit "
                f"{state['drift_psi']['bandit_decay_applied']:.4f} (nominal 0.98).")
            if unified["status"] == "SEVERE":
                db.add_audit_log(
                    "DRIFT_PSI_SEVERE", audit_ip(),
                    f"Drift sévère (PSI {dd['max_psi']:.3f} / CUSUM "
                    f"{'détecté' if cusum_state.get('detected') else 'OK'}) — oubli bandit à "
                    f"{state['drift_psi']['bandit_decay_applied']:.4f}.")
        return dict(dd)
    except Exception as e:
        log.warning(f"Drift PSI check failed ({e}) — comportement nominal conservé.")
        return dict(state.get("drift_psi", {}))

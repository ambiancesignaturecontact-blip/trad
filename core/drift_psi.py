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
PSI_STABLE_THRESHOLD: float = settings.get_float("drift", "psi_stable_threshold", 0.10)
PSI_SEVERE_THRESHOLD: float = settings.get_float("drift", "psi_severe_threshold", 0.25)
# Fenêtres (en barres) : référence = historique long, récente = fin de série
PSI_WINDOW_REFERENCE: int = settings.get_int("drift", "psi_window_reference", 300)
PSI_WINDOW_RECENT: int = settings.get_int("drift", "psi_window_recent", 100)
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

# Features surveillées. Leçon de calibrage (LOT D, mesurée sur données) :
# les features LISSÉES par rolling (volatility_20, volume_zscore) sont
# FORTEMENT autocorrélées — le PSI par percentiles y explose (> 10) même
# sur un marché homogène, car la fenêtre récente corrélée tombe en bloc
# dans les bins extrêmes de la référence. On surveille donc des features
# quasi-iid, qui discriminent correctement :
#   - returns_1       : rendements bruts (drift de moyenne ET de vol)
#   - returns_abs     : |rendements| (détecteur de vol direct, iid)
#   - momentum_10     : tendance 10 barres
#   - volume_log      : log(volume brut) (participation, normalisée)
PSI_FEATURES: list[str] = ["returns_1", "returns_abs", "momentum_10", "volume_log"]


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

        momentum_10 = (close / close.shift(10) - 1.0).replace(
            [np.inf, -np.inf], np.nan).dropna().values
        if len(momentum_10) >= 10:
            out["momentum_10"] = momentum_10

        # volume en log (normalise la distribution asymétrique ; le niveau
        # absolu détecte les changements de participation)
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
        self.last_max_psi = max(psi_map.values())
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


def run_drift_check(state: dict, db, drift_monitor: DriftMonitor, meta_engine,
                    audit_ip, logger=None) -> dict:
    """
    Tick périodique du drift PSI (appelé par la boucle live de main.py) :
      1. toutes les PSI_INTERVAL_SECONDS (sinon renvoie l'état courant) ;
      2. charge les candles RÉELLES du cache DB (BTCUSDT, référence) ;
      3. met à jour le DriftMonitor et expose le résultat dans state ;
      4. en drift non-STABLE, applique le decay recommandé au bandit
         (oubli accéléré, borné [BANDIT_DECAY_MIN, BANDIT_DECAY_MAX]) ;
      5. audit log si drift SÉVÈRE.

    Jamais bloquant : toute erreur renvoie l'état courant inchangé (le bot
    continue avec le comportement nominal).
    """
    log = logger if logger is not None else logging.getLogger("InstitutionalTradingBot")
    try:
        now = time.time()
        if now - float(state.get("drift_psi_last_ts", 0.0)) < PSI_INTERVAL_SECONDS:
            return dict(state.get("drift_psi", {}))
        state["drift_psi_last_ts"] = now

        psi_df = db.load_candles("BTCUSDT", limit=700)
        if psi_df is None or len(psi_df) < 50:
            return dict(state.get("drift_psi", {}))

        drift_monitor.update(psi_df, now=now)
        dd = drift_monitor.to_dict()
        state["drift_psi"] = dd

        decay = float(dd["bandit_decay_recommended"])
        if abs(decay - float(getattr(meta_engine, "bandit_decay", decay))) > 1e-6:
            meta_engine.set_bandit_decay(decay)
        state["drift_psi"]["bandit_decay_applied"] = float(
            getattr(meta_engine, "bandit_decay", decay))

        if dd["status"] != "STABLE":
            log.info(
                f"📈 DRIFT PSI {dd['status']} : max_psi={dd['max_psi']:.3f} "
                f"({dd['psi_per_feature']}) -> oubli bandit "
                f"{state['drift_psi']['bandit_decay_applied']:.4f} (nominal 0.98).")
            if dd["status"] == "SEVERE":
                db.add_audit_log(
                    "DRIFT_PSI_SEVERE", audit_ip(),
                    f"PSI sévère {dd['max_psi']:.3f} — oubli bandit accéléré à "
                    f"{state['drift_psi']['bandit_decay_applied']:.4f} "
                    f"(features {list(dd['psi_per_feature'])}).")
        return dict(dd)
    except Exception as e:
        log.warning(f"Drift PSI check failed ({e}) — comportement nominal conservé.")
        return dict(state.get("drift_psi", {}))

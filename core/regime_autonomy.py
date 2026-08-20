"""
AUTONOMIE STRATÉGIQUE (LOT B / F2) — auto-adaptation BORNÉE des paramètres
de risque au régime HMM.

Faiblesse F2 de l'audit : le bot est autonome sur l'EXÉCUTION, mais les
paramètres de risque (Kelly, plafond par actif, drawdowns) restent fixes
quel que soit le régime de marché. Ce module rend le RISK-TAKING adaptatif
au régime, avec des bornes dures et des garde-fous anti-oscillation :

  1. BORNES DURES  : facteur ∈ [FACTOR_MIN, FACTOR_MAX] — jamais plus de
     1.25x ni moins de 0.60x la config de base (l'opérateur garde la main
     sur le risque maximal, même en régime favorable).
  2. CONFIDENCE    : le facteur n'est appliqué qu'à hauteur de la confiance
     du HMM (régime incertain -> facteur ramené vers 1.0, « je ne sais pas »
     — mentalité n°5). Un régime détecté sans confiance ne change rien.
  3. LISSAGE EMA   : pas de saut brutal lors d'un changement de régime
     (alpha configurable) — l'état HMM est un signal bruité, on ne remodèle
     pas tout le portefeuille sur un seul tick.
  4. DRAWDOWNS     : JAMAIS élargis (×min(facteur, 1.0)) — en régime
     défensif on resserre les circuit breakers, on ne les assouplit jamais.
     « Survivre d'abord » : la borne de perte maximale est un plancher dur.

Régimes HMM (models/regime_detector.py) :
    0 = Bull (low vol, rendements positifs)   -> léger boost (1.10)
    1 = Bear (high vol, rendements négatifs)  -> défensif (0.75)
    2 = Range (vol très faible)               -> neutre (1.00)
    3 = Erratic High Vol                      -> défensif (0.85)

DÉMO == RÉAL : aucun paramètre de mode ici — le même chemin de code
s'applique en DEMO et en REAL (fidélité exigée par l'opérateur).
"""
import logging

from core.config import settings

logger = logging.getLogger("RegimeAutonomy")

# --------------------------------------------------------------------------- #
# CONFIG (LOT 3 : tout tunable passe par core/config.py / config.yaml)
# --------------------------------------------------------------------------- #
AUTONOMY_ENABLED: bool = settings.get_bool("risk", "autonomy_enabled", True)

# Facteur cible par régime (borné [FACTOR_MIN, FACTOR_MAX] ci-dessous).
_REGIME_AGGRESSIVENESS_CFG = settings.get(
    "risk", "autonomy_aggressiveness", {0: 1.10, 1: 0.75, 2: 1.00, 3: 0.85})

FACTOR_MIN: float = settings.get_float("risk", "autonomy_factor_min", 0.60)
FACTOR_MAX: float = settings.get_float("risk", "autonomy_factor_max", 1.25)
EMA_ALPHA: float = settings.get_float("risk", "autonomy_ema_alpha", 0.30)
# En dessous de CONFIDENCE_MIN -> facteur neutre (1.0) ; au-dessus de
# CONFIDENCE_FULL -> facteur pleinement appliqué. Interpolation linéaire.
CONFIDENCE_MIN: float = settings.get_float("risk", "autonomy_confidence_min", 0.30)
CONFIDENCE_FULL: float = settings.get_float("risk", "autonomy_confidence_full", 0.80)

# Bornes ABSOLUES des paramètres effectifs (appliquées par RiskManager).
KELLY_MIN: float = settings.get_float("risk", "autonomy_kelly_min", 0.05)
KELLY_MAX: float = settings.get_float("risk", "autonomy_kelly_max", 0.25)
MAX_PER_ASSET_MIN: float = settings.get_float("risk", "autonomy_max_per_asset_min", 0.10)
MAX_PER_ASSET_MAX: float = settings.get_float("risk", "autonomy_max_per_asset_max", 0.30)
DAILY_DD_FLOOR: float = settings.get_float("risk", "autonomy_daily_drawdown_floor", 0.015)
TOTAL_DD_FLOOR: float = settings.get_float("risk", "autonomy_total_drawdown_floor", 0.05)

# Valeurs par défaut des paramètres de base (utilisées pour les bornes ;
# les valeurs réelles viennent des paramètres du RiskManager / config.yaml).
_BASE_KELLY: float = settings.get_float("risk", "kelly_multiplier_default", 0.15)
_BASE_MAX_PER_ASSET: float = settings.get_float("risk", "max_per_asset_pct", 0.25)
_BASE_DAILY_DD: float = settings.get_float("risk", "daily_drawdown_normal", 0.025)
_BASE_TOTAL_DD: float = settings.get_float("risk", "max_total_drawdown_normal", 0.08)


def _coerce_regime_map(raw) -> dict[int, float]:
    """Valide la table des facteurs par régime (jamais de valeur hors bornes)."""
    out: dict[int, float] = {}
    if not isinstance(raw, dict):
        return {0: 1.10, 1: 0.75, 2: 1.00, 3: 0.85}
    for k, v in raw.items():
        try:
            out[int(k)] = min(FACTOR_MAX, max(FACTOR_MIN, float(v)))
        except (TypeError, ValueError):
            continue
    return out


REGIME_AGGRESSIVENESS: dict[int, float] = _coerce_regime_map(_REGIME_AGGRESSIVENESS_CFG)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def regime_aggressiveness(regime_id: int | None, confidence: float | None = 1.0) -> float:
    """
    Facteur d'agressivité CIBLE pour un régime HMM, borné [FACTOR_MIN, FACTOR_MAX].

    - Régime inconnu (None) -> 1.0 (neutre : on ne punit pas une information
      absente, on n'invente pas de régime).
    - Le facteur est tiré vers 1.0 quand la confiance du HMM est faible
      (régime incertain -> comportement de base, pas de pari sur le régime).
    """
    mapped = REGIME_AGGRESSIVENESS.get(int(regime_id), 1.0) if regime_id is not None else 1.0
    try:
        conf = _clamp(float(confidence) if confidence is not None else 1.0, 0.0, 1.0)
    except (TypeError, ValueError):
        conf = 0.5
    if CONFIDENCE_FULL > CONFIDENCE_MIN:
        pull = _clamp((conf - CONFIDENCE_MIN) / (CONFIDENCE_FULL - CONFIDENCE_MIN), 0.0, 1.0)
    else:
        pull = 1.0 if conf >= CONFIDENCE_FULL else 0.0
    factor = 1.0 + (mapped - 1.0) * pull
    return float(_clamp(factor, FACTOR_MIN, FACTOR_MAX))


class RegimeAutonomy:
    """
    Facteur d'agressivité de risque LISSÉ (EMA), piloté par le régime HMM.

    Usage live (main.py) : à chaque tick où le régime est mis à jour,
        autonomy.update(regime_id, confidence)
        risk_manager.apply_regime_factor(autonomy.factor)
    puis exposition télémétrique via to_dict().
    """

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = _clamp(float(alpha), 0.05, 0.95)
        self.ema = 1.0
        self.last_seen: tuple | None = None
        self.last_regime_id: int | None = None
        self.last_confidence: float | None = None
        self.last_raw = 1.0
        self._n_updates = 0

    @property
    def factor(self) -> float:
        """Facteur EFFECTIF appliqué (lissé EMA, borné)."""
        return self.ema

    @property
    def enabled(self) -> bool:
        return bool(AUTONOMY_ENABLED)

    def update(self, regime_id: int | None, confidence: float | None = 1.0) -> float:
        """
        Met à jour le facteur lissé. Dédoublonne (régime, confiance arrondie) :
        un même état de marché ne déplace l'EMA qu'une fois par tick (le bloc
        de détection tourne par symbole ; on évite N mises à jour identiques).
        Retourne le facteur effectif.
        """
        try:
            conf = float(confidence) if confidence is not None else 1.0
        except (TypeError, ValueError):
            conf = 0.5
        key = (int(regime_id) if regime_id is not None else None, round(conf, 2))
        if key == self.last_seen:
            return self.ema
        self.last_seen = key

        if not self.enabled:
            self.ema = 1.0
            self.last_raw = 1.0
            return self.ema

        raw = regime_aggressiveness(regime_id, conf)
        self.last_raw = raw
        self.ema = _clamp(self.alpha * raw + (1.0 - self.alpha) * self.ema,
                          FACTOR_MIN, FACTOR_MAX)
        self._n_updates += 1

        if regime_id != self.last_regime_id:
            logger.info(
                f"RÉGIME {self.last_regime_id} -> {regime_id} : facteur "
                f"d'agressivité cible {raw:.2f}x (lissé {self.ema:.2f}x, "
                f"confiance {conf:.2f}) — bornes [{FACTOR_MIN:.2f}, {FACTOR_MAX:.2f}].")
        self.last_regime_id = regime_id
        self.last_confidence = conf
        return self.ema

    def reset(self) -> None:
        """Réinitialise le facteur à 1.0 (neutre) — tests / reprise propre."""
        self.ema = 1.0
        self.last_seen = None
        self.last_regime_id = None
        self.last_confidence = None
        self.last_raw = 1.0
        self._n_updates = 0

    def effective_params(self,
                         base_kelly: float | None = None,
                         base_max_per_asset: float | None = None,
                         base_daily_dd: float | None = None,
                         base_total_dd: float | None = None) -> dict:
        """
        Paramètres de risque EFFECTIFS (bornés) à partir des bases données
        (ou des défauts config). Les drawdowns ne sont JAMAIS élargis :
        ×min(facteur, 1.0) avec plancher dur. Le RiskManager applique les
        mêmes bornes sur SES paramètres (source unique : ce module).
        """
        f = self.factor
        kelly = _clamp((base_kelly if base_kelly is not None else _BASE_KELLY) * f,
                       KELLY_MIN, KELLY_MAX)
        mpa = _clamp((base_max_per_asset if base_max_per_asset is not None else _BASE_MAX_PER_ASSET) * f,
                     MAX_PER_ASSET_MIN, MAX_PER_ASSET_MAX)
        daily = _clamp((base_daily_dd if base_daily_dd is not None else _BASE_DAILY_DD) * min(f, 1.0),
                       DAILY_DD_FLOOR,
                       (base_daily_dd if base_daily_dd is not None else _BASE_DAILY_DD))
        total = _clamp((base_total_dd if base_total_dd is not None else _BASE_TOTAL_DD) * min(f, 1.0),
                       TOTAL_DD_FLOOR,
                       (base_total_dd if base_total_dd is not None else _BASE_TOTAL_DD))
        return {
            "fractional_kelly_multiplier": round(kelly, 4),
            "max_exposure_per_asset_pct": round(mpa, 4),
            "max_daily_drawdown_pct": round(daily, 4),
            "max_total_drawdown_pct": round(total, 4),
        }

    def to_dict(self) -> dict:
        """Exposition télémétrique honnête (valeurs réellement appliquées)."""
        return {
            "enabled": self.enabled,
            "regime_id": self.last_regime_id,
            "confidence": self.last_confidence,
            "factor": round(self.ema, 4),
            "factor_raw": round(self.last_raw, 4),
            "factor_bounds": [round(FACTOR_MIN, 4), round(FACTOR_MAX, 4)],
            "regime_aggressiveness": {str(k): round(v, 4)
                                      for k, v in sorted(REGIME_AGGRESSIVENESS.items())},
            "n_updates": self._n_updates,
            "effective": self.effective_params(),
        }

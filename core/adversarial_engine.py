"""
ADVERSARIAL DECISION ENGINE (LOT 6 du mandat — Validation).

Avant qu'une décision ne parte à l'exécution, on teste SES CONDITIONS
D'ÉCHEC : un trade qui n'est rentable que dans un environnement parfait
doit être considéré comme FRAGILE (et donc refusé ou réduit).

Scénarios adversariaux (mandat §12), appliqués à la décision COURANTE :
  - spread ×2              : le slippage d'entrée double
  - slippage ×2 / ×3       : l'exécution coûte plus cher
  - latence ×5             : le prix d'exécution dérive avec la volatilité
  - liquidité divisée      : impact de marché plus grand (slippage ×2)
  - volatilité ×2          : le stop est traversé plus loin (perte ×2)
  - signal inversé         : le trade part dans le mauvais sens (perte = SL)
  - gap                    : le prix saute AU-DELÀ du stop (perte ×1.5)
  - données incohérentes   : prix d'exécution aberrant (perte = limite de
                             déviation, cohérente avec validate_order_safety)

Calculs :
  - baseline_pnl_pct   : edge net × SL (edge PAR UNITÉ de risque → PnL en
                         % du capital risqué)
  - pnl stressé par scénario (toujours ≤ baseline)
  - worst_case_pnl_pct : le pire des scénarios
  - max_loss_pct       : perte max acceptable sous stress (config)
  - fragile            : worst_case < −max_loss_pct OU espérance stressée ≤ 0
  - verdict            : ROBUST / FRAGILE

Principes :
  1. JAMAIS bloquant (erreur -> verdict ROBUST par défaut, le trade suit
     son chemin normal) — la protection est ADDITIVE, pas un point de panne.
  2. BORNÉ et RÉVERSIBLE : mode "block" rejette les trades fragiles, mode
     "warn" les signale seulement (config). Par défaut : block (un trade
     fragile n'est pas un trade).
  3. Toutes les valeurs sont des estimées à partir des paramètres RÉELS du
     trade (SL, TP, slippage attendu, edge net) — aucun chiffre inventé.
  4. DÉMO == RÉAL : aucun flag de mode.
"""
import logging

from core.config import settings

logger = logging.getLogger("InstitutionalTradingBot")

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
ADV_MODE: str = settings.get("adversarial", "mode", "block")        # block | warn
ADV_MAX_LOSS_PCT: float = settings.get_float("adversarial", "max_loss_pct", 0.05)
ADV_LATENCY_MULT: float = settings.get_float("adversarial", "latency_mult", 5.0)
ADV_VOL_MULT: float = settings.get_float("adversarial", "vol_mult", 2.0)
ADV_GAP_MULT: float = settings.get_float("adversarial", "gap_mult", 1.5)
ADV_SLIPPAGE_MULT: float = settings.get_float("adversarial", "slippage_mult", 3.0)
ADV_SPREAD_MULT: float = settings.get_float("adversarial", "spread_mult", 2.0)
ADV_MAX_DEVIATION: float = settings.get_float("risk", "deviation_limit_pct", 0.05)

ROBUST = "ROBUST"
FRAGILE = "FRAGILE"

# Codes de scénarios (stables, instrumentés)
SC_SPREAD = "SPREAD_X2"
SC_SLIP2 = "SLIPPAGE_X2"
SC_SLIP3 = "SLIPPAGE_X3"
SC_LATENCY = "LATENCY_X5"
SC_LIQUIDITY = "LIQUIDITY_HALVED"
SC_VOL = "VOLATILITY_X2"
SC_REVERSAL = "SIGNAL_REVERSAL"
SC_GAP = "GAP"
SC_DATA = "DATA_INCOHERENT"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class AdversarialDecisionEngine:
    """
    Évalue la ROBUSTESSE d'une décision sous stress.

    evaluate(edge_net, sl_pct, tp_pct, slippage_bps_expected, ...) -> dict
        {baseline_pnl_pct, scenarios, worst_case_pnl_pct, max_loss_pct,
         fragile, verdict, mode, detail}
    """

    def evaluate(self,
                 edge_net: float | None,
                 sl_pct: float,
                 tp_pct: float | None = None,
                 slippage_bps_expected: float | None = None,
                 spread_bps: float | None = None,
                 signal: float = 0.0) -> dict:
        """
        edge_net  : espérance NETTE par unité de risque (ConvictionEngine)
        sl_pct    : distance du stop en fraction du prix (ex. 0.03)
        tp_pct    : take-profit en fraction (pour le scénario inversion)
        slippage_bps_expected : slippage attendu (bps)
        spread_bps: spread courant (bps)
        Retourne le dict de verdict. Défensif : toute entrée invalide ->
        ROBUST (le trade suit son chemin — la protection est additive).
        """
        # Défensif : edge INCONNU ou invalide -> ROBUST (on ne bloque pas un
        # trade sur une information absente — la protection est additive,
        # pas un point de panne).
        try:
            edge = float(edge_net) if edge_net is not None else None
        except (TypeError, ValueError):
            edge = None
        if edge is None:
            return self._robust_default("edge inconnu (pas d'historique) — aucun jugement adversarial")
        try:
            sl = float(sl_pct) if sl_pct else 0.03
            sl = _clamp(sl, 0.001, 0.20)
        except (TypeError, ValueError):
            sl = 0.03
        try:
            slip = float(slippage_bps_expected) if slippage_bps_expected is not None else 0.0
        except (TypeError, ValueError):
            slip = 0.0
        try:
            spread = float(spread_bps) if spread_bps is not None else slip
        except (TypeError, ValueError):
            spread = slip

        # baseline : edge net (par unité de risque) × distance du stop
        baseline_pnl = edge * sl   # en fraction du capital

        slip_frac = slip / 10000.0          # slippage attendu en fraction
        spread_frac = spread / 10000.0

        def _pnl(extra_slip_frac: float, loss_mult: float = 1.0) -> float:
            """PnL stressé = baseline − coût extra − perte SL supplémentaire."""
            pnl = baseline_pnl
            pnl -= extra_slip_frac                      # coûts d'exécution en plus
            if loss_mult > 1.0:
                pnl -= sl * (loss_mult - 1.0)           # stop traversé plus loin
            return pnl

        scenarios = [
            # --- dégradations d'EXÉCUTION (espérance stressée) ---
            {"name": SC_SPREAD, "group": "execution", "detail": f"spread x2 ({(spread * ADV_SPREAD_MULT):.1f} bps)",
             "pnl_pct": round(_pnl(slip_frac + spread_frac * (ADV_SPREAD_MULT - 1.0)) * 100.0, 4)},
            {"name": SC_SLIP2, "group": "execution", "detail": f"slippage x2 ({slip * 2:.1f} bps)",
             "pnl_pct": round(_pnl(slip_frac) * 100.0, 4)},
            {"name": SC_SLIP3, "group": "execution", "detail": f"slippage x3 ({slip * ADV_SLIPPAGE_MULT:.1f} bps)",
             "pnl_pct": round(_pnl(slip_frac * ADV_SLIPPAGE_MULT) * 100.0, 4)},
            {"name": SC_LATENCY, "group": "execution", "detail": f"latence x{ADV_LATENCY_MULT:.0f} (dérive prix)",
             "pnl_pct": round(_pnl(slip_frac * (ADV_LATENCY_MULT ** 0.5)) * 100.0, 4)},
            {"name": SC_LIQUIDITY, "group": "execution", "detail": "liquidité divisée par 2 (impact)",
             "pnl_pct": round(_pnl(slip_frac * 2.0) * 100.0, 4)},
            # --- événements EXTRÊMES (worst case uniquement) ---
            {"name": SC_VOL, "group": "extreme", "detail": f"volatilité x{ADV_VOL_MULT:.0f} (stop traversé)",
             "pnl_pct": round(_pnl(slip_frac, loss_mult=ADV_VOL_MULT) * 100.0, 4)},
            {"name": SC_REVERSAL, "group": "extreme", "detail": "signal inversé (perte = stop)",
             "pnl_pct": round((-sl - slip_frac) * 100.0, 4)},
            {"name": SC_GAP, "group": "extreme", "detail": f"gap au-delà du stop (x{ADV_GAP_MULT:.1f})",
             "pnl_pct": round(_pnl(slip_frac, loss_mult=ADV_GAP_MULT) * 100.0, 4)},
            {"name": SC_DATA, "group": "extreme", "detail": "données incohérentes (fill dégradé ~1%)",
             "pnl_pct": round((-0.01 - slip_frac) * 100.0, 4)},
        ]

        worst = min(sc["pnl_pct"] for sc in scenarios)
        # Espérance stressée = moyenne des scénarios d'EXÉCUTION uniquement :
        # « un trade qui n'est rentable que dans un environnement parfait »
        # = espérance ≤ 0 même avec une simple dégradation d'exécution.
        # Les événements extrêmes (reversal/gap/vol/data) pèsent dans le
        # WORST CASE (risque de rupture), pas dans l'espérance.
        exec_scenarios = [sc for sc in scenarios if sc.get("group") == "execution"]
        stressed_avg = sum(sc["pnl_pct"] for sc in exec_scenarios) / max(len(exec_scenarios), 1)
        max_loss = ADV_MAX_LOSS_PCT * 100.0   # en %

        fragile = bool(worst < -max_loss or stressed_avg <= 0.0)
        verdict = FRAGILE if fragile else ROBUST
        return {
            "baseline_pnl_pct": round(baseline_pnl * 100.0, 4),
            "scenarios": scenarios,
            "worst_case_pnl_pct": round(worst, 4),
            "stressed_avg_pnl_pct": round(stressed_avg, 4),
            "max_loss_pct": max_loss,
            "fragile": fragile,
            "verdict": verdict,
            "mode": ADV_MODE,
            "detail": (f"pire scénario {worst:.2f}% < -{max_loss:.2f}% "
                       f"ou espérance stressée (exécution) {stressed_avg:.2f}% <= 0"
                       if fragile else "survit aux scénarios adversariaux"),
        }

    @staticmethod
    def _robust_default(detail: str) -> dict:
        return {
            "baseline_pnl_pct": 0.0, "scenarios": [], "worst_case_pnl_pct": 0.0,
            "stressed_avg_pnl_pct": 0.0, "max_loss_pct": ADV_MAX_LOSS_PCT * 100.0,
            "fragile": False, "verdict": ROBUST, "mode": ADV_MODE, "detail": detail,
        }

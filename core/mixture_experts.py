"""
VISION §2 - APPRENDRE: a cognitive architecture instead of one agent.

- Mixture of Experts per horizon (scalping / swing / position) with a gating
  network conditioned on regime + volatility
- risk-adjusted reward (Sharpe-like + drawdown penalty)
- OFFLINE RL: train experts on the replayable event journal (real past decisions)
- curriculum: sort training samples by volatility (calm first, then volatile)
"""
import logging
import time
from typing import Dict, List, Optional

import numpy as np

from models.price_predictor import PPOTRAgent

logger = logging.getLogger("MixtureExperts")

HORIZONS = ["scalping", "swing", "position"]


class HorizonExpert:
    """One PPO agent + its experience buffer for a given horizon."""

    def __init__(self, horizon: str, lookback: int, state_dim: int = 4):
        self.horizon = horizon
        self.lookback = lookback
        self.agent = PPOTRAgent(state_dim=state_dim, action_dim=1, hidden_dim=16)
        self.buffer: List[dict] = []
        self.max_buffer = 2000

    def collect(self, state, action, logp, reward, next_state, terminal=False):
        self.buffer.append({
            "state": state, "action": action, "log_prob": logp,
            "reward": float(reward), "next_state": next_state, "terminal": terminal,
        })
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer:]

    def train_offline(self, samples: List[dict]) -> int:
        """Trains on a list of (state, action, logp, reward, next) samples."""
        if len(samples) < 30:
            return 0
        self.agent.train_step(
            states=[s["state"] for s in samples],
            actions=[s["action"] for s in samples],
            log_probs_old=[s["log_prob"] for s in samples],
            rewards=[s["reward"] for s in samples],
            next_states=[s["next_state"] for s in samples],
            terminals=[s["terminal"] for s in samples],
        )
        return len(samples)


class MixtureOfExperts:
    """Gated ensemble of horizon-specialized PPO agents."""

    def __init__(self, state_dim: int = 4):
        self.experts = {
            "scalping": HorizonExpert("scalping", lookback=10, state_dim=state_dim),
            "swing": HorizonExpert("swing", lookback=60, state_dim=state_dim),
            "position": HorizonExpert("position", lookback=200, state_dim=state_dim),
        }
        self.gate_weights = np.array([0.4, 0.4, 0.2])
        # LOT 4 (PDF Pilier C) : contribution RÉELLE au PnL par expert + sommeil
        self.pnl_contrib = {h: 0.0 for h in HORIZONS}   # somme des PnL attribués
        self.expert_samples = {h: 0 for h in HORIZONS}  # nb de décisions tracées
        self.sleeping = set()                           # experts mis en sommeil

    def record_pnl_contribution(self, horizon: str, pnl_pct: float) -> None:
        """
        LOT 4 : évaluation RÉELLE des experts par contribution marginale au
        PnL (pas seulement par bandit sur signaux bruts). Chaque trade clôturé
        est attribué à l'expert dominant (horizon) du moment.
        """
        if horizon not in self.experts:
            return
        self.pnl_contrib[horizon] = self.pnl_contrib.get(horizon, 0.0) + float(pnl_pct)
        self.expert_samples[horizon] = self.expert_samples.get(horizon, 0) + 1

    def sleep_useless_experts(self, min_samples: int = 10, min_contrib_pct: float = 0.0) -> list:
        """
        LOT 4 : mise en SOMMEIL des experts inutiles — un expert dont la
        contribution au PnL est NÉGATIVE sur un échantillon suffisant est
        retiré du gate (poids 0) jusqu'à preuve du contraire.

        Mentalité n°17 : l'alpha décroît — on abandonne sans attachement ce
        qui s'érode. Retourne la liste des experts endormis.
        """
        newly_sleeping = []
        for h in HORIZONS:
            n = self.expert_samples.get(h, 0)
            contrib = self.pnl_contrib.get(h, 0.0)
            if n >= min_samples and contrib <= min_contrib_pct and h not in self.sleeping:
                self.sleeping.add(h)
                newly_sleeping.append(h)
                logger.warning(
                    f"🧟 MIXTURE-OF-EXPERTS: expert '{h}' MIS EN SOMMEIL "
                    f"(contribution PnL {contrib*100:.2f}% sur {n} trades)")
        return newly_sleeping

    def expert_contribution_report(self) -> Dict:
        """Rapport de contribution pour la télémétrie / l'audit."""
        return {
            h: {
                "pnl_contrib_pct": round(self.pnl_contrib.get(h, 0.0) * 100.0, 4),
                "n_trades": self.expert_samples.get(h, 0),
                "sleeping": h in self.sleeping,
            }
            for h in HORIZONS
        }

    def gate(self, regime_id: int, vol_mean: float) -> Dict[str, float]:
        """
        Soft gating conditioned on regime + volatility (VISION §2a).
        Bull/calm -> swing/position; erratic/high-vol -> scalping.
        """
        g = self.gate_weights.copy()
        if regime_id in (0, 1):          # bull/bear trend
            g[1] += 0.15                 # swing favored
        if regime_id == 3:               # erratic
            g[0] += 0.20                 # scalping favored
        if vol_mean > 0.004:
            g[0] += 0.15
        if vol_mean < 0.001:
            g[2] += 0.15
        g = np.clip(g, 0.05, 0.8)
        # LOT 4 : les experts endormis n'ont PAS le droit de voter (poids 0)
        for h in self.sleeping:
            g[HORIZONS.index(h)] = 0.0
        total = g.sum()
        if total <= 0:
            g = np.array([1/3, 1/3, 1/3])  # filet de sécurité (jamais de division par 0)
            total = g.sum()
        return {h: float(w) for h, w in zip(HORIZONS, g / total)}

    def decide(self, state: np.ndarray, regime_id: int, vol_mean: float) -> dict:
        """Each expert votes; the gate blends their actions (soft, not hard)."""
        gate = self.gate(regime_id, vol_mean)
        blended = 0.0
        votes = {}
        for h in HORIZONS:
            action, logp = self.experts[h].agent.get_action(state)
            votes[h] = round(float(action), 4)
            blended += gate[h] * float(action)
        return {"action": float(np.clip(blended, -1.0, 1.0)), "votes": votes, "gate": gate}

    def collect_experience(self, state, action, logp, reward, next_state, horizon: str = "swing"):
        self.experts.get(horizon, self.experts["swing"]).collect(state, action, logp, reward, next_state)


def risk_adjusted_reward(actual_return: float, action: float, equity_history: list,
                         impact_cost: float = 0.0005, drawdown_penalty: float = 2.0) -> float:
    """
    VISION §2b: reward = net return - impact - drawdown penalty.
    Penalizes actions when equity is below its running max (drawdown awareness).
    """
    r = float(actual_return) - impact_cost * abs(action)
    if len(equity_history) > 5:
        eq = np.array([float(e) for e in equity_history if isinstance(e, (int, float)) and e > 0])
        if len(eq) > 5:
            peak = float(np.max(eq))
            current = float(eq[-1])
            if peak > 0 and current < peak:
                dd = (peak - current) / peak
                r -= drawdown_penalty * dd * abs(action)
    return float(np.clip(r, -1.0, 1.0))


def curriculum_sort(samples: List[dict], vol_key: str = "vol") -> List[dict]:
    """VISION §2d: order samples by volatility, calm first, then volatile."""
    if not samples:
        return samples
    return sorted(samples, key=lambda s: float(s.get(vol_key, 0.0)))

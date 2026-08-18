"""
VISION §3 - INVENTER: the bot becomes a research scientist.

- generate_hypotheses(): mutates parameterized signal variants (real params)
- evaluate + promote: each candidate passes the Deflated-Sharpe gate; winners are
  admitted into the live signal set (registry), losers are recorded & retired
- meta-prior: per-family win rates steer future generation (Thompson sampling)
"""
import logging
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from core.signal_library import evaluate_signal, SIGNAL_LIBRARY

logger = logging.getLogger("HypothesisGen")

# parameterized variants: {family: {name: (fn, param_domain)}}
PARAM_SPACE = {
    "momentum_roc": {"period": (8, 96)},
    "momentum_cross": {"fast": (5, 30), "slow": (20, 80)},
    "rsi_meanrev": {"period": (5, 30)},
    "bollinger_revert": {"period": (10, 40), "std": (1.5, 3.0)},
    "vol_breakout": {"period": (10, 40)},
}

FAMILIES = ["momentum", "meanrev", "vol", "microstructure", "onchain", "sentiment"]


class HypothesisGenerator:
    """Generates, evaluates and promotes signal hypotheses autonomously."""

    def __init__(self, db=None):
        self.db = db
        self.meta_prior = {f: 1.0 for f in FAMILIES}   # Thompson alpha
        self.meta_failures = {f: 1.0 for f in FAMILIES}  # Thompson beta
        self.admitted: Dict[str, dict] = {}             # promoted signals
        self.max_admitted = 12
        # LOT 4 (PDF Pilier D) : discipline anti-fouille de données
        self.research_budget_per_day = 20      # max d'hypothèses testées / jour
        self.freeze_after_promotions = 3       # gel si >= 3 promotions récentes
        self.promotion_history: List[float] = []  # timestamps des promotions
        self._daily_reset_ts = 0.0
        self._today_tests = 0

    def can_run_research(self) -> dict:
        """
        LOT 4 : budget de recherche LIMITÉ (anti-fouille de données) + période
        de GEL des nouvelles hypothèses si trop de déploiements récents
        (mentalité n°3 : se méfier du surentraînement et de la fouille de
        données).
        """
        now = time.time()
        if now - self._daily_reset_ts > 86400:
            self._daily_reset_ts = now
            self._today_tests = 0
        if self._today_tests >= self.research_budget_per_day:
            return {"allowed": False,
                    "reason": f"budget quotidien épuisé ({self._today_tests}/{self.research_budget_per_day})"}
        recent = [t for t in self.promotion_history if now - t < 7 * 86400]
        if len(recent) >= self.freeze_after_promotions:
            return {"allowed": False,
                    "reason": f"gel anti-overfitting : {len(recent)} promotions récentes (>= {self.freeze_after_promotions})"}
        return {"allowed": True, "remaining": self.research_budget_per_day - self._today_tests}

    def consume_budget(self) -> None:
        self._today_tests += 1

    def register_promotion(self) -> None:
        self.promotion_history.append(time.time())
        self.promotion_history = [t for t in self.promotion_history
                                  if time.time() - t < 30 * 86400]

    def _family_of(self, name: str) -> str:
        if name.startswith("momentum"):
            return "momentum"
        if name.startswith(("rsi", "bollinger")):
            return "meanrev"
        if name.startswith("vol"):
            return "vol"
        if name.startswith("vpin"):
            return "microstructure"
        if name.startswith("onchain"):
            return "onchain"
        if name.startswith("sentiment"):
            return "sentiment"
        return "momentum"

    def generate_hypotheses(self, n: int = 8, rng=None) -> List[dict]:
        """Mutates real parameter variants (Thompson-biased by family)."""
        rng = rng or random
        candidates = []
        for _ in range(n):
            # sample a family by Thompson (meta-prior), then a base signal in it
            fam = self._sample_family(rng)
            pool = [k for k, v in PARAM_SPACE.items() if self._family_of(k) == fam] or list(PARAM_SPACE.keys())
            base = rng.choice(pool)
            params = {}
            for pname, (lo, hi) in PARAM_SPACE[base].items():
                params[pname] = round(rng.uniform(lo, hi), 2)
            candidates.append({
                "name": f"{base}_p{abs(hash(frozenset(params.items()))) % 100000}",
                "base": base, "params": params, "family": fam,
            })
        return candidates

    def _sample_family(self, rng) -> str:
        alphas = np.array([self.meta_prior[f] for f in FAMILIES])
        betas = np.array([self.meta_failures[f] for f in FAMILIES])
        samples = np.random.beta(alphas, betas)
        return FAMILIES[int(np.argmax(samples))]

    def evaluate_hypothesis(self, cand: dict, df: pd.DataFrame, market_data: dict,
                            num_trials: int = 30) -> dict:
        """Evaluates one candidate with the Deflated-Sharpe gate (real data)."""
        base_fn = SIGNAL_LIBRARY.get(cand["base"])
        if base_fn is None:
            return {"valid": False, "reason": "unknown base"}
        # wrap the parameterized variant
        def variant(df_i, md_i):
            params = cand["params"]
            if cand["base"] == "momentum_roc":
                return SIGNAL_LIBRARY["momentum_roc"](df_i, md_i, period=int(params["period"]))
            if cand["base"] == "momentum_cross":
                return SIGNAL_LIBRARY["momentum_cross"](df_i, md_i, fast=int(params["fast"]), slow=int(params["slow"]))
            if cand["base"] == "rsi_meanrev":
                return SIGNAL_LIBRARY["rsi_meanrev"](df_i, md_i, period=int(params["period"]))
            if cand["base"] == "bollinger_revert":
                return SIGNAL_LIBRARY["bollinger_revert"](df_i, md_i, period=int(params["period"]), std=float(params["std"]))
            if cand["base"] == "vol_breakout":
                return SIGNAL_LIBRARY["vol_breakout"](df_i, md_i, period=int(params["period"]))
            return base_fn(df_i, md_i)

        res = evaluate_signal(df, variant, market_data, fee_pct=0.001)
        res["name"] = cand["name"]
        res["family"] = cand["family"]
        res["params"] = cand["params"]
        return res

    def run_research_cycle(self, df: pd.DataFrame, market_data: dict,
                           promotion_threshold: float = 0.95, n_candidates: int = 8) -> dict:
        """
        One autonomous research cycle: generate -> evaluate -> promote/retire.
        Returns the promoted signals + stats.
        """
        # LOT 4 : budget de recherche + gel anti-overfitting
        gate = self.can_run_research()
        if not gate["allowed"]:
            logger.info(f"🧪 Recherche GELÉE: {gate['reason']}")
            return {"status": "FROZEN", "reason": gate["reason"],
                    "admitted": list(self.admitted.keys()), "results": []}
        candidates = self.generate_hypotheses(n_candidates)
        self.consume_budget()
        results = []
        promoted = []
        for cand in candidates:
            res = self.evaluate_hypothesis(cand, df, market_data)
            res["candidate"] = cand["name"]
            family = cand["family"]
            if res.get("valid") and res.get("deflated_sharpe", 0.0) >= promotion_threshold:
                promoted.append(cand)
                self.meta_prior[family] += 1.0
                self._admit(cand)
                logger.info(f"🧪 HYPOTHESIS PROMOTED: {cand['name']} (DSR {res['deflated_sharpe']:.3f})")
                self.register_promotion()
            else:
                self.meta_failures[family] += 1.0
                logger.info(f"🧪 hypothesis rejected: {cand['name']} (DSR {res.get('deflated_sharpe', 0.0):.3f})")
            results.append(res)
            if self.db is not None:
                try:
                    self.db.add_experiment(
                        hypothesis=f"[AUTO] {cand['name']} params={cand['params']}",
                        status="PROMOTED" if res.get("valid") and res.get("deflated_sharpe", 0) >= promotion_threshold else "REJECTED",
                        result=f"DSR={res.get('deflated_sharpe', 0.0):.3f} sharpe={res.get('sharpe', 0.0):.3f}",
                    )
                except Exception:
                    pass
        return {"candidates": len(candidates), "promoted": [c["name"] for c in promoted],
                "admitted": list(self.admitted.keys()), "results": results}

    def _admit(self, cand: dict):
        self.admitted[cand["name"]] = cand
        if len(self.admitted) > self.max_admitted:
            # retire the oldest admitted (simple FIFO)
            self.admitted.pop(next(iter(self.admitted)))

    def get_status(self) -> dict:
        return {
            "admitted_signals": list(self.admitted.keys()),
            "meta_prior": {k: round(v, 2) for k, v in self.meta_prior.items()},
            "meta_failures": {k: round(v, 2) for k, v in self.meta_failures.items()},
        }

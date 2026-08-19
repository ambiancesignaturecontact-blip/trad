"""
ATTRIBUTION DE PERFORMANCE & MÉTRIQUES (PROMPT MAÎTRE, Pilier Q).

Un pro sait EXACTEMENT d'où vient chaque dollar : « ce dollar vient du
momentum BTC en régime haussier », pas « on a gagné 2 % ».

Ce module calcule :
  1. Métriques de qualité : Sharpe, Sortino, Calmar, max drawdown, win rate,
     profit factor, expectancy — depuis une courbe d'équité et les trades.
  2. Attribution par FACTEUR (market / momentum / carry / vol) — chaque trade
     est classé dans le facteur dominant de sa stratégie.
  3. Attribution par régime de marché + actif + stratégie (consolidation).

Le tout alimente le rapport quotidien et le dashboard (mentalité n°9 :
tout doit être traçable).
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("Attribution")

# Mapping stratégie -> facteur dominant (documenté)
STRATEGY_FACTOR = {
    "Trend Following": "momentum",
    "Momentum": "momentum",
    "Cross-Sectional Momentum": "momentum",
    "Mean Reversion": "meanrev",
    "Market Making": "carry",
    "Carry": "carry",
    "Statistical Arbitrage": "market",
    "Inter-Exchange Arbitrage": "market",
    "Grid Trading": "meanrev",
    "Scalping": "market",
    "Volatility Breakout": "vol",
    "Multi-Timeframe": "momentum",
    "META_MODEL": "market",
}


# --------------------------------------------------------------------------- #
# 1. MÉTRIQUES DE QUALITÉ
# --------------------------------------------------------------------------- #
def quality_metrics(equity_curve: List[float],
                    trades: Optional[List[dict]] = None,
                    periods_per_year: float = 365.0) -> dict:
    """
    Métriques institutionnelles depuis la courbe d'équité :
      Sharpe  = mean(ret)/std(ret) * sqrt(periods)
      Sortino = mean(ret)/downside_std(ret) * sqrt(periods)
      Calmar  = annual_return / max_drawdown
      MaxDD   = plus grand drawdown observé
    Depuis les trades (si fournis) :
      win rate, profit factor, expectancy (espérance par trade).
    """
    if not equity_curve or len(equity_curve) < 3:
        return {"available": False, "reason": "courbe d'équité insuffisante"}

    eq = np.asarray(equity_curve, dtype=float)
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
    rets = rets[~np.isnan(rets)]

    if len(rets) < 2:
        return {"available": False, "reason": "rendements insuffisants"}

    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets)) + 1e-12
    sharpe = mean_r / std_r * np.sqrt(periods_per_year)

    downside = rets[rets < 0]
    downside_std = float(np.std(downside)) + 1e-12 if len(downside) > 0 else std_r
    sortino = mean_r / downside_std * np.sqrt(periods_per_year)

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-9)
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    # Rendement annualisé approximatif + Calmar
    total_ret = eq[-1] / eq[0] - 1.0 if eq[0] > 0 else 0.0
    n_periods = max(len(rets), 1)
    ann_ret = (1.0 + total_ret) ** (periods_per_year / n_periods) - 1.0 if total_ret > -1 else -1.0
    calmar = ann_ret / max_dd if max_dd > 1e-9 else 0.0

    result = {
        "available": True,
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "calmar": round(float(calmar), 4),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "annual_return_pct": round(ann_ret * 100.0, 2),
        "n_periods": int(len(rets)),
    }

    # Métriques depuis les trades (si fournis)
    if trades:
        pnls = [float(t.get("pnl", 0.0)) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        result["win_rate_pct"] = round(len(wins) / max(len(pnls), 1) * 100.0, 2)
        result["profit_factor"] = round(gross_win / max(gross_loss, 1e-9), 4)
        result["expectancy_pct"] = round(float(np.mean(pnls)) * 100.0, 4) if pnls else 0.0
        result["n_trades"] = len(pnls)
    return result


# --------------------------------------------------------------------------- #
# 2. ATTRIBUTION PAR FACTEUR / RÉGIME / ACTIF
# --------------------------------------------------------------------------- #
class PerformanceAttribution:
    """Attribue chaque trade clôturé à son facteur, régime, actif et stratégie."""

    def __init__(self):
        self.trades: List[dict] = []

    def record(self, symbol: str, strategy: str, pnl_pct: float,
               regime_name: str = "", pnl_usd: float = 0.0) -> None:
        """Enregistre un trade clôturé avec son contexte d'attribution."""
        factor = STRATEGY_FACTOR.get(strategy, "market")
        self.trades.append({
            "symbol": symbol, "strategy": strategy, "factor": factor,
            "regime": regime_name or "Unknown", "pnl_pct": float(pnl_pct),
            "pnl_usd": float(pnl_usd), "ts": __import__("time").time(),
        })
        if len(self.trades) > 2000:
            self.trades = self.trades[-2000:]

    def by_factor(self) -> Dict[str, dict]:
        """PnL par FACTEUR (market/momentum/carry/vol/meanrev)."""
        out: Dict[str, dict] = {}
        for t in self.trades:
            f = t["factor"]
            s = out.setdefault(f, {"pnl_pct": 0.0, "n": 0, "wins": 0})
            s["pnl_pct"] += t["pnl_pct"]
            s["n"] += 1
            if t["pnl_pct"] > 0:
                s["wins"] += 1
        for s in out.values():
            s["win_rate_pct"] = round(s["wins"] / max(s["n"], 1) * 100.0, 1)
            s["pnl_pct"] = round(s["pnl_pct"], 4)
        return out

    def by_regime(self) -> Dict[str, dict]:
        """PnL par RÉGIME de marché (haussier, baissier, range, erratique)."""
        out: Dict[str, dict] = {}
        for t in self.trades:
            r = t["regime"]
            s = out.setdefault(r, {"pnl_pct": 0.0, "n": 0})
            s["pnl_pct"] += t["pnl_pct"]
            s["n"] += 1
        for s in out.values():
            s["pnl_pct"] = round(s["pnl_pct"], 4)
        return out

    def by_asset(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for t in self.trades:
            a = t["symbol"]
            s = out.setdefault(a, {"pnl_pct": 0.0, "n": 0, "wins": 0})
            s["pnl_pct"] += t["pnl_pct"]
            s["n"] += 1
            if t["pnl_pct"] > 0:
                s["wins"] += 1
        for s in out.values():
            s["win_rate_pct"] = round(s["wins"] / max(s["n"], 1) * 100.0, 1)
            s["pnl_pct"] = round(s["pnl_pct"], 4)
        return out

    def by_strategy(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for t in self.trades:
            st = t["strategy"]
            s = out.setdefault(st, {"pnl_pct": 0.0, "n": 0, "wins": 0})
            s["pnl_pct"] += t["pnl_pct"]
            s["n"] += 1
            if t["pnl_pct"] > 0:
                s["wins"] += 1
        for s in out.values():
            s["win_rate_pct"] = round(s["wins"] / max(s["n"], 1) * 100.0, 1)
            s["pnl_pct"] = round(s["pnl_pct"], 4)
        return out

    def full_report(self) -> dict:
        return {
            "by_factor": self.by_factor(),
            "by_regime": self.by_regime(),
            "by_asset": self.by_asset(),
            "by_strategy": self.by_strategy(),
            "n_trades": len(self.trades),
        }

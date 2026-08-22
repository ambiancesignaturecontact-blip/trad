"""
CONTINUOUS BENCHMARKING (PHASE 3 — §8).

Compare la performance RÉELLE du bot (paper) aux benchmarks simples :
  - Buy & Hold par actif (rendement de l'actif sur la même fenêtre)
  - Stratégie naïve (buy & hold du panier moyen)
  - Alpha = rendement bot − rendement benchmark (même fenêtre)

Utilise UNIQUEMENT des données réelles : clôtures du journal + candles DB.
Sans échantillon -> dict honnête « insuffisant », jamais de chiffre inventé.
"""
import json
import logging
import time

logger = logging.getLogger("InstitutionalTradingBot")


def _returns_from_closed(db, since_ts: float = 0.0) -> list:
    """pnl_pct des trades clôturés (events closed_trade_alpha) depuis since."""
    try:
        evs = db.list_events(event_type="closed_trade_alpha", since=since_ts, limit=5000)
        out = []
        for e in evs:
            try:
                d = json.loads(e.get("payload", "{}"))
                p = float(d.get("pnl_pct", 0.0))
                out.append({"symbol": d.get("symbol", "?"), "pnl_pct": p,
                            "strategy": d.get("strategy", "?")})
            except Exception:
                continue
        return out
    except Exception as e:
        logger.warning(f"benchmark returns failed ({e})")
        return []


def buy_and_hold_return(db, symbol: str, window_bars: int = 400) -> float | None:
    """Rendement buy & hold de l'actif sur la fenêtre de candles disponible."""
    try:
        df = db.load_candles(symbol, limit=window_bars)
        if df is None or df.empty or len(df) < 20:
            return None
        first = float(df["close"].iloc[0])
        last = float(df["close"].iloc[-1])
        if first <= 0:
            return None
        return (last / first - 1.0) * 100.0
    except Exception:
        return None


def benchmark_report(db, since_ts: float = 0.0,
                     symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSD",
                              "EURUSD", "AAPL", "TSLA")) -> dict:
    """
    Rapport de benchmarking :
      - trades clôturés : n, PnL cumulé %, expectancy, par actif
      - buy & hold par actif (même fenêtre de candles)
      - alpha estimé : PnL bot cumulé − moyenne des buy&hold (indicatif)
    """
    closed = _returns_from_closed(db, since_ts=since_ts)
    if not closed:
        return {"n_closed": 0, "note": "aucun trade clôturé dans la fenêtre — "
                                       "benchmark impossible (aucun chiffre inventé)",
                "ts": time.time()}

    total_pnl = sum(t["pnl_pct"] for t in closed) * 100.0
    n = len(closed)
    by_symbol = {}
    for t in closed:
        by_symbol.setdefault(t["symbol"], []).append(t["pnl_pct"])

    bnh = {}
    for s in symbols:
        r = buy_and_hold_return(db, s)
        if r is not None:
            bnh[s] = round(r, 2)

    # alpha indicatif : PnL bot cumulé vs moyenne des buy&hold disponibles
    bnh_values = [v for v in bnh.values()]
    avg_bnh = sum(bnh_values) / len(bnh_values) if bnh_values else None

    return {
        "n_closed": n,
        "bot_pnl_cumul_pct": round(total_pnl, 2),
        "bot_expectancy_pct": round(total_pnl / n, 3),
        "bot_by_symbol": {s: round(sum(ps) * 100.0, 2) for s, ps in by_symbol.items()},
        "buy_and_hold_pct": bnh,
        "alpha_vs_avg_bnh_pct": round(total_pnl - (avg_bnh or 0.0), 2)
        if avg_bnh is not None else None,
        "note": "Comparaison indicative sur fenêtres différentes (trades vs candles) — "
                "l'alpha n'est une preuve qu'avec des fenêtres alignées et n >= 30.",
        "ts": time.time(),
    }


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 3 — BOOTSTRAP SHARPE (limite PHASE 2 : équité en mémoire,
# non persistée -> bootstrap impossible). L'équité est désormais persistée
# par `equity_history_scheduler` (toutes les 5 min) ; cette fonction calcule
# les rendements journaliers et bootstrap le Sharpe (López de Prado, AFML).
# Honnête : < MIN_DAYS points quotidiens -> « insuffisant », jamais inventé.
# --------------------------------------------------------------------------- #
MIN_EQUITY_DAYS = 20
EQUITY_BOOTSTRAP_SIMS = 10_000


def bootstrap_sharpe(db, mode: str = "DEMO", n_sims: int = EQUITY_BOOTSTRAP_SIMS,
                     min_days: int = MIN_EQUITY_DAYS) -> dict:
    """
    Sharpe annualisé OBSERVÉ + intervalle de confiance bootstrap (ré-échantillon
    avec remise des rendements quotidiens). Retourne un dict honnête :
    n_days < min_days -> sharpe None + note explicite.
    """
    out = {"mode": mode, "n_points": 0, "n_days": 0, "sharpe_obs": None,
           "sharpe_p5": None, "sharpe_p95": None, "note": None}
    try:
        if db is None or not hasattr(db, "load_equity_series"):
            out["note"] = "persistance équité indisponible"
            return out
        series = db.load_equity_series(mode=mode)
        out["n_points"] = len(series)
        if len(series) < 3:
            out["note"] = (f"insuffisant : {len(series)} points d'équité persistés "
                           "(le scheduler écrit toutes les 5 min)")
            return out
        import numpy as np
        import pandas as pd
        eq = pd.Series([e for _, e in series],
                       index=pd.to_datetime([t for t, _ in series], unit="s", utc=True))
        # rendements journaliers (dernier prix du jour vs dernier du jour préc.)
        daily = eq.resample("1D").last().dropna()
        ret = daily.pct_change().dropna()
        ret = ret[(ret > -1.0) & (ret < 10.0)]  # garde-fou valeurs aberrantes
        out["n_days"] = int(len(ret))
        if len(ret) < min_days:
            out["note"] = (f"insuffisant : {len(ret)} rendements journaliers "
                           f"(< {min_days}) — le bootstrap n'est pas significatif")
            return out
        sr = ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else None
        out["sharpe_obs"] = round(float(sr), 3) if sr is not None else None
        # bootstrap : ré-échantillon avec remise de la série quotidienne
        rng = np.random.default_rng(2026)
        n = len(ret)
        boot = []
        for _ in range(n_sims):
            s = rng.choice(ret.values, size=n, replace=True)
            sd = s.std()
            boot.append(s.mean() / sd * np.sqrt(365) if sd > 0 else 0.0)
        boot = np.asarray(boot)
        out["sharpe_p5"] = round(float(np.percentile(boot, 5)), 3)
        out["sharpe_p95"] = round(float(np.percentile(boot, 95)), 3)
        out["note"] = (f"bootstrap {n_sims} sims sur {len(ret)} jours "
                       f"([p5, p95] = [{out['sharpe_p5']}, {out['sharpe_p95']}])")
    except Exception as e:
        out["note"] = f"indisponible ({e})"
    return out

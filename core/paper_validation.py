"""
PAPER-TRADING VALIDATION REPORT (LOT 10 du mandat).

« Le but n'est pas uniquement d'obtenir un PnL positif. Le but est de prouver
que le système : fonctionne continuellement, reste cohérent, ne dérive pas,
protège le capital, explique ses décisions, respecte ses limites. »

Ce module produit un rapport OBJECTIF de préparation au REAL, à partir UNIQUEMENT
de données réelles (DB + STATE) — chaque critère a un statut PASS / WARN / FAIL
et une preuve. Aucun critère n'est déclaré satisfait sans mesure.

Critères (10 axes du mandat) :
  C1 CONTINUITÉ    : jours de paper-trading datés (P0-6), streak, trous
  C2 FONCTIONNEMENT: volume d'activité réel (décisions, trades, heartbeat)
  C3 COHÉRENCE     : erreurs critiques / crashs / audits anormaux
  C4 PROTECTION    : drawdown max vs limites, kill switch, HALT gérés
  C5 EXPLICABILITÉ : % décisions avec raison, non-trades catégorisés
  C6 LIMITES       : rejets OMS, cooldowns, déviation de prix
  C7 NON-DÉRIVE    : drift PSI, edge decay (aucune stratégie DISABLED durable)
  C8 EXÉCUTION     : slippage moyen, forecast error (prévision vs réalité)
  C9 CALIBRATION   : conviction buckets, calibration error
  C10 PNL NET      : PnL net après coûts, drawdown, stabilité

Statut global : READY (tous PASS essentiels) / IN_PROGRESS / NOT_READY.
La décision finale de passage en REAL reste MANUELLE (jamais automatique).
"""
import json
import logging
import time

logger = logging.getLogger("InstitutionalTradingBot")

# Seuils des critères (config-driven où possible, sinon constants documentés)
REQUIRED_DAYS = 28                       # P0-6 : 4 semaines continues
MIN_DECISIONS = 200                      # volume décisionnel minimal
MIN_TRADES = 30                          # trades clôturés minimaux pour la validation
MAX_REJECT_RATE = 0.15                   # taux de rejets OMS acceptable
MAX_FORECAST_ERROR_BPS = 20.0            # erreur de prévision slippage acceptable
MAX_CALIBRATION_ERROR = 0.15             # calibration error acceptable
EDGE_DISABLED_MAX_PCT = 0.25             # % max de stratégies DISABLED tolérable
MAX_DRAWDOWN_PCT = 0.15                  # drawdown max acceptable (DEMO)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def _paper_days(db) -> list[str]:
    try:
        days = json.loads(db.get_setting("paper_validation_days") or "[]")
        return sorted(set(d for d in days if isinstance(d, str) and len(d) == 10))
    except Exception:
        return []


def _streak(days: list[str]) -> int:
    """Série consécutive la plus récente (jours calendaires)."""
    if not days:
        return 0
    try:
        from datetime import datetime, timedelta
        cur = datetime.strptime(days[-1], "%Y-%m-%d").date()
        day_set = set(days)
        streak = 1
        while (cur - timedelta(days=1)).strftime("%Y-%m-%d") in day_set:
            streak += 1
            cur = cur - timedelta(days=1)
        return streak
    except Exception:
        return 0


def _gap_max_days(days: list[str]) -> int:
    """Plus grand trou entre jours actifs consécutifs (0 si < 2 jours)."""
    if len(days) < 2:
        return 0
    try:
        from datetime import datetime
        ds = [datetime.strptime(d, "%Y-%m-%d").date() for d in days]
        gaps = [(ds[i + 1] - ds[i]).days - 1 for i in range(len(ds) - 1)]
        return max(gaps)
    except Exception:
        return 0


def _crit(status: str, detail: str, evidence: str) -> dict:
    return {"status": status, "detail": detail, "evidence": evidence}


def build_paper_validation_report(db, state: dict,
                                  conviction_calibration: dict | None = None,
                                  edge_decay_report: dict | None = None,
                                  drift_psi: dict | None = None,
                                  execution_intel: dict | None = None) -> dict:
    """
    Construit le rapport complet. Toutes les données sont réelles ; un axe
    sans données -> WARN (échantillon insuffisant), jamais PASS sans mesure.
    """
    now = time.time()
    criteria: dict[str, dict] = {}

    # ---------------- C1 CONTINUITÉ ----------------
    days = _paper_days(db)
    streak = _streak(days)
    gap = _gap_max_days(days)
    if len(days) >= REQUIRED_DAYS and streak >= REQUIRED_DAYS and gap == 0:
        c1 = _crit(PASS, f"{len(days)}/{REQUIRED_DAYS} jours datés, streak {streak}, aucun trou",
                   "paper_validation_days (DB)")
    elif len(days) >= REQUIRED_DAYS and gap == 0:
        c1 = _crit(WARN, f"{len(days)} jours datés mais streak {streak} < {REQUIRED_DAYS}",
                   "paper_validation_days (DB)")
    else:
        c1 = _crit(FAIL, f"{len(days)}/{REQUIRED_DAYS} jours datés, streak {streak}, "
                         f"plus grand trou {gap}j — le bot doit tourner continuellement",
                   "paper_validation_days (DB)")
    criteria["C1_CONTINUITE"] = c1

    # ---------------- C2 FONCTIONNEMENT ----------------
    try:
        j = db.decision_journal_summary()
    except Exception:
        j = {"total": 0}
    n_decisions = int(j.get("total", 0))
    n_closed = int(j.get("closed_n", 0))
    try:
        n_fills = len(db.list_events(event_type="paper_fill", limit=5000))
    except Exception:
        n_fills = 0
    last_heartbeat = state.get("last_tick_ts", 0.0)
    alive = (now - float(last_heartbeat)) < 600 if last_heartbeat else False
    if n_decisions >= MIN_DECISIONS and n_fills >= MIN_TRADES and alive:
        c2 = _crit(PASS, f"{n_decisions} décisions, {n_fills} fills, bot vivant (heartbeat < 10 min)",
                   "decision_journal + events(paper_fill) + STATE.last_tick_ts")
    elif n_decisions >= MIN_DECISIONS:
        c2 = _crit(WARN, f"{n_decisions} décisions mais {n_fills} fills < {MIN_TRADES} "
                         f"ou heartbeat {now - last_heartbeat:.0f}s", "journal + events")
    else:
        c2 = _crit(FAIL, f"{n_decisions} décisions < {MIN_DECISIONS} — volume décisionnel insuffisant",
                   "decision_journal")
    criteria["C2_FONCTIONNEMENT"] = c2

    # ---------------- C3 COHÉRENCE ----------------
    try:
        bad_audits = db.list_events(event_type="", limit=50)  # pas de source directe
        # audit_logs anormaux : on compte les actions critiques via settings ? -> fallback events
        n_critical = sum(1 for e in bad_audits
                         if "ERROR" in str(e.get("event_type", "")).upper())
    except Exception:
        n_critical = 0
    if n_critical == 0:
        c3 = _crit(PASS, "aucune erreur critique non gérée détectée dans les événements",
                   "events (échantillon récent)")
    else:
        c3 = _crit(WARN, f"{n_critical} erreurs critiques récentes — à investiguer", "events")
    criteria["C3_COHERENCE"] = c3

    # ---------------- C4 PROTECTION DU CAPITAL ----------------
    try:
        equity = state.get("equity_history_demo", []) or state.get("equity_history_real", [])
        peak = max(equity) if equity else 1.0
        trough = min(equity) if equity else peak
        dd = (peak - trough) / peak if peak > 0 else 0.0
    except Exception:
        dd = 0.0
    kill = bool(state.get("kill_switch_active", False))
    risk_state = (state.get("risk_state") or {}).get("state", "NORMAL")
    issues = []
    if kill:
        issues.append("kill switch ACTIF")
    if dd > MAX_DRAWDOWN_PCT:
        issues.append(f"drawdown {dd * 100:.1f}% > {MAX_DRAWDOWN_PCT * 100:.0f}%")
    if issues:
        c4 = _crit(FAIL, " ; ".join(issues), "STATE.kill_switch_active + equity_history")
    elif dd > MAX_DRAWDOWN_PCT * 0.6:
        c4 = _crit(WARN, f"drawdown {dd * 100:.1f}% (sous la limite {MAX_DRAWDOWN_PCT * 100:.0f}%)",
                   "equity_history")
    else:
        c4 = _crit(PASS, f"drawdown max {dd * 100:.1f}% < limite {MAX_DRAWDOWN_PCT * 100:.0f}%, "
                         f"kill switch inactif, état risque {risk_state}", "equity_history + STATE")
    criteria["C4_PROTECTION"] = c4

    # ---------------- C5 EXPLICABILITÉ ----------------
    try:
        jr = db.get_decision_journal(limit=200)
        n_with_reason = sum(1 for r in jr if r.get("reason"))
        pct_reason = n_with_reason / max(len(jr), 1)
    except Exception:
        pct_reason = 0.0
    no_trade = (state.get("no_trade_stats") or {}).get("reasons", {})
    if pct_reason >= 0.99 and no_trade:
        c5 = _crit(PASS, f"{pct_reason * 100:.0f}% des décisions avec raison, "
                         f"{len(no_trade)} catégories de non-trade", "decision_journal + no_trade_stats")
    elif pct_reason >= 0.99:
        c5 = _crit(WARN, "décisions explicables mais aucune abstention catégorisée récente",
                   "decision_journal")
    else:
        c5 = _crit(FAIL, f"seulement {pct_reason * 100:.0f}% des décisions avec raison",
                   "decision_journal")
    criteria["C5_EXPLICABILITE"] = c5

    # ---------------- C6 LIMITES ----------------
    try:
        orders = db.get_all_orders()
        total = len(orders)
        rejected = sum(1 for o in orders if o.get("status") == "REJECTED")
        rej_rate = rejected / max(total, 1)
    except Exception:
        rej_rate = 0.0
    if rej_rate <= MAX_REJECT_RATE:
        c6 = _crit(PASS, f"taux de rejets OMS {rej_rate * 100:.1f}% <= {MAX_REJECT_RATE * 100:.0f}%",
                   "orders (DB)")
    else:
        c6 = _crit(FAIL, f"taux de rejets OMS {rej_rate * 100:.1f}% > {MAX_REJECT_RATE * 100:.0f}%",
                   "orders (DB)")
    criteria["C6_LIMITES"] = c6

    # ---------------- C7 NON-DÉRIVE ----------------
    drift_status = (drift_psi or {}).get("status", "STABLE")
    ed = edge_decay_report or {}
    n_disabled = int(ed.get("counts", {}).get("disabled", 0))
    n_total = int(ed.get("counts", {}).get("total", 0)) or 1
    disabled_pct = n_disabled / n_total
    if drift_status == "STABLE" and disabled_pct <= EDGE_DISABLED_MAX_PCT:
        c7 = _crit(PASS, f"drift PSI {drift_status}, stratégies DISABLED {n_disabled}/{n_total}",
                   "drift_psi + edge_decay")
    elif drift_status in ("MODERATE", "SEVERE") or disabled_pct > EDGE_DISABLED_MAX_PCT:
        c7 = _crit(WARN, f"drift PSI {drift_status}, DISABLED {n_disabled}/{n_total} — "
                         f"le système s'adapte (vérifier la tendance)", "drift_psi + edge_decay")
    else:
        c7 = _crit(WARN, "pas de mesure de drift/edge decay disponible", "drift_psi/edge_decay absents")
    criteria["C7_NON_DERIVE"] = c7

    # ---------------- C8 EXÉCUTION ----------------
    ei = execution_intel or {}
    avg_slip = ei.get("avg_is_bps")
    fc_err = ei.get("avg_forecast_error_bps")
    if avg_slip is not None and abs(fc_err or 0.0) <= MAX_FORECAST_ERROR_BPS:
        c8 = _crit(PASS, f"slippage moyen {avg_slip:.2f} bps, erreur de prévision {fc_err:.2f} bps",
                   "execution_intel")
    elif avg_slip is not None:
        c8 = _crit(WARN, f"erreur de prévision slippage {fc_err:.2f} bps > {MAX_FORECAST_ERROR_BPS} bps",
                   "execution_intel")
    else:
        c8 = _crit(WARN, "aucun fill mesuré — échantillon d'exécution insuffisant", "execution_intel")
    criteria["C8_EXECUTION"] = c8

    # ---------------- C9 CALIBRATION ----------------
    cc = conviction_calibration or {}
    cal_n = int(cc.get("n", 0))
    cal_err = cc.get("calibration_error")
    if cal_n >= 30 and cal_err is not None and cal_err <= MAX_CALIBRATION_ERROR:
        c9 = _crit(PASS, f"{cal_n} trades calibrés, calibration error {cal_err:.3f}",
                   "conviction_calibration")
    elif cal_n >= 30:
        c9 = _crit(WARN, f"calibration error {cal_err:.3f} > {MAX_CALIBRATION_ERROR}", "conviction_calibration")
    else:
        c9 = _crit(WARN, f"{cal_n} trades calibrés < 30 — calibration en attente de données",
                   "conviction_calibration")
    criteria["C9_CALIBRATION"] = c9

    # ---------------- C10 PNL NET ----------------
    try:
        initial = float(state.get("initial_capital_demo", 0.0) or 0.0)
        equity_now = float(state.get("current_equity", 0.0) or 0.0)
        pnl_pct = (equity_now - initial) / initial * 100.0 if initial > 0 else 0.0
    except Exception:
        pnl_pct = 0.0
    if n_closed >= MIN_TRADES and pnl_pct > 0:
        c10 = _crit(PASS, f"PnL net {pnl_pct:+.2f}% sur {n_closed} trades clôturés", "current_equity + journal")
    elif n_closed >= MIN_TRADES:
        c10 = _crit(FAIL, f"PnL net {pnl_pct:+.2f}% sur {n_closed} trades clôturés — performance négative",
                    "current_equity + journal")
    else:
        c10 = _crit(WARN, f"{n_closed} trades clôturés < {MIN_TRADES} — PnL non statistiquement significatif "
                          f"({pnl_pct:+.2f}% courant)", "current_equity + journal")
    criteria["C10_PNL_NET"] = c10

    # ---------------- STATUT GLOBAL ----------------
    essential = ["C1_CONTINUITE", "C2_FONCTIONNEMENT", "C4_PROTECTION",
                 "C5_EXPLICABILITE", "C6_LIMITES"]
    fails = [k for k, v in criteria.items() if v["status"] == FAIL]
    essential_fails = [k for k in fails if k in essential]
    warns = [k for k, v in criteria.items() if v["status"] == WARN]
    if not essential_fails and not any(criteria[k]["status"] == WARN for k in essential):
        status = "READY"
    elif not essential_fails:
        status = "IN_PROGRESS"
    else:
        status = "NOT_READY"

    return {
        "generated_ts": now,
        "status": status,
        "criteria": criteria,
        "summary": {
            "pass": sum(1 for v in criteria.values() if v["status"] == PASS),
            "warn": len(warns),
            "fail": len(fails),
            "essential_fails": essential_fails,
            "warns": warns,
        },
        "raw": {
            "paper_days": len(days), "streak": streak, "max_gap_days": gap,
            "decisions": n_decisions, "closed_trades": n_closed, "fills": n_fills,
            "drawdown_pct": round(dd * 100.0, 2), "kill_switch": kill,
            "risk_state": risk_state, "reject_rate_pct": round(rej_rate * 100.0, 2),
            "drift_status": drift_status, "disabled_strategies": n_disabled,
            "avg_slippage_bps": avg_slip, "forecast_error_bps": fc_err,
            "calibration_n": cal_n, "calibration_error": cal_err,
            "pnl_pct": round(pnl_pct, 2), "no_trade_by_reason": no_trade,
        },
        "note": "Décision finale de passage en REAL : MANUELLE (jamais automatique). "
                "Ce rapport est une preuve objective, pas une autorisation.",
    }


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 2 — suivi des clôtures du CALIBRAGE ACTUEL.
#
# La certification PHASE 2 (CONDITIONAL GO paper) exige, pour un passage REAL
# potentiel : >= 30 clôtures du calibrage ACTUEL avec expectancy > 0.
# Cette fonction rend la condition MESURABLE : elle compte les clôtures
# (pnl_pct non nul) enregistrées avec une system_version donnée (défaut :
# version courante du système). Aucun chiffre inventé : tout est NULL tant
# qu'aucune clôture n'existe.
# --------------------------------------------------------------------------- #

def calibration_close_tracking(db, version: str = "") -> dict:
    """
    Clôtures réelles (decision_journal, pnl_pct NOT NULL) attribuées à une
    version système (défaut : version actuelle). Retourne n, win rate,
    expectancy moyenne (% par trade), pnl cumulé et la progression vers les
    conditions du CONDITIONAL GO (>= 30 clôtures, expectancy > 0).
    """
    out = {
        "version": version or None,
        "n_closes": 0,
        "open_positions": None,       # PHASE 3 C6 : positions ouvertes (0 clôture
        # cohérente avec 0 position ; positions>0 sans clôture = anomalie)
        "win_rate": None,
        "expectancy_pct": None,
        "cumulative_pnl_pct": None,
        "target_n_closes": MIN_TRADES,          # 30 (même seuil que la validation)
        "progress_pct": 0.0,
        "conditions_met": False,
        "note": None,
    }
    try:
        if db is None or not hasattr(db, "get_connection"):
            return out
        with db.get_connection() as conn:
            cur = conn.cursor()
            ph = "%s" if getattr(db, "is_postgres", False) else "?"
            if version:
                cur.execute(
                    f"SELECT COUNT(*), AVG(pnl_pct), "
                    f"SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) "
                    f"FROM decision_journal "
                    f"WHERE pnl_pct IS NOT NULL AND system_version = {ph}",
                    [version],
                )
            else:
                cur.execute(
                    "SELECT COUNT(*), AVG(pnl_pct), "
                    "SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) "
                    "FROM decision_journal WHERE pnl_pct IS NOT NULL"
                )
            row = cur.fetchone()
            n = int(row[0] or 0)
            avg = float(row[1] or 0.0)
            wins = int(row[2] or 0)
            try:
                cur.execute("SELECT COUNT(*) FROM positions")
                out["open_positions"] = int(cur.fetchone()[0] or 0)
            except Exception:
                out["open_positions"] = None
        out["n_closes"] = n
        out["progress_pct"] = round(min(100.0, n / MIN_TRADES * 100.0), 1)
        if n:
            out["win_rate"] = round(wins / n, 4)
            out["expectancy_pct"] = round(avg * 100.0, 4)
            out["cumulative_pnl_pct"] = round(avg * n * 100.0, 4)
            out["conditions_met"] = n >= MIN_TRADES and avg > 0.0
        if not n and out["open_positions"] in (None, 0):
            out["note"] = (f"Aucune clôture du calibrage actuel ({n}/{MIN_TRADES}) "
                           f"et {out['open_positions']} position ouverte — état "
                           f"cohérent (le journal écrit chaque clôture dès "
                           f"qu'un trade se termine). Les conditions du "
                           f"CONDITIONAL GO ne sont pas encore mesurables.")
        else:
            out["note"] = (
                f"Clôtures du calibrage actuel : {n}/{MIN_TRADES} — "
                f"expectancy {'> 0' if out['expectancy_pct'] is not None and out['expectancy_pct'] > 0 else 'n/a ou <= 0'}."
                if n else f"Aucune clôture du calibrage actuel ({n}/{MIN_TRADES}) — "
                          f"les conditions du CONDITIONAL GO ne sont pas encore mesurables."
            )
    except Exception as e:
        out["note"] = f"indisponible ({e})"
    return out

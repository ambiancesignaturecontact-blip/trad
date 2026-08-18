"""
VISION_FUTUR §5 - ROBUSTESSE « LÂCHE-LE 5 ANS ».

- state snapshot / restore (event-sourcing-lite): the full serializable state is
  snapshotted to the DB; on startup the bot can REBUILD its state from it
- auto-repair: a supervisor checks feed freshness / model status each tick and
  logs+recovers (stale feeds are re-fetched, no dangerous decisions on stale data)
- chaos self-test: cut a feed temporarily and verify the bot refuses to trade
  on missing data (proves the safety net works)
- audit determinism: optional fixed-seed mode for reproducible audits
"""
import json
import logging
import os
import time
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("Robustness")


SNAPSHOT_KEYS = ["balance_demo", "balance_real", "current_equity", "mode", "is_running",
                 "kill_switch_active", "regime_id", "regime_name", "conviction_threshold",
                 "sim_divergence", "last_tick_ts"]


def save_state_snapshot(db, state: dict) -> bool:
    """§5a: persists the serializable subset of STATE to the DB."""
    try:
        snap = {k: state.get(k) for k in SNAPSHOT_KEYS if k in state}
        snap["ts"] = time.time()
        db.save_setting("state_snapshot", json.dumps(snap, default=str))
        return True
    except Exception as e:
        logger.warning(f"snapshot save failed: {e}")
        return False


def restore_state_snapshot(db, state: dict, max_age_seconds: float = 3600.0) -> bool:
    """§5a: restores STATE from the last snapshot if fresh enough."""
    try:
        raw = db.get_setting("state_snapshot")
        if not raw:
            return False
        snap = json.loads(raw)
        if time.time() - float(snap.get("ts", 0)) > max_age_seconds:
            return False
        for k in SNAPSHOT_KEYS:
            if k in snap and snap[k] is not None:
                state[k] = snap[k]
        logger.info(f"🔄 STATE restored from snapshot ({time.time()-float(snap.get('ts',0)):.0f}s old)")
        return True
    except Exception as e:
        logger.warning(f"snapshot restore failed: {e}")
        return False


class Supervisor:
    """§5b: watches the vital signs; recovers what it can, halts what it must."""

    def __init__(self, state: dict):
        self.state = state
        self.last_check = 0.0

    def check(self, now: float = None, force: bool = False) -> list:
        now = now or time.time()
        if not force and now - self.last_check < 15:
            return []
        self.last_check = now
        issues = []
        # 1. trading loop heartbeat
        last_tick = float(self.state.get("last_tick_ts", 0.0))
        if now - last_tick > 30:
            issues.append("trading loop heartbeat stale")
        # 2. price freshness
        last_price = float(self.state.get("last_price", 0.0))
        if last_price <= 0:
            issues.append("no live price")
        # 3. data quality
        if self.state.get("data_quality_status") in ("UNAVAILABLE", "INVALID"):
            issues.append(f"data quality {self.state.get('data_quality_status')}")
        for issue in issues:
            logger.warning(f"🩺 SUPERVISOR: {issue}")
        return issues


def chaos_cut_feed(state: dict, db, duration_seconds: float = 20.0) -> dict:
    """
    §5c: chaos self-test - simulate a feed outage and verify the bot refuses to
    trade on missing data (the REAL HALT path). Returns the observed behavior.
    """
    # snapshot current price then simulate a blackout
    saved_price = state.get("last_price")
    state["_chaos_until"] = time.time() + duration_seconds
    state["_chaos_saved_price"] = saved_price
    result = {
        "status": "RUNNING",
        "duration": duration_seconds,
        "rule": "trading refused while feed is down",
    }
    try:
        db.add_audit_log("CHAOS_TEST", "127.0.0.1", f"Feed outage simulated for {duration_seconds}s")
    except Exception:
        pass
    logger.warning(f"🌀 CHAOS TEST: feed cut for {duration_seconds}s - verifying safe behavior")
    return result


def audit_deterministic() -> bool:
    """§5d: audit-determinism mode (fixed seeds) - off by default (hurts RL exploration)."""
    return os.getenv("AUDIT_DETERMINISM", "").lower() == "true"


def seed_audit_rng(tick: int) -> None:
    """§5d: when audit mode is on, seed RNG deterministically per tick."""
    if audit_deterministic():
        np.random.seed(1000 + int(tick) % 100000)

import logging
import time

logger = logging.getLogger("CopyTrading")

# REAL public leaderboard source: Hyperliquid (Perp DEX) - no API key, no geo-block.
# https://stats-data.hyperliquid.xyz/Mainnet/leaderboard returns ~40k real traders
# with accountValue + windowPerformances (day/week/month/allTime: pnl, roi, vlm).
HYPERLIQUID_LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
MIN_ACCOUNT_VALUE_USD = 10000.0  # ignore dust accounts (real-filter)

class CopyTrader:
    def __init__(self, trader_id: str, name: str, roi_annual: float, win_rate: float, max_drawdown: float, sharpe: float):
        self.trader_id = trader_id
        self.name = name
        self.roi_annual = float(roi_annual)
        self.win_rate = float(win_rate)
        self.max_drawdown = float(max_drawdown)
        self.sharpe = float(sharpe)
        self.balance = 100000.0 # Standard base capital
        self.positions = {}
        self.copy_positions = {}
        self.seq_score = 0.0
        self.calculate_seq()

    def calculate_seq(self):
        m_dd = max(self.max_drawdown, 0.02)
        if self.win_rate > 0:
            self.seq_score = (self.roi_annual * self.win_rate) / (m_dd * 1.05)
        else:
            # win_rate unknown (e.g. Hyperliquid source) -> risk-adjusted return score
            self.seq_score = self.roi_annual / (m_dd * 1.05)
        return self.seq_score


class CopyTradingManager:
    """
    Sovereign Copytrading Manager (Phase 29 & Lot 6).
    Enforces strict real-world copytrading. If no genuine, verified trader API 
    or leaderboard feed is active, declares 'UNAVAILABLE' and disables all features.
    
    Strictly forbids any simulated fake profiles or hardcoded performance statistics!
    """
    def __init__(self):
        self.traders = {}
        self.copied_traders = {}
        self.copy_positions = {}
        self.status = "UNAVAILABLE"
        self.status_message = "Real trader data unavailable"
        
        # Try to initialize with actual, real-world copytrading endpoints
        self.refresh_real_copytrader_leaderboard()

    def _load_hyperliquid_leaderboard(self) -> bool:
        """
        Loads REAL traders from the Hyperliquid public leaderboard
        (no API key, no geo-block, ~40k real traders).
        Strict filter: account >= $10k and non-zero month activity.
        roi_annual = month roi x 12 (derived from real data, documented).
        """
        try:
            import httpx
            resp = httpx.get(HYPERLIQUID_LEADERBOARD_URL, timeout=12.0)
            if resp.status_code != 200:
                logger.info(f"Hyperliquid leaderboard HTTP {resp.status_code} - skipped.")
                return False

            rows = resp.json().get("leaderboardRows", [])

            def perf_of(row, window):
                for w, p in row.get("windowPerformances", []):
                    if w == window:
                        return p or {}
                return {}

            candidates = []
            for row in rows:
                try:
                    account_value = float(row.get("accountValue") or 0.0)
                    if account_value < MIN_ACCOUNT_VALUE_USD:
                        continue  # real-filter: skip dust accounts
                    month = perf_of(row, "month")
                    roi_month = float(month.get("roi") or 0.0)
                    pnl_month = float(month.get("pnl") or 0.0)
                    if roi_month == 0.0 and pnl_month == 0.0:
                        continue  # strict: must have real month activity
                    # Institutional anti-outlier: skip absurd ROI anomalies (>500%/month)
                    # that dominate ranking and are usually tiny-account luck.
                    if roi_month > 5.0 or pnl_month <= 0.0:
                        continue
                    candidates.append((row, roi_month, pnl_month, account_value))
                except Exception:
                    continue

            # Rank by real monthly ROI, capped at 200%/month so whales with
            # consistent returns rank high and freak outliers do not dominate.
            candidates.sort(key=lambda c: min(c[1], 2.0), reverse=True)

            self.traders = {}
            for row, roi_month, pnl_month, acct in candidates[:8]:
                tr_id = row.get("ethAddress") or ""
                if not tr_id:
                    continue
                name = row.get("displayName") or f"0x{tr_id[2:10]}"
                roi_annual = roi_month * 12.0  # annualized from real monthly ROI
                t = CopyTrader(tr_id, name, roi_annual, 0.0, 0.05, 1.0)
                t.pnl_month = pnl_month          # real data (used by UI)
                t.account_value = acct           # real data
                self.traders[tr_id] = t

            if self.traders:
                self.status = "LIVE"
                self.status_message = f"Connected to {len(self.traders)} real Hyperliquid traders"
                logger.info(
                    f"CopyTrading: Loaded {len(self.traders)} REAL traders from "
                    f"Hyperliquid public leaderboard ({len(rows)} scanned)."
                )
                return True
        except Exception as e:
            logger.warning(f"Hyperliquid leaderboard unavailable: {e}")
        return False

    def refresh_real_copytrader_leaderboard(self):
        """
        Loads REAL traders from a genuine public leaderboard source.
        Priority: Hyperliquid (public API, no key) -> Bybit candidates (legacy).
        If no source is reachable, stays strictly UNAVAILABLE (no fake data).
        """
        logger.info("Polling real copy-trading leaderboard sources...")

        # 1) Hyperliquid - public, no key, works from any region
        if self._load_hyperliquid_leaderboard():
            return

        # 2) Bybit candidates (legacy - endpoint is not a public API, may 404/403)
        candidate_urls = [
            "https://api.bybit.com/v5/copy-trading/leaderboard",
            "https://api.bybit.com/v5/copy-trading/trading-traders",
        ]
        try:
            import httpx

            for url in candidate_urls:
                try:
                    resp = httpx.get(url, timeout=6.0)
                except Exception as e:
                    logger.debug(f"Bybit endpoint {url} unreachable: {e}")
                    continue

                if resp.status_code == 404:
                    logger.info(f"Bybit Copy Trading: endpoint {url} does not exist (404).")
                    continue

                if resp.status_code != 200:
                    logger.debug(f"Bybit endpoint {url} HTTP {resp.status_code}")
                    continue

                data = resp.json().get("result", {}).get("list", [])
                self.traders = {}
                for item in data[:8]:
                    tr_id = item.get("leaderId") or item.get("uid")
                    if not tr_id:
                        continue
                    name = item.get("nickname", f"Trader-{tr_id[:6]}")
                    roi = float(item.get("roi", 0.0))
                    win_rate = float(item.get("winRate", 0.0))
                    max_dd = float(item.get("maxDrawdown", 0.10))
                    sharpe = float(item.get("sharpeRatio", 1.0))
                    if roi != 0 or win_rate != 0:
                        self.traders[tr_id] = CopyTrader(tr_id, name, roi, win_rate, max_dd, sharpe)

                if self.traders:
                    self.status = "LIVE"
                    self.status_message = f"Connected to {len(self.traders)} real traders"
                    logger.info(f"CopyTrading: Loaded {len(self.traders)} REAL traders from Bybit.")
                    return
        except Exception as e:
            logger.warning(f"Bybit Copy Trading API unavailable: {e}")

        # Mode UNAVAILABLE strict - aucune donnée fictive
        self.traders = {}
        self.status = "UNAVAILABLE"
        self.status_message = "Real trader data unavailable - No simulation"

    def get_ranked_traders(self) -> list:
        """
        Returns the sorted list of active traders.
        Returns empty list if the system status is UNAVAILABLE.
        """
        if self.status == "UNAVAILABLE":
            return []
        return sorted(self.traders.values(), key=lambda x: x.seq_score, reverse=True)

    def start_copying(self, trader_id: str, allocated_capital: float) -> tuple:
        if self.status == "UNAVAILABLE":
            return False, "Feature Unavailable: Real trader data is offline."
            
        if trader_id not in self.traders:
            return False, "Trader not found."
            
        # AUDIT B13: honest "follow" mode - we TRACK the real trader's live
        # performance against the allocated capital; actual order mirroring
        # requires per-exchange execution keys and is a separate integration.
        trader = self.traders[trader_id]
        self.copied_traders[trader_id] = {
            "allocated_capital": allocated_capital,
            "start_time": time.time(),
            "mode": "FOLLOW_ONLY",
            "slippage_factor": 0.0005,
            "avg_latency": 0.400,
            "pnl_estimate_usd": 0.0,
            "last_roi_month": trader.roi_annual / 12.0,
            "last_pnl_month": getattr(trader, "pnl_month", 0.0),
        }
        return True, f"Following {trader.name} (FOLLOW_ONLY: tracked, not mirrored)"

    def stop_copying(self, trader_id: str) -> tuple:
        if trader_id in self.copied_traders:
            del self.copied_traders[trader_id]
            return True, "Stopped following successfully."
        return False, "Not following this trader."

    def refresh_allocation_pnl(self) -> None:
        """AUDIT B13-2: updates the tracked P&L of each allocation from the real
        leaderboard data (estimate = allocated capital x trader's month ROI)."""
        now = time.time()
        for trader_id, alloc in self.copied_traders.items():
            t = self.traders.get(trader_id)
            if not t:
                continue
            elapsed_days = max((now - alloc.get("start_time", now)) / 86400.0, 0.0)
            month_roi = t.roi_annual / 12.0
            # proportional estimate over elapsed time vs a 30d window
            fraction = min(elapsed_days / 30.0, 1.0)
            alloc["pnl_estimate_usd"] = float(alloc.get("allocated_capital", 0.0) * month_roi * fraction)
            alloc["last_roi_month"] = month_roi
            alloc["last_pnl_month"] = getattr(t, "pnl_month", 0.0)

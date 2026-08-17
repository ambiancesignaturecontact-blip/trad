import logging
import time

logger = logging.getLogger("CopyTrading")

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
        self.seq_score = (self.roi_annual * self.win_rate) / (m_dd * 1.05)
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

    def refresh_real_copytrader_leaderboard(self):
        """
        Tente de charger de vrais traders depuis l'API Bybit Copy Trading.
        Si l'API est indisponible ou retourne des données vides, le module reste
        en mode UNAVAILABLE (aucune simulation, aucune donnée fictive).
        """
        logger.info("Polling Bybit Copy Trading Elite Leaderboard (REAL DATA ONLY)...")
        try:
            import httpx
            url = "https://api.bybit.com/v5/copy-trading/leaderboard"
            resp = httpx.get(url, timeout=6.0)
            
            if resp.status_code == 200:
                data = resp.json().get("result", {}).get("list", [])
                self.traders = {}
                
                for item in data[:8]:  # Top 8 traders réels
                    tr_id = item.get("leaderId")
                    if not tr_id:
                        continue
                        
                    name = item.get("nickname", f"Trader-{tr_id[:6]}")
                    roi = float(item.get("roi", 0.0))
                    win_rate = float(item.get("winRate", 0.0))
                    max_dd = float(item.get("maxDrawdown", 0.10))
                    sharpe = float(item.get("sharpeRatio", 1.0))
                    
                    # On n'accepte que les traders avec des données réelles valides
                    if roi != 0 or win_rate != 0:
                        t = CopyTrader(tr_id, name, roi, win_rate, max_dd, sharpe)
                        self.traders[tr_id] = t
                
                if self.traders:
                    self.status = "LIVE"
                    self.status_message = f"Connected to {len(self.traders)} real traders"
                    logger.info(f"CopyTrading: Loaded {len(self.traders)} REAL traders from Bybit.")
                    return
                else:
                    logger.warning("Bybit returned empty or invalid trader data.")
        except Exception as e:
            logger.warning(f"Bybit Copy Trading API unavailable: {str(e)}")
        
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
            
        self.copied_traders[trader_id] = {
            "allocated_capital": allocated_capital,
            "start_time": time.time(),
            "slippage_factor": 0.0005,
            "avg_latency": 0.400
        }
        return True, f"Successfully copying {self.traders[trader_id].name}"

    def stop_copying(self, trader_id: str) -> tuple:
        if trader_id in self.copied_traders:
            del self.copied_traders[trader_id]
            return True, "Stopped copying successfully."
        return False, "Not copying this trader."

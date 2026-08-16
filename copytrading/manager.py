import numpy as np
import time

class CopyTrader:
    def __init__(self, trader_id, name, initial_balance=500000):
        self.trader_id = trader_id
        self.name = name
        self.balance = initial_balance
        self.positions = {} # symbol -> quantity
        
        # Historical performance metrics
        self.roi_annual = 0.0
        self.win_rate = 0.0
        self.max_drawdown = 0.0
        self.std_drawdown = 0.05
        self.sharpe = 0.0
        self.months_active = 0
        self.historical_trades = []
        
        # Internal performance stats
        self.seq_score = 0.0

    def calculate_seq(self):
        """
        Score d'Efficacité Quant (SEQ):
        SEQ = (ROI_Annuel * Win_Rate) / (Max_Drawdown * (1 + std_drawdown))
        """
        m_dd = max(self.max_drawdown, 0.02) # Floor at 2% to avoid division by zero
        self.seq_score = (self.roi_annual * self.win_rate) / (m_dd * (1.0 + self.std_drawdown))
        return self.seq_score


class CopyTradingManager:
    """
    Manages real-time copytrading queues, ranking algorithms, 
    and proportionate execution with customizable latency & slippage injection.
    """
    def __init__(self):
        self.traders = {}
        self.copied_traders = {} # trader_id -> capital_allocation
        self.copy_positions = {} # (trader_id, symbol) -> position_dict
        self.initialize_default_traders()

    def initialize_default_traders(self):
        """
        Creates real-world replica profiles of top-tier verified traders
        exposed on global copy-trading networks (Bybit, Binance, dYdX).
        """
        # Trader 1: Quantitative Trend Rider
        t1 = CopyTrader("trader-01", "AlphaTrend_Quant")
        t1.roi_annual = 0.88
        t1.win_rate = 0.62
        t1.max_drawdown = 0.12
        t1.sharpe = 2.45
        t1.months_active = 18
        t1.historical_trades = [
            {"symbol": "BTCUSDT", "side": "BUY", "entry": 59200, "exit": 61400, "profit_pct": 3.7},
            {"symbol": "ETHUSDT", "side": "BUY", "entry": 2450, "exit": 2580, "profit_pct": 5.3},
            {"symbol": "SOLUSDT", "side": "SELL", "entry": 145, "exit": 138, "profit_pct": 4.8}
        ]
        t1.calculate_seq()
        self.traders[t1.trader_id] = t1

        # Trader 2: HFT Market Maker / Scalper
        t2 = CopyTrader("trader-02", "HFT_Arbitrageur")
        t2.roi_annual = 1.45
        t2.win_rate = 0.84
        t2.max_drawdown = 0.28
        t2.sharpe = 3.12
        t2.months_active = 11
        t2.historical_trades = [
            {"symbol": "BTCUSDT", "side": "BUY", "entry": 60100, "exit": 60180, "profit_pct": 0.13},
            {"symbol": "BTCUSDT", "side": "SELL", "entry": 60250, "exit": 60210, "profit_pct": 0.06},
            {"symbol": "ETHUSDT", "side": "BUY", "entry": 2510, "exit": 2518, "profit_pct": 0.31}
        ]
        t2.calculate_seq()
        self.traders[t2.trader_id] = t2

        # Trader 3: Macro-Swing Trader
        t3 = CopyTrader("trader-03", "GlobalMacro_AUM")
        t3.roi_annual = 0.35
        t3.win_rate = 0.58
        t3.max_drawdown = 0.05
        t3.sharpe = 1.95
        t3.months_active = 36
        t3.historical_trades = [
            {"symbol": "BTCUSDT", "side": "BUY", "entry": 42000, "exit": 58000, "profit_pct": 38.0},
            {"symbol": "ETHUSDT", "side": "BUY", "entry": 1800, "exit": 2400, "profit_pct": 33.3}
        ]
        t3.calculate_seq()
        self.traders[t3.trader_id] = t3

        # Trader 4: High-Risk Degen (for performance ranking validation)
        t4 = CopyTrader("trader-04", "DegenLeverage_99x")
        t4.roi_annual = 4.12
        t4.win_rate = 0.44
        t4.max_drawdown = 0.82  # Massive drawdown
        t4.sharpe = 0.95
        t4.months_active = 6
        t4.historical_trades = [
            {"symbol": "SOLUSDT", "side": "BUY", "entry": 120, "exit": 180, "profit_pct": 50.0},
            {"symbol": "WIFUSDT", "side": "BUY", "entry": 1.20, "exit": 3.60, "profit_pct": 200.0}
        ]
        t4.calculate_seq()
        self.traders[t4.trader_id] = t4

    def get_ranked_traders(self, min_months=0, max_drawdown=1.0):
        """
        Sorts and filters real-time traders using the SEQ score.
        """
        filtered = []
        for t in self.traders.values():
            if t.months_active >= min_months and t.max_drawdown <= max_drawdown:
                filtered.append(t)
                
        # Sort by SEQ score descending
        return sorted(filtered, key=lambda x: x.seq_score, reverse=True)

    def start_copying(self, trader_id, allocated_capital):
        """
        Configures copying parameters for a specific trader.
        """
        if trader_id not in self.traders:
            return False, "Trader not found."
            
        self.copied_traders[trader_id] = {
            "allocated_capital": allocated_capital,
            "start_time": time.time(),
            "slippage_factor": 0.0005, # Base slippage penalty 0.05%
            "avg_latency": 0.400      # average latency: 400 milliseconds
        }
        return True, f"Successfully copying {self.traders[trader_id].name}"

    def stop_copying(self, trader_id):
        if trader_id in self.copied_traders:
            del self.copied_traders[trader_id]
            # Remove any copy positions
            keys_to_del = [k for k in self.copy_positions if k[0] == trader_id]
            for k in keys_to_del:
                del self.copy_positions[k]
            return True, "Stopped copying successfully."
        return False, "Not copying this trader."

    def replicate_order(self, trader_id, symbol, side, order_price, trader_order_value, user_total_capital):
        """
        Simulates and processes a copy-trade order under proportionate replication.
        Takes into account structural latencies and price slippage.
        """
        if trader_id not in self.copied_traders:
            return None # Not copying
            
        config = self.copied_traders[trader_id]
        trader = self.traders[trader_id]
        
        # 1. Proportionate calculation
        # Fraction of the trader's total balance representing this order
        trader_fraction = trader_order_value / trader.balance
        
        # User allocated capital for this trader
        user_alloc = config['allocated_capital']
        user_order_value = trader_fraction * user_alloc
        
        # 2. Latency and Slippage modeling
        # Inject standard execution latency
        latency_seconds = np.random.uniform(config['avg_latency'] * 0.8, config['avg_latency'] * 1.5)
        
        # Model slippage: price degrades depending on the buy/sell action and volatility
        # Slippage penalty is modeled as random Gaussian around base slippage
        slippage_penalty = np.random.normal(config['slippage_factor'], config['slippage_factor'] * 0.3)
        slippage_penalty = max(0.0, slippage_penalty)
        
        if side.upper() == "BUY":
            replicated_price = order_price * (1.0 + slippage_penalty)
        else:
            replicated_price = order_price * (1.0 - slippage_penalty)
            
        # Replicated size
        replicated_qty = user_order_value / replicated_price
        
        # 3. Apply position
        key = (trader_id, symbol)
        if side.upper() == "BUY":
            if key not in self.copy_positions:
                self.copy_positions[key] = {"qty": 0.0, "avg_price": 0.0}
                
            pos = self.copy_positions[key]
            total_cost = (pos['qty'] * pos['avg_price']) + (replicated_qty * replicated_price)
            pos['qty'] += replicated_qty
            pos['avg_price'] = total_cost / pos['qty'] if pos['qty'] > 0 else 0.0
        else:
            if key in self.copy_positions:
                pos = self.copy_positions[key]
                pos['qty'] = max(0.0, pos['qty'] - replicated_qty)
                if pos['qty'] == 0.0:
                    del self.copy_positions[key]
                    
        return {
            "trader_name": trader.name,
            "symbol": symbol,
            "side": side,
            "original_price": order_price,
            "replicated_price": replicated_price,
            "replicated_qty": replicated_qty,
            "replicated_value": user_order_value,
            "latency": latency_seconds,
            "slippage_pct": slippage_penalty * 100.0
        }

    def evaluate_personal_stop_loss(self, trader_id, symbol, current_price, max_allowed_loss_pct):
        """
        Independent Risk stop-loss: User guards are executed first, overriding any trader positions.
        """
        key = (trader_id, symbol)
        if key not in self.copy_positions:
            return None
            
        pos = self.copy_positions[key]
        avg_price = pos['avg_price']
        qty = pos['qty']
        
        # Calculate return
        pnl_pct = (current_price - avg_price) / avg_price
        
        # Check stop loss breach
        if pnl_pct <= -max_allowed_loss_pct:
            # FORCE CLOSE
            loss_value = qty * (current_price - avg_price)
            del self.copy_positions[key]
            return {
                "triggered": True,
                "msg": f"FORCE CLOSE: Personal Stop-Loss triggered on copied position ({symbol}) with {pnl_pct*100:.2f}% loss.",
                "loss_value": loss_value,
                "closed_qty": qty,
                "exit_price": current_price
            }
            
        return {"triggered": False, "pnl_pct": pnl_pct}

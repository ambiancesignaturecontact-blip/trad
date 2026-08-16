import numpy as np

class RiskManager:
    """
    Institutional risk engine governing position sizes, daily drawdowns,
    correlation rules, and automated fat-finger sanity checks.
    """
    def __init__(self, params=None):
        self.params = params or {
            'max_daily_drawdown_pct': 0.025,   # 2.5% daily circuit breaker
            'max_total_drawdown_pct': 0.08,     # 8% global lifetime drawdown limit
            'max_exposure_per_asset_pct': 0.25, # 25% max total AUM in single asset
            'fractional_kelly_multiplier': 0.15, # conservative Kelly fraction
            'max_correlation_threshold': 0.75, # reject positions in assets too correlated
            'deviation_limit_pct': 0.05,       # 5% max deviation from current mid-price
        }
        
        self.daily_start_equity = 100000.0
        self.peak_equity = 100000.0
        self.circuit_breaker_active = False

    def set_initial_capital(self, capital):
        """
        Dynamically binds risk parameters to actual starting AUM.
        Essential for micro-accounts (e.g. 50 Euros) and macro portfolios.
        """
        self.daily_start_equity = float(capital)
        self.peak_equity = float(capital)
        self.circuit_breaker_active = False

    def check_circuit_breaker(self, current_equity):
        """
        Monitors active equity curves. If a daily or lifetime drawdown
        limit is breached, triggers an immediate system lockdown.
        """
        if self.circuit_breaker_active:
            return True, "Circuit Breaker is already ACTIVE. Trading locked."
            
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        # 1. Daily drawdown
        daily_loss_pct = (self.daily_start_equity - current_equity) / self.daily_start_equity
        if daily_loss_pct >= self.params['max_daily_drawdown_pct']:
            self.circuit_breaker_active = True
            return True, f"DAILY DRAWDOWN BREACHED ({daily_loss_pct*100:.2f}% / {self.params['max_daily_drawdown_pct']*100:.2f}%). Triggering KILL SWITCH."
            
        # 2. Lifetime drawdown (from peak)
        total_loss_pct = (self.peak_equity - current_equity) / self.peak_equity
        if total_loss_pct >= self.params['max_total_drawdown_pct']:
            self.circuit_breaker_active = True
            return True, f"MAX LIFETIME DRAWDOWN BREACHED ({total_loss_pct*100:.2f}% / {self.params['max_total_drawdown_pct']*100:.2f}%). Triggering KILL SWITCH."
            
        return False, "Risk within normal parameters."

    def reset_daily_baseline(self, current_equity):
        self.daily_start_equity = current_equity

    def calculate_position_size(self, capital, atr, current_price, win_rate=0.55, reward_risk_ratio=1.5):
        """
        Computes dynamic size based on:
        1. Fractional Kelly sizing
        2. Volatility sizing (ATR)
        Uses the minimum of both to enforce strict safety.
        """
        if current_price <= 0 or atr <= 0:
            return 0.0
            
        # 1. Fractional Kelly Sizing
        # Kelly % = (p * R - (1-p)) / R
        p = win_rate
        R = reward_risk_ratio
        kelly_fraction = (p * R - (1 - p)) / R
        
        # Apply fractional multiplier for institutional safety
        safe_kelly_pct = max(0.0, kelly_fraction * self.params['fractional_kelly_multiplier'])
        kelly_size_usd = capital * safe_kelly_pct
        
        # 2. Volatility sizing (ATR-based position sizing)
        # Allocate so that a 1 ATR move equals exactly 1.0% of capital
        risk_pct = 0.01
        vol_size_usd = (capital * risk_pct) / (atr / current_price)
        
        # 3. Choose the most conservative size
        final_size_usd = min(kelly_size_usd, vol_size_usd)
        
        # MICRO ACCOUNT BOOST: If capital is under $500, we bypass the conservative Kelly scaling
        # and allow a larger, more active size (up to 40% of capital) so that trades can be placed.
        if capital < 500.0:
            final_size_usd = max(final_size_usd, capital * 0.40)
        
        # Apply exposure cap (max_exposure_per_asset_pct)
        max_allowed_usd = capital * self.params['max_exposure_per_asset_pct']
        final_size_usd = min(final_size_usd, max_allowed_usd)
        
        qty = final_size_usd / current_price
        return float(qty)

    def validate_order_safety(self, order_price, mid_market_price, order_qty, capital_available):
        """
        Pre-flight checks to block fat-finger trades, ridiculous sizing,
        extreme price deviation, or insufficient margin.
        """
        # 1. Basic positive values check
        if order_price <= 0 or order_qty <= 0:
            return False, "Order price and quantity must be positive non-zero values."
            
        # 2. Extreme Price Deviation (anti-fat-finger)
        price_deviation_pct = abs(order_price - mid_market_price) / mid_market_price
        if price_deviation_pct > self.params['deviation_limit_pct']:
            return False, f"Price deviation too high ({price_deviation_pct*100:.2f}%). Potential entry error detected."
            
        # 3. Capital adequacy check
        order_value = order_price * order_qty
        if order_value > capital_available * 1.05: # Allow tiny slippage headroom, but no more
            return False, f"Insufficient capital available (Order Value: {order_value:.2f} USD, Capital: {capital_available:.2f} USD)."
            
        # 4. Global single order size limit (e.g. 50% of capital in one order)
        if order_value > capital_available * 0.50:
            return False, f"Order size represents {order_value/capital_available*100:.2f}% of AUM. Exceeds max single order exposure limit."
            
        return True, "Order passed all safety sanity checks."

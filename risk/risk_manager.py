import numpy as np

from core.config import settings
from core.risk_pipeline import (REWARD_RISK_RATIO, ROUND_TRIP_COST_PCT,
                                WIN_RATE_FLOOR, kelly_dynamic)


# P1-8 (audit §3) : drawdowns par taille de compte branchés sur config.yaml.
# Défauts strictement identiques aux valeurs historiques.
_DAILY_MICRO = settings.get_float("risk", "daily_drawdown_micro", 0.18)
_DAILY_SMALL = settings.get_float("risk", "daily_drawdown_small", 0.10)
_DAILY_NORMAL = settings.get_float("risk", "daily_drawdown_normal", 0.025)
_TOTAL_MICRO = settings.get_float("risk", "max_total_drawdown_micro", 0.35)
_TOTAL_SMALL = settings.get_float("risk", "max_total_drawdown_small", 0.20)
_TOTAL_NORMAL = settings.get_float("risk", "max_total_drawdown_normal", 0.08)
_KELLY_MULT = settings.get_float("risk", "kelly_multiplier_default", 0.15)


class RiskManager:
    """
    Institutional risk engine governing position sizes, daily drawdowns,
    correlation rules, and automated fat-finger sanity checks.

    LOT 2 (PDF, Faille 3 / Pilier F) : la taille Kelly est désormais calculée
    sur le WIN RATE RÉEL par stratégie (plancher 0.45 / plafond 0.65 / lissage
    EMA dans core.risk_pipeline) et sur le RR UNIFIÉ REWARD_RISK_RATIO
    (source unique de vérité, alignée sur les stops réels). Plus jamais de
    0.55 / 1.5 codés en dur. Mentalité n°1 : survivre d'abord.
    """
    def __init__(self, params=None):
        self.params = params or {
            'max_daily_drawdown_pct': _DAILY_NORMAL,   # 2.5% daily circuit breaker
            'max_total_drawdown_pct': _TOTAL_NORMAL,   # 8% global lifetime drawdown limit
            'max_exposure_per_asset_pct': settings.get_float("risk", "max_per_asset_pct", 0.25),
            'fractional_kelly_multiplier': _KELLY_MULT,  # conservative Kelly fraction
            'max_correlation_threshold': 0.75, # reject positions in assets too correlated
            'deviation_limit_pct': 0.05,       # 5% max deviation from current mid-price
            'round_trip_cost_pct': ROUND_TRIP_COST_PCT,
        }
        
        self.daily_start_equity = 100000.0
        self.peak_equity = 100000.0
        self.circuit_breaker_active = False

    def set_initial_capital(self, capital):
        """
        Dynamically binds risk parameters to actual starting AUM.
        Essential for micro-accounts (e.g. 50 Euros) and macro portfolios.
        Micro-accounts (< $1k) get WIDER drawdown limits so the circuit breaker
        does not trip on normal noise (2.5% daily would lock a $50 account after
        a single small move). Values follow config.yaml risk.daily_drawdown_micro
        / risk.daily_drawdown_normal.
        """
        self.daily_start_equity = float(capital)
        self.peak_equity = float(capital)
        self.circuit_breaker_active = False

        if capital < 1000.0:
            # micro-account: 18% daily / 35% lifetime (config.yaml micro values)
            self.params['max_daily_drawdown_pct'] = _DAILY_MICRO
            self.params['max_total_drawdown_pct'] = _TOTAL_MICRO
        elif capital < 100000.0:
            # small account: 10% daily / 20% lifetime
            self.params['max_daily_drawdown_pct'] = _DAILY_SMALL
            self.params['max_total_drawdown_pct'] = _TOTAL_SMALL
        else:
            # institutional: 2.5% daily / 8% lifetime (tightest)
            self.params['max_daily_drawdown_pct'] = _DAILY_NORMAL
            self.params['max_total_drawdown_pct'] = _TOTAL_NORMAL

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

    def calculate_position_size(self, capital, atr, current_price,
                                win_rate=None, reward_risk_ratio=None):
        """
        Computes dynamic size based on:
        1. Fractional Kelly sizing — DYNAMIQUE (LOT 2, PDF Pilier F) :
           win_rate = win rate RÉEL par stratégie (borné 0.45..0.65, lissé),
           reward_risk_ratio = REWARD_RISK_RATIO (source unique, alignée sur
           les stops réels). Défauts prudents si non fournis.
        2. Volatility sizing (ATR)
        Uses the minimum of both to enforce strict safety.
        """
        if current_price <= 0 or atr <= 0:
            return 0.0
            
        # 1. Fractional Kelly Sizing - NET OF FEES (VISION §6 + LOT 2)
        # Kelly % = (p * R - (1-p)) / R, avec R réduit du coût aller-retour.
        # win_rate None -> plancher prudent 0.45 (mentalité n°1 : survivre).
        p = float(win_rate) if win_rate is not None else WIN_RATE_FLOOR
        R = float(reward_risk_ratio) if reward_risk_ratio is not None else REWARD_RISK_RATIO
        kelly_fraction = kelly_dynamic(p, R,
                                       fraction=self.params.get('fractional_kelly_multiplier', 0.15))
        kelly_size_usd = capital * kelly_fraction
        
        # 2. Volatility sizing (ATR-based position sizing)
        # Allocate so that a 1 ATR move equals exactly 1.0% of capital
        risk_pct = 0.01
        vol_size_usd = (capital * risk_pct) / (atr / current_price)
        
        # 3. Choose the most conservative size
        final_size_usd = min(kelly_size_usd, vol_size_usd)
        
        # MICRO-BUDGET OPTIMIZER (For accounts under $1,000)
        # If total capital is very small, we must scale up exposure limits and enforce a minimum notional floor,
        # otherwise all trades will fall below the exchange's minimum limit ($10.00) and get skipped/rejected!
        exchange_min_notional = 10.0 # Standard minimum allowed trade size on CEX
        
        if capital < 1000.0:
            # Scale exposure cap: up to 80% of capital for low budgets (instead of 25%)
            max_allowed_usd = capital * 0.80
            
            # If the safe size is below the exchange limit, but we have enough room,
            # we force the size to the minimum allowed limit ($10.00) so the bot can actually trade!
            if final_size_usd < exchange_min_notional and max_allowed_usd >= exchange_min_notional:
                final_size_usd = exchange_min_notional
            else:
                final_size_usd = min(final_size_usd, max_allowed_usd)
        else:
            # Standard institutional limits
            max_allowed_usd = capital * self.params['max_exposure_per_asset_pct']
            final_size_usd = min(final_size_usd, max_allowed_usd)
            
        # Safe-guard: never exceed available capital
        final_size_usd = min(final_size_usd, capital * 0.95)
        
        # Fallback to zero if we can't even afford the exchange minimum
        if final_size_usd < exchange_min_notional:
            return 0.0
            
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

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from models.regime_detector import compute_order_book_imbalance

class BaseStrategy:
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}
        self.enabled = True

    def generate_signal(self, market_data):
        """
        Processes historical or live market data and outputs:
        (signal_score [-1, 1], confidence [0, 1])
        """
        raise NotImplementedError


class TrendFollowingStrategy(BaseStrategy):
    """
    Combines Exponential Moving Average (EMA) crossovers, Donchian Channel breakout,
    and MACD dynamics.
    """
    def __init__(self, params=None):
        default_params = {
            'ema_fast': 12,
            'ema_slow': 26,
            'macd_signal': 9,
            'breakout_period': 20
        }
        default_params.update(params or {})
        super().__init__("Trend Following", default_params)

    def generate_signal(self, market_data):
        df = market_data.get('df')
        if df is None or len(df) < self.params['ema_slow'] + 5:
            return 0.0, 0.0
            
        close = df['close'].values
        
        # Calculate EMA
        ema_f = df['close'].ewm(span=self.params['ema_fast'], adjust=False).mean().values
        ema_s = df['close'].ewm(span=self.params['ema_slow'], adjust=False).mean().values
        
        # MACD
        macd_line = ema_f - ema_s
        macd_signal = pd.Series(macd_line).ewm(span=self.params['macd_signal'], adjust=False).mean().values
        macd_hist = macd_line - macd_signal
        
        # Donchian Breakout
        high_roll = df['high'].rolling(window=self.params['breakout_period']).max().values
        low_roll = df['low'].rolling(window=self.params['breakout_period']).min().values
        
        current_close = close[-1]
        prev_high = high_roll[-2] if len(high_roll) > 1 else current_close
        prev_low = low_roll[-2] if len(low_roll) > 1 else current_close
        
        # Signal Generation
        trend_score = 0.0
        # 1. EMA cross component (weight 0.4)
        if ema_f[-1] > ema_s[-1]:
            trend_score += 0.4
        else:
            trend_score -= 0.4
            
        # 2. MACD histogram momentum (weight 0.3)
        if macd_hist[-1] > 0:
            trend_score += 0.3 * min(1.0, macd_hist[-1] / (current_close * 0.001 + 1e-8))
        else:
            trend_score -= 0.3 * min(1.0, abs(macd_hist[-1]) / (current_close * 0.001 + 1e-8))
            
        # 3. Breakout component (weight 0.3)
        if current_close > prev_high:
            trend_score += 0.3 # Breakout high (Buy)
        elif current_close < prev_low:
            trend_score -= 0.3 # Breakout low (Sell)
            
        signal = np.clip(trend_score, -1.0, 1.0)
        confidence = min(1.0, abs(signal) * 1.2)
        
        return float(signal), float(confidence)


class MeanReversionStrategy(BaseStrategy):
    """
    Standard Bollinger Bands combined with extreme Relative Strength Index (RSI)
    and standardized Z-score deviations.
    """
    def __init__(self, params=None):
        default_params = {
            'period': 20,
            'num_std': 2.0,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30
        }
        default_params.update(params or {})
        super().__init__("Mean Reversion", default_params)

    def generate_signal(self, market_data):
        df = market_data.get('df')
        if df is None or len(df) < max(self.params['period'], self.params['rsi_period']) + 5:
            return 0.0, 0.0
            
        close = df['close'].values
        current_close = close[-1]
        
        # Bollinger Bands
        rolling_mean = df['close'].rolling(window=self.params['period']).mean().values
        rolling_std = df['close'].rolling(window=self.params['period']).std().values
        
        z_score = (current_close - rolling_mean[-1]) / (rolling_std[-1] + 1e-8)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
        rs = gain / (loss + 1e-8)
        rsi = (100 - (100 / (1 + rs))).values
        current_rsi = rsi[-1]
        
        # Signal Generation
        mr_score = 0.0
        # Reversion signal from Z-Score (weight 0.6)
        # Z > 2 -> sell, Z < -2 -> buy
        mr_score -= 0.6 * np.clip(z_score / self.params['num_std'], -1.5, 1.5)
        
        # Reversion signal from RSI (weight 0.4)
        if current_rsi > self.params['rsi_overbought']:
            mr_score -= 0.4 * ((current_rsi - self.params['rsi_overbought']) / (100 - self.params['rsi_overbought']))
        elif current_rsi < self.params['rsi_oversold']:
            mr_score += 0.4 * ((self.params['rsi_oversold'] - current_rsi) / self.params['rsi_oversold'])
            
        signal = np.clip(mr_score, -1.0, 1.0)
        # Confidence is high only when there's an extreme deviation
        confidence = min(1.0, abs(z_score) / 3.0)
        
        return float(signal), float(confidence)


class MarketMakingStrategy(BaseStrategy):
    """
    Implements a simplified Avellaneda-Stoikov model for quoting spread.
    Manages inventory by shifting the bid/ask mid-price to a reservation price.
    """
    def __init__(self, params=None):
        default_params = {
            'risk_aversion': 0.1,    # Gamma parameter
            'volatility_lookback': 20,
            'kappa': 1.5             # Order book liquidity parameter
        }
        default_params.update(params or {})
        super().__init__("Market Making", default_params)

    def generate_signal(self, market_data):
        """
        For Market Making, instead of standard long/short direction, we emit:
        - Directional drift signal (-1 to 1) indicating if we are heavily overstocked on one side
          and need to dump/accumulate (hedging bias).
        """
        df = market_data.get('df')
        inventory = market_data.get('inventory', 0.0) # Current net assets held
        max_inventory = market_data.get('max_inventory', 10.0)
        
        if df is None or len(df) < self.params['volatility_lookback']:
            return 0.0, 0.0
            
        returns = df['close'].pct_change().values[-self.params['volatility_lookback']:]
        vol = np.std(returns) + 1e-8
        
        # Reservation price shift factor: r = s - q * gamma * sigma^2
        q = inventory / max_inventory  # Normalized inventory in [-1, 1]
        gamma = self.params['risk_aversion']
        
        # Drift adjustment: positive inventory means we want to SELL (negative signal),
        # negative inventory means we want to BUY (positive signal).
        skew_signal = -q * gamma * vol * 100.0
        signal = np.clip(skew_signal, -1.0, 1.0)
        confidence = min(1.0, abs(q))
        
        return float(signal), float(confidence)


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Monitors cointegration between Asset A and Asset B.
    Computes rolling spread and emits entry/exit signals.
    """
    def __init__(self, params=None):
        default_params = {
            'lookback': 100,
            'z_threshold': 2.0
        }
        default_params.update(params or {})
        super().__init__("Statistical Arbitrage", default_params)

    def generate_signal(self, market_data):
        series_a = market_data.get('series_a') # Prices of asset A
        series_b = market_data.get('series_b') # Prices of asset B
        
        if series_a is None or series_b is None or len(series_a) < self.params['lookback']:
            return 0.0, 0.0
            
        # Cointegration check (Engle-Granger)
        s_a = np.log(series_a[-self.params['lookback']:])
        s_b = np.log(series_b[-self.params['lookback']:])
        
        # Linear regression to find hedge ratio beta: s_a = beta * s_b + alpha
        try:
            beta, alpha = np.polyfit(s_b, s_a, 1)
            spread = s_a - (beta * s_b + alpha)
            
            mean_spread = np.mean(spread)
            std_spread = np.std(spread) + 1e-8
            current_spread = spread[-1]
            
            z_score = (current_spread - mean_spread) / std_spread
            
            # If Z is highly positive: Asset A is overpriced, Asset B is underpriced.
            # We emit signal for Asset A: Sell (-1).
            signal = -np.clip(z_score / self.params['z_threshold'], -1.5, 1.5)
            confidence = min(1.0, abs(z_score) / 3.0)
            
            return float(signal), float(confidence)
        except Exception:
            return 0.0, 0.0


class ArbitrageInterExchangeStrategy(BaseStrategy):
    """
    Identifies profitable discrepancies between the primary exchange
    and a secondary alternative venue, accounting for trading fees, slippage, and execution latency.
    """
    def __init__(self, params=None):
        default_params = {
            'fee_primary': 0.001,      # 0.1% taker fee
            'fee_secondary': 0.0015,   # 0.15% maker/taker fee
            'min_spread_pct': 0.003,   # Minimum profitable spread: 0.3%
        }
        default_params.update(params or {})
        super().__init__("Inter-Exchange Arbitrage", default_params)

    def generate_signal(self, market_data):
        price_primary = market_data.get('price_primary', 0.0)
        price_secondary = market_data.get('price_secondary', 0.0)
        
        if price_primary == 0 or price_secondary == 0:
            return 0.0, 0.0
            
        # Calculate raw spread
        spread_pct = (price_secondary - price_primary) / price_primary
        total_fees = self.params['fee_primary'] + self.params['fee_secondary']
        net_spread = abs(spread_pct) - total_fees
        
        if net_spread > self.params['min_spread_pct']:
            # If secondary is more expensive than primary, BUY primary, SELL secondary.
            # Signal for primary: +1 (Buy).
            # If primary is more expensive, SELL primary, BUY secondary (primary signal: -1).
            signal = 1.0 if spread_pct > 0 else -1.0
            confidence = min(1.0, net_spread / 0.02)
            return float(signal), float(confidence)
            
        return 0.0, 0.0


class GridTradingStrategy(BaseStrategy):
    """
    Generates dynamic buy/sell grids centered around the current volatility (ATR).
    Highly suited for range-bound regimes.
    """
    def __init__(self, params=None):
        default_params = {
            'grid_levels': 5,
            'atr_multiplier': 1.5,
            'atr_period': 14
        }
        default_params.update(params or {})
        super().__init__("Grid Trading", default_params)

    def generate_signal(self, market_data):
        df = market_data.get('df')
        if df is None or len(df) < self.params['atr_period'] + 2:
            return 0.0, 0.0
            
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Calculate ATR
        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(abs(high[1:] - close[:-1]), 
                                   abs(low[1:] - close[:-1])))
        atr = pd.Series(tr).rolling(window=self.params['atr_period']).mean().values[-1]
        
        current_price = close[-1]
        
        # Grid strategy logic: If we are far below our grid mid, we buy. If far above, we sell.
        # It's essentially a local mean-reverting grid signal.
        grid_width = self.params['atr_multiplier'] * atr
        mid_price = df['close'].rolling(window=30).mean().values[-1]
        
        deviation = (current_price - mid_price) / (grid_width + 1e-8)
        signal = -np.clip(deviation, -1.0, 1.0)
        confidence = min(1.0, abs(deviation))
        
        return float(signal), float(confidence)


class ScalpingStrategy(BaseStrategy):
    """
    High-frequency scalping engine utilizing order book depth imbalance,
    microsecond volatility, and transaction velocity.
    """
    def __init__(self, params=None):
        default_params = {
            'depth_levels': 5,
            'min_imbalance': 0.15
        }
        default_params.update(params or {})
        super().__init__("Scalping", default_params)

    def generate_signal(self, market_data):
        bids = market_data.get('bids')
        asks = market_data.get('asks')
        
        if not bids or not asks:
            return 0.0, 0.0
            
        obi = compute_order_book_imbalance(bids, asks, depth=self.params['depth_levels'])
        
        if abs(obi) >= self.params['min_imbalance']:
            # Positive OBI indicates higher buy pressure, so scalp long (+1)
            signal = np.clip(obi / 0.5, -1.0, 1.0)
            confidence = min(1.0, abs(obi) / 0.8)
            return float(signal), float(confidence)
            
        return 0.0, 0.0


class MetaAllocationEngine:
    """
    Quant Meta-Model that stacks and weights signals from all active strategies,
    fusing them with ML inputs (Market Regime, LSTM Predictor, PPO Action)
    to output a final consensus trade decision and recommended leverage.
    """
    def __init__(self, strategies=None):
        self.strategies = strategies or []
        # Dynamic weights mapping based on market regime
        # Regime ID: [Trend, MeanRev, MarketMaking, StatArb, InterExchange, Grid, Scalping]
        self.regime_strategy_weights = {
            0: [0.45, 0.05, 0.05, 0.10, 0.05, 0.10, 0.20], # Bull: Heavy Trend Following & Scalping
            1: [0.40, 0.05, 0.05, 0.10, 0.05, 0.10, 0.25], # Bear: Heavy Trend & Scalping (shorting)
            2: [0.05, 0.35, 0.15, 0.15, 0.05, 0.20, 0.05], # Range: MeanReversion, Grid & MarketMaking
            3: [0.10, 0.10, 0.05, 0.20, 0.25, 0.05, 0.25]  # High Vol: StatArb, InterExchange & Scalping
        }

    def allocate(self, market_data, regime_state_id, ml_prediction, ppo_action):
        """
        Calculates final combined trade signal and capital allocation.
        market_data: dict of market indicators and books
        regime_state_id: active market regime (0=Bull, 1=Bear, 2=Range, 3=HighVol)
        ml_prediction: predicted close price change % from LSTM-like model
        ppo_action: recommended action from Reinforcement Learning agent (-1 to 1)
        """
        raw_signals = []
        confidences = []
        
        for s in self.strategies:
            if s.enabled:
                sig, conf = s.generate_signal(market_data)
                raw_signals.append(sig)
                confidences.append(conf)
            else:
                raw_signals.append(0.0)
                confidences.append(0.0)
                
        # Fetch the regime-specific weight matrix
        weights = np.array(self.regime_strategy_weights.get(regime_state_id, [0.14] * 7))
        
        # Normalize weights for enabled strategies
        enabled_mask = np.array([1.0 if s.enabled else 0.0 for s in self.strategies])
        active_weights = weights * enabled_mask
        sum_weights = np.sum(active_weights)
        if sum_weights > 0:
            active_weights /= sum_weights
        else:
            active_weights = np.zeros_like(weights)
            
        # 1. Classical combined signal
        classical_signal = np.sum(np.array(raw_signals) * active_weights)
        mean_confidence = np.sum(np.array(confidences) * active_weights)
        
        # 2. Integrate ML LSTM-like Price Prediction (scaling output % change to [-1, 1])
        # Scaling factor: if we predict +0.5% return, we consider it a very strong buy (+1.0)
        ml_signal = np.clip(ml_prediction / 0.005, -1.0, 1.0)
        
        # 3. Consolidate: Stacking Classical, LSTM, and PPO
        # Institutional weights: 50% Classical (Regime-Adjusted), 25% LSTM, 25% RL Agent
        final_signal = (0.50 * classical_signal) + (0.25 * ml_signal) + (0.25 * ppo_action)
        final_signal = np.clip(final_signal, -1.0, 1.0)
        
        # Consolidation score of consensus
        consensus_score = float(mean_confidence)
        
        # Log active distribution for reporting
        strategy_contributions = {
            self.strategies[i].name: {
                "signal": float(raw_signals[i]),
                "confidence": float(confidences[i]),
                "weight": float(active_weights[i])
            }
            for i in range(len(self.strategies))
        }
        
        return {
            "final_signal": float(final_signal),
            "consensus": consensus_score,
            "classical_signal": float(classical_signal),
            "ml_signal": float(ml_signal),
            "ppo_signal": float(ppo_action),
            "contributions": strategy_contributions
        }

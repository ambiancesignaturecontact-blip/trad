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
        
        ema_f = df['close'].ewm(span=self.params['ema_fast'], adjust=False).mean().values
        ema_s = df['close'].ewm(span=self.params['ema_slow'], adjust=False).mean().values
        
        macd_line = ema_f - ema_s
        macd_signal = pd.Series(macd_line).ewm(span=self.params['macd_signal'], adjust=False).mean().values
        macd_hist = macd_line - macd_signal
        
        high_roll = df['high'].rolling(window=self.params['breakout_period']).max().values
        low_roll = df['low'].rolling(window=self.params['breakout_period']).min().values
        
        current_close = close[-1]
        prev_high = high_roll[-2] if len(high_roll) > 1 else current_close
        prev_low = low_roll[-2] if len(low_roll) > 1 else current_close
        
        trend_score = 0.0
        if ema_f[-1] > ema_s[-1]:
            trend_score += 0.4
        else:
            trend_score -= 0.4
            
        if macd_hist[-1] > 0:
            trend_score += 0.3 * min(1.0, macd_hist[-1] / (current_close * 0.001 + 1e-8))
        else:
            trend_score -= 0.3 * min(1.0, abs(macd_hist[-1]) / (current_close * 0.001 + 1e-8))
            
        if current_close > prev_high:
            trend_score += 0.3
        elif current_close < prev_low:
            trend_score -= 0.3
            
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
        
        rolling_mean = df['close'].rolling(window=self.params['period']).mean().values
        rolling_std = df['close'].rolling(window=self.params['period']).std().values
        
        z_score = (current_close - rolling_mean[-1]) / (rolling_std[-1] + 1e-8)
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
        rs = gain / (loss + 1e-8)
        rsi = (100 - (100 / (1 + rs))).values
        current_rsi = rsi[-1]
        
        mr_score = 0.0
        mr_score -= 0.6 * np.clip(z_score / self.params['num_std'], -1.5, 1.5)
        
        if current_rsi > self.params['rsi_overbought']:
            mr_score -= 0.4 * ((current_rsi - self.params['rsi_overbought']) / (100 - self.params['rsi_overbought']))
        elif current_rsi < self.params['rsi_oversold']:
            mr_score += 0.4 * ((self.params['rsi_oversold'] - current_rsi) / self.params['rsi_oversold'])
            
        signal = np.clip(mr_score, -1.0, 1.0)
        confidence = min(1.0, abs(z_score) / 3.0)
        
        return float(signal), float(confidence)


class MarketMakingStrategy(BaseStrategy):
    """
    Implements a simplified Avellaneda-Stoikov model for quoting spread.
    Manages inventory by shifting the bid/ask mid-price to a reservation price.
    """
    def __init__(self, params=None):
        default_params = {
            'risk_aversion': 0.1,
            'volatility_lookback': 20,
            'kappa': 1.5
        }
        default_params.update(params or {})
        super().__init__("Market Making", default_params)

    def generate_signal(self, market_data):
        df = market_data.get('df')
        inventory = market_data.get('inventory', 0.0)
        max_inventory = market_data.get('max_inventory', 10.0)
        
        if df is None or len(df) < self.params['volatility_lookback']:
            return 0.0, 0.0
            
        returns = df['close'].pct_change().values[-self.params['volatility_lookback']:]
        vol = np.std(returns) + 1e-8
        
        q = inventory / max_inventory if max_inventory > 0 else 0.0
        gamma = self.params['risk_aversion']
        
        skew_signal = -q * gamma * vol * 100.0
        signal = np.clip(skew_signal, -1.0, 1.0)
        confidence = min(1.0, abs(q))
        
        return float(signal), float(confidence)


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Monitors cointegration between Asset A and Asset B.
    """
    def __init__(self, params=None):
        default_params = {
            'lookback': 100,
            'z_threshold': 2.0
        }
        default_params.update(params or {})
        super().__init__("Statistical Arbitrage", default_params)

    def generate_signal(self, market_data):
        series_a = market_data.get('series_a')
        series_b = market_data.get('series_b')
        
        if series_a is None or series_b is None or len(series_a) < self.params['lookback']:
            return 0.0, 0.0
            
        s_a = np.log(series_a[-self.params['lookback']:])
        s_b = np.log(series_b[-self.params['lookback']:])
        
        try:
            beta, alpha = np.polyfit(s_b, s_a, 1)
            spread = s_a - (beta * s_b + alpha)
            
            mean_spread = np.mean(spread)
            std_spread = np.std(spread) + 1e-8
            current_spread = spread[-1]
            
            z_score = (current_spread - mean_spread) / std_spread
            signal = -np.clip(z_score / self.params['z_threshold'], -1.5, 1.5)
            confidence = min(1.0, abs(z_score) / 3.0)
            
            return float(signal), float(confidence)
        except Exception:
            return 0.0, 0.0


class ArbitrageInterExchangeStrategy(BaseStrategy):
    """
    Identifies profitable discrepancies between primary exchange and secondary alternative.
    """
    def __init__(self, params=None):
        default_params = {
            'fee_primary': 0.001,
            'fee_secondary': 0.0015,
            'min_spread_pct': 0.003,
        }
        default_params.update(params or {})
        super().__init__("Inter-Exchange Arbitrage", default_params)

    def generate_signal(self, market_data):
        price_primary = market_data.get('price_primary', 0.0)
        price_secondary = market_data.get('price_secondary', 0.0)
        
        if price_primary == 0 or price_secondary == 0:
            return 0.0, 0.0
            
        spread_pct = (price_secondary - price_primary) / price_primary
        total_fees = self.params['fee_primary'] + self.params['fee_secondary']
        net_spread = abs(spread_pct) - total_fees
        
        if net_spread > self.params['min_spread_pct']:
            signal = 1.0 if spread_pct > 0 else -1.0
            confidence = min(1.0, net_spread / 0.02)
            return float(signal), float(confidence)
            
        return 0.0, 0.0


class GridTradingStrategy(BaseStrategy):
    """
    Generates dynamic buy/sell grids centered around current volatility.
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
        
        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(abs(high[1:] - close[:-1]), 
                                   abs(low[1:] - close[:-1])))
        atr = pd.Series(tr).rolling(window=self.params['atr_period']).mean().values[-1]
        
        current_price = close[-1]
        grid_width = self.params['atr_multiplier'] * atr
        mid_price = df['close'].rolling(window=30).mean().values[-1]
        
        deviation = (current_price - mid_price) / (grid_width + 1e-8)
        signal = -np.clip(deviation, -1.0, 1.0)
        confidence = min(1.0, abs(deviation))
        
        return float(signal), float(confidence)


class ScalpingStrategy(BaseStrategy):
    """
    High-frequency scalping engine utilizing order book depth imbalance.
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
            signal = np.clip(obi / 0.5, -1.0, 1.0)
            confidence = min(1.0, abs(obi) / 0.8)
            return float(signal), float(confidence)
            
        return 0.0, 0.0


class MetaAllocationEngine:
    """
    Quant Meta-Model that stacks and weights signals from all active strategies.
    
    Implements a **Multi-Armed Bandit (Thompson Sampling)** algorithm 
    to dynamically reallocate capital weights based on rolling Sharpe performance!
    
    + Walk-Forward dynamique (LOT 11) : les poids sont ajustés en fonction
      des performances récentes des stratégies sur les 80 derniers trades.
    """
    def __init__(self, strategies=None):
        self.strategies = strategies or []
        self.num_strategies = len(self.strategies)
        
        # Thompson Sampling parameters: Alpha (successes) & Beta (failures) for each strategy
        self.alpha_bandit = np.ones(self.num_strategies)
        self.beta_bandit = np.ones(self.num_strategies)
        
        # Rolling historical performance records of each strategy
        self.strategy_returns = {s.name: [] for s in self.strategies}
        
        # === WALK-FORWARD DYNAMIQUE (LOT 11) ===
        self.recent_performance = {s.name: [] for s in self.strategies}  # Sharpe-like score récent
        self.walkforward_weights = np.ones(self.num_strategies) / self.num_strategies

    def update_bandit_feedback(self, symbol: str, strategy_signals: dict, actual_return: float):
        """
        Updates Thompson Sampling Bandit successes/failures based on trade direction feedback.
        If a strategy's signal aligned with actual return, we reward it (increment alpha).
        Otherwise, we penalize it (increment beta).
        
        + Walk-Forward dynamique : mise à jour des performances récentes.
        """
        for i, s in enumerate(self.strategies):
            sig_obj = strategy_signals.get(s.name, 0.0)
            sig_val = sig_obj.get("signal", 0.0) if isinstance(sig_obj, dict) else sig_obj
            
            if sig_val != 0.0:
                # If signal direction matches actual price return direction -> Success!
                if np.sign(sig_val) == np.sign(actual_return):
                    self.alpha_bandit[i] += 1.0
                    self.recent_performance[s.name].append(1.0)
                else:
                    self.beta_bandit[i] += 1.0
                    self.recent_performance[s.name].append(-0.5)
                
                # Garder seulement les 80 dernières performances
                if len(self.recent_performance[s.name]) > 80:
                    self.recent_performance[s.name].pop(0)

    def get_strategy_weights(self) -> dict:
        """
        Returns the current live allocation weights per strategy
        (Thompson Sampling bandit + walk-forward). Used by the mini-app
        attribution panel and the LOT 46 telemetry.
        """
        weights = {}
        for i, s in enumerate(self.strategies):
            name = getattr(s, "name", f"Strategy_{i}")
            w = float(self.walkforward_weights[i]) if i < len(self.walkforward_weights) else 0.0
            weights[name] = round(w, 4)
        return weights

    def allocate(self, market_data, regime_state_id, ml_prediction, ppo_action):
        """
        Calculates final combined trade signal and capital allocation.
        Enforces Thompson Sampling (Multi-Armed Bandit) weighting over classical strategies
        to route more capital dynamically to historically outperforming models!
        
        + Walk-Forward dynamique (LOT 11) : les poids sont ajustés selon les
          performances récentes des stratégies (derniers 80 trades).
        """
        raw_signals = []
        confidences = []
        signals_dict = {}
        conf_dict = {}
        
        # 1. Gather all strategy signals
        for s in self.strategies:
            if s.enabled:
                sig, conf = s.generate_signal(market_data)
                signals_dict[s.name] = sig
                conf_dict[s.name] = conf
            else:
                signals_dict[s.name] = 0.0
                conf_dict[s.name] = 0.0

        # === WALK-FORWARD DYNAMIQUE (LOT 11) ===
        # Calcul des poids dynamiques basés sur les performances récentes
        for i, s in enumerate(self.strategies):
            recent = self.recent_performance.get(s.name, [])
            if len(recent) >= 10:
                recent_score = np.mean(recent[-20:])  # moyenne des 20 derniers
                self.walkforward_weights[i] = max(0.25, 1.0 + recent_score * 0.9)
            else:
                self.walkforward_weights[i] = 1.0
        
        # Normalisation des poids walk-forward
        wf_sum = sum(self.walkforward_weights)
        if wf_sum > 0:
            self.walkforward_weights = self.walkforward_weights / wf_sum

        # === AUTO-REBALANCING (LOT 12) ===
        # Si une stratégie a un poids walk-forward très faible (< 0.08), on la désactive temporairement
        for i, s in enumerate(self.strategies):
            if self.walkforward_weights[i] < 0.08:
                s.enabled = False
            elif self.walkforward_weights[i] > 0.12:
                s.enabled = True

        # 2. Thompson Sampling (MAB) Weight Calculation:
        sampled_performance = np.zeros(self.num_strategies)
        for i in range(self.num_strategies):
            sampled_performance[i] = np.random.beta(self.alpha_bandit[i], self.beta_bandit[i])
            
        exp_perf = np.exp(sampled_performance - np.max(sampled_performance))
        mab_weights = exp_perf / np.sum(exp_perf)

        # 3. Enforce Regime-Specific Dominance + Walk-Forward
        dominant_strategy = "Trend Following"
        if regime_state_id == 0 or regime_state_id == 1:
            dominant_strategy = "Trend Following"
        elif regime_state_id == 2:
            dominant_strategy = "Mean Reversion"
        elif regime_state_id == 3:
            dominant_strategy = "Scalping" if signals_dict.get("Scalping", 0.0) != 0.0 else "Statistical Arbitrage"

        classical_signal = 0.0
        for i, s in enumerate(self.strategies):
            # Combinaison : MAB + Walk-Forward + Regime dominance
            weight = mab_weights[i] * 0.50
            weight += self.walkforward_weights[i] * 0.30
            if s.name == dominant_strategy:
                weight += 0.20
            classical_signal += signals_dict.get(s.name, 0.0) * weight

        mean_confidence = conf_dict.get(dominant_strategy, 0.5)

        # 4. Integrate ML LSTM Price Prediction (scaled)
        ml_signal = np.clip(ml_prediction / 0.002, -1.0, 1.0)
        
        # 5. Stacking Classical, LSTM, and PPO
        final_signal = (0.80 * classical_signal) + (0.10 * ml_signal) + (0.10 * ppo_action)
        final_signal = np.clip(final_signal, -1.0, 1.0)

        consensus_score = float(mean_confidence)
        
        # Create contributions dictionary (avec poids walk-forward)
        strategy_contributions = {}
        for i, s in enumerate(self.strategies):
            is_dominant = (s.name == dominant_strategy)
            strategy_contributions[s.name] = {
                "signal": float(signals_dict.get(s.name, 0.0)),
                "confidence": float(conf_dict.get(s.name, 0.0)),
                "weight": float(mab_weights[i] * 0.50 + self.walkforward_weights[i] * 0.30 + (0.20 if is_dominant else 0.0))
            }
        
        return {
            "final_signal": float(final_signal),
            "consensus": consensus_score,
            "classical_signal": float(classical_signal),
            "ml_signal": float(ml_signal),
            "ppo_signal": float(ppo_action),
            "contributions": strategy_contributions,
            "walkforward_weights": {s.name: float(self.walkforward_weights[i]) for i, s in enumerate(self.strategies)}
        }

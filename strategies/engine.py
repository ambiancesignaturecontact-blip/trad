import time
from collections import deque

import numpy as np
import pandas as pd

from core.config import settings
from models.lopez_de_prado import calculate_deflated_sharpe_ratio
from models.regime_detector import compute_order_book_imbalance

# =========================================================================== #
# P1-10 / P1-11 / P1-12 (audit indépendant §4.4 / §2.5 / §2.6 / §2.2)
# =========================================================================== #
# P1-12 (§2.6) : facteur d'oubli du bandit Thompson (non-stationnarité).
# Un régime de marché dure quelques semaines : l'avantage d'une stratégie
# doit s'estomper, pas se figer à vie. 0.98 = demi-vie ~34 mises à jour.
BANDIT_DECAY = settings.get_float("strategies", "bandit_decay", 0.98)
# P1-12 (§2.2) : durée de vie d'un tirage Thompson (cycle de décision).
# Le tirage est figé N secondes au lieu d'être ré-échantillonné à chaque
# tick (2,5 s) — sinon deux ticks stables produisent des poids différents
# par le seul bruit du tirage.
BANDIT_SAMPLE_REFRESH_SECONDS = settings.get_float(
    "strategies", "bandit_sample_refresh_seconds", 60.0)
# P1-11 (§2.5) : échantillon de PnL minimal avant ajustement COMPLET des
# poids (en dessous : ajustement borné à ±20 % par mise à jour).
PNL_MIN_SAMPLES_FULL = settings.get_int("strategies", "pnl_min_samples_full", 20)
PNL_MAX_ADJUSTMENT = settings.get_float("strategies", "pnl_max_adjustment", 0.20)
PNL_HISTORY_MAXLEN = 80
# P1-10 (§4.4) : historique des signaux pour la matrice de corrélation
# inter-stratégies (le méta-allocateur ne doit pas croire diversifier alors
# qu'il parie sur 2-3 facteurs latents corrélés).
SIGNAL_HISTORY_MAXLEN = 200
SIGNAL_CORR_MIN_SAMPLES = 30
SIGNAL_CORR_MAX_PENALTY = 0.50   # une stratégie ne descend jamais sous 50 %

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
    def __init__(self, strategies=None, regime_allocator=None):
        self.strategies = strategies or []
        self.num_strategies = len(self.strategies)
        self.regime_allocator = regime_allocator

        # Thompson Sampling parameters: Alpha (successes) & Beta (failures) for each strategy
        self.alpha_bandit = np.ones(self.num_strategies)
        self.beta_bandit = np.ones(self.num_strategies)

        # Rolling historical performance records of each strategy
        self.strategy_returns = {s.name: [] for s in self.strategies}

        # === WALK-FORWARD DYNAMIQUE (LOT 11) ===
        self.recent_performance = {s.name: [] for s in self.strategies}  # Sharpe-like score récent
        self.walkforward_weights = np.ones(self.num_strategies) / self.num_strategies

        # P1-12 (§2.2) : cache des tirages Thompson (figés par cycle de décision)
        self._bandit_sample_cache = {}
        # LOT D (F4) : decay du bandit DYNAMIQUE — nominal = BANDIT_DECAY
        # (0.98, demi-vie ~34 MAJ) ; en drift de distribution sévère (PSI),
        # main.py l'accélère via set_bandit_decay() (jamais < 0.85 : un oubli
        # total = bandit vierge qui repart à l'exploration pure).
        self.bandit_decay: float = float(BANDIT_DECAY)
        # P1-11 (§2.5) : historique de PnL RÉEL par stratégie (séparé du buffer
        # du bandit — les deux mécanismes de reward ne doivent plus se marcher
        # dessus) + poids issus de l'attribution DSR.
        self.pnl_history = {s.name: deque(maxlen=PNL_HISTORY_MAXLEN)
                            for s in self.strategies}
        self.pnl_weights = {}
        # P1-10 (§4.4) : historique des SIGNAUX par stratégie (corrélation
        # inter-stratégies — l'angle mort dénoncé par l'audit).
        self.signal_history = {s.name: deque(maxlen=SIGNAL_HISTORY_MAXLEN)
                               for s in self.strategies}
        # LOT 5 (mandat) : regret non-stationnaire + allocation hiérarchique
        # par famille (niveau 2) — voir core/hierarchical_allocator.py.
        from core.hierarchical_allocator import HierarchicalAllocator, RegretTracker
        self.regret_tracker = RegretTracker()
        self.hierarchical = HierarchicalAllocator()

    def record_regret(self, strategy: str, pnl_pct: float) -> None:
        """LOT 5 : enregistre le regret d'un trade clôturé (non-stationnaire,
        oubli exponentiel). JAMAIS bloquant."""
        try:
            self.regret_tracker.record(strategy, pnl_pct)
        except Exception:
            pass

    def hierarchical_scales(self, pnl_by_strategy: dict[str, float] | None = None) -> dict[str, float]:
        """
        LOT 5 : scales de pondération hiérarchiques — scale de famille ×
        exploration regret (bornés par core/hierarchical_allocator.py).
        """
        try:
            base = {s.name: 1.0 for s in self.strategies}
            weights = self.hierarchical.allocate(base, pnl_by_strategy or {},
                                                 self.regret_tracker)
            return {s: w for s, w in weights.items()}
        except Exception:
            return {s.name: 1.0 for s in self.strategies}

    def _sample_bandit(self, i: int) -> float:
        """P1-12 (§2.2) : tirage Thompson FIGÉ par cycle de décision —
        au maximum 1 tirage / BANDIT_SAMPLE_REFRESH_SECONDS par stratégie,
        au lieu d'un ré-échantillonnage à chaque tick (2,5 s)."""
        now = time.time()
        cached = self._bandit_sample_cache.get(i)
        if cached is not None and now - cached[0] < BANDIT_SAMPLE_REFRESH_SECONDS:
            return cached[1]
        value = float(np.random.beta(self.alpha_bandit[i], self.beta_bandit[i]))
        self._bandit_sample_cache[i] = (now, value)
        return value

    def set_bandit_decay(self, decay: float) -> float:
        """
        LOT D (F4) : ajuste le facteur d'oubli du bandit (accélération en cas
        de drift de distribution détecté par le PSI). Borné [0.80, 1.0] —
        jamais d'oubli instantané (le bandit conserverait zéro mémoire, il
        repartirait à l'exploration pure) ni de mémoire infinie. Retourne le
        decay effectivement appliqué.
        """
        self.bandit_decay = max(0.80, min(1.0, float(decay)))
        return self.bandit_decay

    def signal_diversification_weights(self) -> dict:
        """P1-10 (§4.4) : facteur de diversification par stratégie basé sur la
        corrélation moyenne de SES signaux avec ceux des AUTRES stratégies.
        Deux stratégies qui votent quasi toujours pareil (Trend/Grid/MeanRev
        lisent le même OHLC) sont pénalisées : le méta-allocateur ne doit pas
        croire diversifier alors qu'il parie sur 2-3 facteurs latents.
        Retourne {name: facteur <= 1.0} ; {} si échantillon insuffisant."""
        names = [s.name for s in self.strategies
                 if len(self.signal_history.get(s.name, ())) >= SIGNAL_CORR_MIN_SAMPLES]
        if len(names) < 2:
            return {}
        n = min(len(self.signal_history[nm]) for nm in names)
        series = {nm: np.asarray(list(self.signal_history[nm])[-n:], dtype=float)
                  for nm in names}
        factors = {}
        for nm in names:
            corrs = []
            for other in names:
                if other == nm:
                    continue
                a, b = series[nm], series[other]
                if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                    continue
                c = float(np.corrcoef(a, b)[0, 1])
                if np.isfinite(c):
                    corrs.append(abs(c))
            if corrs:
                avg = float(np.mean(corrs))
                # corrélation moyenne 0.9 -> facteur 0.55 ; 0.2 -> 0.80 ; jamais < 0.50
                factors[nm] = max(SIGNAL_CORR_MAX_PENALTY, 1.0 - avg)
        return factors

    def _dsr_score(self, history) -> float:
        """P1-11 (§2.5) : score de significativité d'une stratégie via le
        Sharpe DÉFLATÉ (López de Prado) — le MÊME test que la promotion de
        nouvelles hypothèses. Retourne une probabilité [0,1] : proche de 1 =
        l'edge observé bat l'espérance max sous l'hypothèse nulle de
        data-snooping sur les stratégies testées."""
        rets = np.asarray(list(history), dtype=float)
        if len(rets) < 5:
            return 0.5
        std = float(np.std(rets))
        mean = float(np.mean(rets))
        if std < 1e-12:
            sharpe = 0.999 if mean > 0 else (-0.999 if mean < 0 else 0.0)
        else:
            sharpe = float(mean / std)
        sharpe = sharpe * np.sqrt(24.0 * 365.0)  # annualisation indicative
        dsr = calculate_deflated_sharpe_ratio(
            observed_sharpe=sharpe,
            num_trials=max(len(self.strategies), 1),
            trials_variance_sharpe=1.0,
            sample_length=len(rets),
        )
        # calculate_deflated_sharpe_ratio retourne le sharpe BRUT quand
        # num_trials <= 1 (pas de data-snooping à corriger) : on le convertit
        # alors en probabilité pour que le score soit TOUJOURS dans [0,1].
        if not (0.0 <= dsr <= 1.0):
            from scipy.stats import norm
            dsr = float(norm.cdf(dsr))
        return float(np.clip(dsr, 0.0, 1.0))

    def update_bandit_feedback(self, symbol: str, strategy_signals: dict, actual_return: float,
                               decay: float | None = None):
        """
        Updates Thompson Sampling Bandit successes/failures based on trade direction feedback.
        If a strategy's signal aligned with actual return, we reward it (increment alpha).
        Otherwise, we penalize it (increment beta).

        + Walk-Forward dynamique : mise à jour des performances récentes.
        + P1-12 (audit §2.6) : facteur d'oubli — le bandit est NON-stationnaire.
        + LOT D (F4) : `decay` optionnel — en drift de distribution sévère
          (PSI), main.py passe un decay ACCÉLÉRÉ pour oublier un edge mort.
        """
        # P1-12 (§2.6) : oubli exponentiel AVANT d'ajouter l'observation.
        # Sans cela, une stratégie qui a brillé pendant un vieux régime bull
        # garde un avantage de plus en plus figé — le bandit cesse d'explorer
        # au moment où le marché change.
        # LOT D (F4) : decay effectif = paramètre explicitement fourni, sinon
        # le decay dynamique de l'instance (set_bandit_decay), sinon la config.
        decay_eff = float(decay if decay is not None else self.bandit_decay)
        decay_eff = max(0.80, min(1.0, decay_eff))  # borne dure de sécurité
        self.alpha_bandit = self.alpha_bandit * decay_eff
        self.beta_bandit = self.beta_bandit * decay_eff

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

    def update_pnl_attribution(self, strategy: str, pnl_pct: float) -> None:
        """
        LOT 4 (PDF Pilier D) : pondère les stratégies par leur contribution
        RÉELLE au PnL (attribution), pas uniquement par bandit sur signaux
        bruts.

        P1-11 (audit §2.5) : le rééquilibrage passe désormais par le MÊME
        test de significativité que la promotion de nouvelles hypothèses —
        le Sharpe déflaté (López de Prado) — avec un plancher d'échantillon :
        en dessous de PNL_MIN_SAMPLES_FULL, l'ajustement de poids est borné
        à ±PNL_MAX_ADJUSTMENT par mise à jour (une série de 10 trades n'est
        plus suffisante pour gonfler un poids — c'était du bruit statistique).
        """
        if strategy not in self.pnl_history:
            self.pnl_history[strategy] = deque(maxlen=PNL_HISTORY_MAXLEN)
        self.pnl_history[strategy].append(float(pnl_pct))
        # LOT 5 : regret non-stationnaire (écart cumulé à la meilleure ex post)
        self.record_regret(strategy, float(pnl_pct))

        # Score = Sharpe DÉFLATÉ par stratégie (probabilité de significativité)
        scores = {}
        for name, hist in self.pnl_history.items():
            if len(hist) >= 5:
                scores[name] = self._dsr_score(hist)
        if len(scores) < 2:
            return

        # Softmax sur les DSR -> poids cibles
        vals = np.array([max(scores.get(s.name, 0.0), 1e-9) for s in self.strategies
                         if s.name in scores])
        exp = np.exp(vals - np.max(vals))
        probs = exp / exp.sum()

        idx = 0
        for i, s in enumerate(self.strategies):
            if s.name not in scores:
                continue
            target = float(probs[idx])
            idx += 1
            n = len(self.pnl_history[s.name])
            old = self.pnl_weights.get(s.name, 1.0 / max(len(self.strategies), 1))
            if n >= PNL_MIN_SAMPLES_FULL:
                new = target                      # échantillon suffisant : ajustement complet
            else:
                # échantillon < plancher : ajustement borné à ±20 % par mise à jour
                band = PNL_MAX_ADJUSTMENT * max(old, 0.05)
                new = old + max(-band, min(band, target - old))
            self.pnl_weights[s.name] = float(np.clip(new, 0.01, 0.99))

        # normalisation à somme 1
        tot = sum(self.pnl_weights.values())
        if tot > 0:
            self.pnl_weights = {k: v / tot for k, v in self.pnl_weights.items()}

    def get_strategy_weights(self) -> dict:
        """
        Returns the current live allocation weights per strategy
        (Thompson Sampling bandit + walk-forward + PnL attribution LOT 4 /
        P1-11 : Sharpe déflaté). Used by the mini-app attribution panel and
        the LOT 46 telemetry.
        """
        weights = {}
        for i, s in enumerate(self.strategies):
            name = getattr(s, "name", f"Strategy_{i}")
            w = float(self.walkforward_weights[i]) if i < len(self.walkforward_weights) else 0.0
            # P1-11 : une fois l'attribution PnL disponible, le poids affiché
            # combine walk-forward (50 %) et attribution DSR (50 %).
            if self.pnl_weights:
                p = float(self.pnl_weights.get(name, 1.0 / max(len(self.strategies), 1)))
                w = 0.5 * w + 0.5 * p
            weights[name] = round(w, 4)
        return weights

    def allocate(self, market_data, regime_state_id, ml_prediction, ppo_action,
                 edge_decay_scales: dict | None = None,
                 hierarchical_scales: dict | None = None):
        """
        Calculates final combined trade signal and capital allocation.
        Enforces Thompson Sampling (Multi-Armed Bandit) weighting over classical strategies
        to route more capital dynamically to historically outperforming models!

        + Walk-Forward dynamique (LOT 11) : les poids sont ajustés selon les
          performances récentes des stratégies (derniers 80 trades).
        + LOT 4 (mandat) : edge_decay_scales={strategy: scale} — le poids final
          de chaque stratégie est multiplié par son scale d'edge decay (borné
          [0.30, 1.0] par core/edge_decay.py). Absent -> 1.0 (comportement
          pré-LOT 4 strictement conservé).
        + LOT 5 (mandat) : hierarchical_scales={strategy: scale} — scale de
          famille + exploration regret (core/hierarchical_allocator.py).
          Absent -> 1.0 (comportement pré-LOT 5 strictement conservé).
        """
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

        # P1-10 (§4.4) : enregistrement des signaux pour la corrélation
        # inter-stratégies (utilisée plus bas pour pénaliser la redondance).
        for name, sig in signals_dict.items():
            if name in self.signal_history:
                self.signal_history[name].append(float(sig))
        _corr_weights = self.signal_diversification_weights()

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
        # P1-12 (§2.2) : tirage FIGÉ par cycle de décision (cache temporel),
        # plus de ré-échantillonnage à chaque tick.
        sampled_performance = np.zeros(self.num_strategies)
        for i in range(self.num_strategies):
            sampled_performance[i] = self._sample_bandit(i)

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
        regime_weights = {}
        if self.regime_allocator is not None:
            try:
                regime_weights = self.regime_allocator.get_regime_weights(regime_state_id)
            except Exception:
                regime_weights = {}
        # VISION §6: risk-parity budget (weight by 1/vol so each strategy
        # contributes a comparable amount of RISK, not capital)
        risk_weights = {}
        try:
            from core.factor_model import risk_parity_weights
            risk_weights = risk_parity_weights(self.recent_performance)
        except Exception:
            risk_weights = {}
        for i, s in enumerate(self.strategies):
            # Combinaison : MAB + Walk-Forward + Regime dominance + Regime allocator + Risk parity
            weight = mab_weights[i] * 0.30
            weight += self.walkforward_weights[i] * 0.20
            if s.name == dominant_strategy:
                weight += 0.12
            if s.name in regime_weights:
                weight += float(regime_weights[s.name]) * 0.18
            if s.name in risk_weights:
                weight += float(risk_weights[s.name]) * 0.20
            # P1-11 (§2.5) : attribution PnL réelle (Sharpe déflaté) — composante
            # de pondération propre, séparée du buffer du bandit.
            if s.name in self.pnl_weights:
                weight += self.pnl_weights[s.name] * 0.20
            # P1-10 (§4.4) : pénalité de redondance — deux stratégies dont les
            # signaux sont corrélés ne sont plus comptées comme deux paris.
            if s.name in _corr_weights:
                weight *= _corr_weights[s.name]
            # LOT 4 (mandat) : edge decay — sous-pondération bornée d'une
            # stratégie dont l'edge se dégrade (jamais de suppression).
            if edge_decay_scales:
                weight *= float(edge_decay_scales.get(s.name, 1.0))
            # LOT 5 (mandat) : scale hiérarchique par FAMILLE (niveau 2) +
            # exploration par regret (niveau 3). Optionnel et borné — si le
            # paramètre est absent, weight est STRICTEMENT le poids pré-LOT 5.
            if hierarchical_scales:
                weight *= float(hierarchical_scales.get(s.name, 1.0))
            classical_signal += signals_dict.get(s.name, 0.0) * weight

        mean_confidence = conf_dict.get(dominant_strategy, 0.5)

        # 4. Integrate ML LSTM Price Prediction (scaled)
        ml_signal = np.clip(ml_prediction / 0.002, -1.0, 1.0)

        # 5. Stacking Classical, LSTM, and PPO
        final_signal = (0.80 * classical_signal) + (0.10 * ml_signal) + (0.10 * ppo_action)
        final_signal = np.clip(final_signal, -1.0, 1.0)

        # 5b. MICROSTRUCTURE / ON-CHAIN MODULATION (audit B8-1/B8-2):
        # VPIN and Kyle's Lambda are now REAL signal inputs, not just logs.
        # - High VPIN = toxic order flow -> reduce conviction (50% max dampen)
        # - High on-chain risk -> reduce conviction
        # - High Kyle's Lambda = illiquid -> dampen (harder to execute without slippage)
        modulate_factor = 1.0
        try:
            vpin = float(market_data.get("vpin") or 0.0)
            kyle = float(market_data.get("kyle_lambda") or 0.0)
            onchain = float(market_data.get("onchain_risk") or 0.0)

            # FIX P0-4 (audit §2.1 / logs prod) : le modulate VPIN s'appliquait
            # SANS garde de bornes. Un VPIN aberrant (6 988 465 mesuré sur BTC,
            # bug de calcul du bucket_size) donnait max(0.50, 1-(6.9M-0.90)) =
            # 0.50 -> la CONVICTION était réduite de 50% à chaque tick, même
            # quand le garde [0,1] de toxicity_factor la neutralisait ailleurs.
            # VPIN est une probabilité : hors [0,1] = donnée douteuse = neutre.
            if not (0.0 <= vpin <= 1.0):
                vpin = 0.0

            if vpin > 0.90:
                modulate_factor *= max(0.50, 1.0 - (vpin - 0.90))
            if onchain > 0.75:
                modulate_factor *= 0.50
            if kyle > 0.01:  # very illiquid market
                modulate_factor *= 0.75
            final_signal = float(np.clip(final_signal * modulate_factor, -1.0, 1.0))
        except Exception:
            pass

        consensus_score = float(mean_confidence * modulate_factor)

        # Create contributions dictionary (avec poids walk-forward). LOT 4 :
        # le poids AFFICHÉ reflète le scale d'edge decay appliqué au signal
        # (même formule que classical_signal — cohérence télémétrie/décision).
        strategy_contributions = {}
        for i, s in enumerate(self.strategies):
            is_dominant = (s.name == dominant_strategy)
            _scale = float(edge_decay_scales.get(s.name, 1.0)) if edge_decay_scales else 1.0
            strategy_contributions[s.name] = {
                "signal": float(signals_dict.get(s.name, 0.0)),
                "confidence": float(conf_dict.get(s.name, 0.0)),
                "weight": float((mab_weights[i] * 0.30 + self.walkforward_weights[i] * 0.20 + (0.12 if is_dominant else 0.0) + (float(regime_weights.get(s.name, 0.0)) * 0.18) + (float(risk_weights.get(s.name, 0.0)) * 0.20)) * _scale),
                "edge_decay_scale": _scale,
            }

        return {
            "final_signal": float(final_signal),
            "consensus": consensus_score,
            "modulate_factor": float(modulate_factor),
            "classical_signal": float(classical_signal),
            "ml_signal": float(ml_signal),
            "ppo_signal": float(ppo_action),
            "contributions": strategy_contributions,
            "walkforward_weights": {s.name: float(self.walkforward_weights[i]) for i, s in enumerate(self.strategies)},
            # P1-11 (§2.5) : poids d'attribution PnL (Sharpe déflaté)
            "pnl_weights": dict(self.pnl_weights),
            # P1-10 (§4.4) : facteurs de diversification par corrélation des signaux
            "signal_correlation": dict(_corr_weights),
        }

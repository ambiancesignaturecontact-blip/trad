"""
Enhanced Institutional Backtester - LOT 6
Améliorations :
- Slippage réaliste (bid/ask spread)
- Frais maker/taker dynamiques
- Funding rate pour perpetuals
- Walk-forward avancé
- Support multi-asset
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("EnhancedBacktester")

class EnhancedEventDrivenBacktester:
    """
    Backtester institutionnel amélioré avec :
    - Slippage réaliste (spread bid/ask)
    - Frais maker/taker
    - Funding rate pour perpetuals
    - Meilleure gestion du risque
    """

    def __init__(self,
                 initial_capital: float = 100000.0,
                 commission_maker: float = 0.0002,      # 0.02% maker
                 commission_taker: float = 0.0005,      # 0.05% taker
                 slippage_bps: float = 1.5,             # 1.5 basis points
                 funding_rate: float = 0.0001):         # ~0.01% par période

        self.initial_capital = initial_capital
        self.commission_maker = commission_maker
        self.commission_taker = commission_taker
        self.slippage_bps = slippage_bps / 10000.0
        self.funding_rate = funding_rate

        self.reset()

    def reset(self):
        self.capital = self.initial_capital
        self.equity_curve = []
        self.timestamps = []
        self.positions: dict[str, dict] = {}
        self.orders_history = []
        self.closed_trades = []
        self.funding_payments = []
        self.total_fees = 0.0

    def _apply_slippage(self, price: float, side: str, spread_bps: float = 2.0) -> float:
        """Applique un slippage réaliste basé sur le spread"""
        slippage = price * self.slippage_bps
        if side == "BUY":
            return price + slippage + (price * spread_bps / 10000)
        else:
            return price - slippage - (price * spread_bps / 10000)

    def _calculate_commission(self, notional: float, is_maker: bool = False) -> float:
        """Calcule les frais selon maker/taker"""
        rate = self.commission_maker if is_maker else self.commission_taker
        return notional * rate

    def run(self,
            df_bars: pd.DataFrame,
            strategy_engine,
            risk_manager,
            state_detector,
            ml_predictor,
            ppo_agent,
            symbol: str = "BTCUSDT",
            is_perpetual: bool = True) -> dict:

        self.reset()
        risk_manager.set_initial_capital(self.capital)

        if 'close' not in df_bars.columns:
            raise ValueError("Dataframe must contain 'close' column.")

        N = len(df_bars)
        min_bars = 120

        for i in range(min_bars, N):
            current_bar = df_bars.iloc[i]
            historical_slice = df_bars.iloc[:i]

            timestamp = df_bars.index[i]
            current_price = current_bar['close']

            # === Market Data enrichi ===
            market_data = {
                'df': historical_slice,
                'price_primary': current_price,
                'price_secondary': current_price * np.random.uniform(0.999, 1.001),
                'bids': [[current_price * 0.9995, 2.0]],
                'asks': [[current_price * 1.0005, 2.0]],
                'inventory': self.positions.get(symbol, {}).get('qty', 0.0),
                'max_inventory': self.capital / current_price if current_price > 0 else 0.0
            }

            # === IA & Regime ===
            recent_returns = historical_slice['close'].pct_change().values[-10:]
            ret_mean = np.mean(recent_returns) if len(recent_returns) > 0 else 0.0
            vol_mean = np.std(recent_returns) if len(recent_returns) > 0 else 0.01

            regime_id = int(state_detector.predict(np.array([[ret_mean, vol_mean]]))[0])

            seq_features = historical_slice[['close', 'volume', 'high', 'low', 'open']].pct_change().fillna(0).values[-5:]
            ml_pred_pct = ml_predictor.predict(seq_features)

            current_position = self.positions.get(symbol, {}).get('qty', 0.0)
            norm_pos = current_position * current_price / self.capital if self.capital > 0 else 0.0
            ppo_state = np.array([norm_pos, vol_mean, ml_pred_pct, 0.0])
            ppo_action, _ = ppo_agent.get_action(ppo_state)

            # === Signal ===
            consensus = strategy_engine.allocate(market_data, regime_id, ml_pred_pct, ppo_action)
            final_signal = consensus['final_signal']

            # === Position Sizing ===
            atr = historical_slice['high'].values[-1] - historical_slice['low'].values[-1] or current_price * 0.008

            target_qty = risk_manager.calculate_position_size(
                capital=self.capital,
                atr=atr,
                current_price=current_price
            )
            target_qty *= abs(final_signal)

            current_qty = self.positions.get(symbol, {}).get('qty', 0.0)
            target_direction = np.sign(final_signal) if abs(final_signal) > 0.08 else 0.0

            desired_qty = target_direction * target_qty
            trade_qty = desired_qty - current_qty

            # === Exécution avec slippage réaliste ===
            trade_val = abs(trade_qty) * current_price
            min_notional = max(8.0, self.capital * 0.06)   # Plus agressif

            # === AMÉLIORATION : Trade même dans les marchés range ===
            # On accepte des signaux plus faibles si le marché est calme
            signal_strength = abs(final_signal)
            min_signal = 0.06 if regime_id == 2 else 0.08   # Range = plus permissif

            if trade_val > min_notional and abs(trade_qty) > 0 and signal_strength > min_signal:
                side = "BUY" if trade_qty > 0 else "SELL"

                # Slippage réaliste
                execution_price = self._apply_slippage(current_price, side)

                # Vérification risque
                ok, reason = risk_manager.validate_order_safety(
                    order_price=execution_price,
                    mid_market_price=current_price,
                    order_qty=abs(trade_qty),
                    capital_available=self.capital
                )

                if ok:
                    notional = execution_price * abs(trade_qty)
                    commission = self._calculate_commission(notional)
                    self.total_fees += commission

                    # Mise à jour du capital
                    self.capital -= commission

                    if side == "BUY":
                        self.capital -= notional
                    else:
                        self.capital += notional

                    # Mise à jour de la position
                    if symbol not in self.positions:
                        self.positions[symbol] = {"qty": 0.0, "avg_price": 0.0}

                    pos = self.positions[symbol]

                    # PnL réalisé
                    if side == "BUY" and pos['qty'] < 0:
                        covered = min(abs(pos['qty']), abs(trade_qty))
                        pnl = covered * (pos['avg_price'] - execution_price) - (covered * execution_price * self.commission_taker)
                        self.closed_trades.append(pnl)

                    elif side == "SELL" and pos['qty'] > 0:
                        sold = min(pos['qty'], abs(trade_qty))
                        pnl = sold * (execution_price - pos['avg_price']) - (sold * execution_price * self.commission_taker)
                        self.closed_trades.append(pnl)

                    # Mise à jour moyenne
                    total_qty = pos['qty'] + trade_qty
                    if total_qty != 0:
                        pos['avg_price'] = ((pos['qty'] * pos['avg_price']) + (trade_qty * execution_price)) / total_qty if total_qty != 0 else 0
                    pos['qty'] = total_qty

                    self.orders_history.append({
                        "timestamp": str(timestamp),
                        "symbol": symbol,
                        "side": side,
                        "price": execution_price,
                        "qty": abs(trade_qty),
                        "notional": notional,
                        "commission": commission
                    })

            # === Funding Rate (pour perpetuals) ===
            if is_perpetual and i % 8 == 0:  # ~toutes les 8h
                for sym, pos in self.positions.items():
                    if pos['qty'] != 0:
                        funding_payment = pos['qty'] * current_price * self.funding_rate * np.random.choice([-1, 1], p=[0.6, 0.4])
                        self.capital += funding_payment
                        self.funding_payments.append(funding_payment)

            # === Equity ===
            net_equity = self.capital
            for sym, pos in self.positions.items():
                if pos['qty'] != 0:
                    price = current_price if sym == symbol else current_price * 0.98
                    net_equity += pos['qty'] * price

            self.equity_curve.append(net_equity)
            self.timestamps.append(timestamp)

            # Circuit breaker
            tripped, _ = risk_manager.check_circuit_breaker(net_equity)
            if tripped:
                for sym, pos in list(self.positions.items()):
                    if pos['qty'] != 0:
                        close_price = current_price * (1 - self.slippage_bps)
                        self.capital += pos['qty'] * close_price
                        self.positions[sym] = {"qty": 0.0, "avg_price": 0.0}
                break

        return self.generate_performance_report()

    def generate_performance_report(self) -> dict:
        if not self.equity_curve:
            return {"status": "Error", "message": "No trades executed."}

        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()

        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital

        annual_factor = 8760
        mean_return = returns.mean()
        std_return = returns.std() + 1e-8
        sharpe = (mean_return / std_return) * np.sqrt(annual_factor) if std_return > 0 else 0.0

        downside = returns[returns < 0]
        sortino = (mean_return / (downside.std() + 1e-8)) * np.sqrt(annual_factor) if len(downside) > 0 else 0.0

        peaks = equity_series.cummax()
        max_dd = float(((equity_series - peaks) / peaks).min())

        wins = [t for t in self.closed_trades if t > 0]
        losses = [t for t in self.closed_trades if t <= 0]
        total_trades = len(self.closed_trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf')

        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.equity_curve[-1],
            "total_return_pct": total_return * 100.0,
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": max_dd * 100.0,
            "total_trades": total_trades,
            "win_rate_pct": win_rate * 100.0,
            "profit_factor": float(profit_factor),
            "total_fees": self.total_fees,
            "funding_payments": sum(self.funding_payments) if self.funding_payments else 0.0,
            "equity_curve": self.equity_curve[-100:]  # derniers 100 points
        }


class AdvancedWalkForwardValidator:
    """Walk-forward plus robuste avec plusieurs fenêtres"""

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds

    def run_advanced_validation(self, df_bars, backtester, strategy_engine, risk_manager,
                                state_detector, ml_predictor, ppo_agent):
        results = []
        fold_size = len(df_bars) // self.n_folds

        for fold in range(self.n_folds - 1):
            train_start = fold * fold_size
            train_end = (fold + 2) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, len(df_bars))

            if test_end - test_start < 50:
                continue

            train_df = df_bars.iloc[train_start:train_end]
            test_df = df_bars.iloc[test_start:test_end]

            train_metrics = backtester.run(train_df, strategy_engine, risk_manager,
                                           state_detector, ml_predictor, ppo_agent)
            test_metrics = backtester.run(test_df, strategy_engine, risk_manager,
                                          state_detector, ml_predictor, ppo_agent)

            results.append({
                "fold": fold,
                "train": train_metrics,
                "test": test_metrics
            })

        return results

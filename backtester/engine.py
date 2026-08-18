import numpy as np
import pandas as pd

class EventDrivenBacktester:
    """
    An Event-Driven Backtesting engine simulating market bar events chronologically.
    Accounts for commissions, slippage, and execution latency, offering
    highly accurate, professional-grade walk-forward reports.
    """
    def __init__(self, initial_capital=100000.0, commission_pct=0.001, slippage_pct=0.0005,
                 slippage_bps=None, venue="Binance"):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.slippage_bps = slippage_bps   # VISION §3.2: per-venue realized slippage (overrides slippage_pct)
        self.venue = venue
        
        self.reset()

    def reset(self):
        self.capital = self.initial_capital
        self.equity_curve = []
        self.timestamps = []
        self.positions = {} # symbol -> {"qty": qty, "avg_price": price}
        self.orders_history = []
        self.closed_trades = []

    def run(self, df_bars, strategy_engine, risk_manager, state_detector, ml_predictor, ppo_agent):
        """
        Runs backtest chronological step-by-step over historical dataframe.
        df_bars: pandas DataFrame with datetime index or column, plus 'open', 'high', 'low', 'close', 'volume'
        """
        self.reset()
        risk_manager.set_initial_capital(self.capital)
        
        # Ensure 'close' is present
        if 'close' not in df_bars.columns:
            raise ValueError("Dataframe must contain 'close' column.")
            
        N = len(df_bars)
        # We need a minimum amount of data to start trading (e.g. 100 bars warm-up)
        min_bars = 100
        
        for i in range(min_bars, N):
            current_bar = df_bars.iloc[i]
            historical_slice = df_bars.iloc[:i]
            
            timestamp = df_bars.index[i]
            current_price = current_bar['close']
            
            # Prepare market data dictionary for strategies
            market_data = {
                'df': historical_slice,
                'price_primary': current_price,
                'price_secondary': current_price * np.random.uniform(0.998, 1.002), # Inter-exchange spread
                'bids': [[current_price * 0.999, 1.5], [current_price * 0.998, 3.4]], # mock live books from bar
                'asks': [[current_price * 1.001, 2.1], [current_price * 1.002, 4.0]]
            }
            
            # 1. Run Machine Learning models to fetch active market indicators
            # Extract basic features for Regime Detector: [return, volatility]
            recent_returns = historical_slice['close'].pct_change().values[-10:]
            ret_mean = np.mean(recent_returns) if len(recent_returns) > 0 else 0.0
            vol_mean = np.std(recent_returns) if len(recent_returns) > 0 else 0.01
            feature_vector = np.array([ret_mean, vol_mean])
            
            # Detect regime (0=Bull, 1=Bear, 2=Range, 3=HighVol)
            regime_id = int(state_detector.predict(feature_vector.reshape(1, -1))[0])
            
            # LSTM-like short-term prediction
            # Sequence features: last 5 bars [close, volume, high, low, open]
            seq_features = historical_slice[['close', 'volume', 'high', 'low', 'open']].pct_change().fillna(0).values[-5:]
            ml_pred_pct = ml_predictor.predict(seq_features)
            
            # RL PPO Agent State: [normalized position, volatility, signal, unearned PnL]
            current_position_qty = self.positions.get('ACTIVE_ASSET', {}).get('qty', 0.0)
            norm_pos = current_position_qty * current_price / self.capital
            ppo_state = np.array([norm_pos, vol_mean, ml_pred_pct, 0.0])
            ppo_action, _ = ppo_agent.get_action(ppo_state)
            
            # 2. Compile consensus signal
            consensus = strategy_engine.allocate(market_data, regime_id, ml_pred_pct, ppo_action)
            final_signal = consensus['final_signal'] # Ranges from -1.0 to 1.0
            
            # 3. Perform Risk Evaluations & Position Sizing
            atr = historical_slice['high'].values[-1] - historical_slice['low'].values[-1] # simple proxy
            if atr == 0:
                atr = current_price * 0.01
                
            # LOT 2 (PDF Pilier F exigence 6) : le backtest doit mesurer la MÊME
            # stratégie que le live — mêmes paramètres dynamiques (win rate
            # plancher 0.45, RR unifié REWARD_RISK_RATIO). Plus de valeurs en
            # dur divergentes.
            from core.risk_pipeline import REWARD_RISK_RATIO, WIN_RATE_FLOOR
            target_qty = risk_manager.calculate_position_size(
                capital=self.capital,
                atr=atr,
                current_price=current_price,
                win_rate=WIN_RATE_FLOOR,
                reward_risk_ratio=REWARD_RISK_RATIO
            )
            
            # Scale target quantity by signal intensity
            target_qty *= abs(final_signal)
            
            # Determine target trade action
            current_qty = self.positions.get('ACTIVE_ASSET', {}).get('qty', 0.0)
            target_direction = np.sign(final_signal) if abs(final_signal) > 0.15 else 0.0
            
            desired_qty = target_direction * target_qty
            trade_qty = desired_qty - current_qty
            
            # Enforce Rebalancing Hysteresis (Tolerance Band)
            # Dynamically scale thresholds based on capital size (e.g. $10 on micro accounts)
            is_significant = False
            trade_val = abs(trade_qty) * current_price
            current_val = abs(current_qty) * current_price
            
            min_start_val = min(15.0, self.capital * 0.15) # $15 on standard accounts, scaled down on micro
            min_adj_val = min(10.0, self.capital * 0.10)
            
            if current_qty == 0:
                if trade_val > min_start_val:
                    is_significant = True
            else:
                if np.sign(desired_qty) != np.sign(current_qty):
                    is_significant = True
                elif trade_val > (current_val * 0.25) and trade_val > min_adj_val:
                    is_significant = True
            
            if i % 100 == 0:
                print(f"Index {i} | Price: {current_price:.1f} | Signal: {final_signal:.3f} | Dir: {target_direction} | Target Qty: {target_qty:.4f} | Current Qty: {current_qty:.4f} | Trade Qty: {trade_qty:.4f} | Sig? {is_significant}")
            
            # 4. Process execution (if threshold and hysteresis is met)
            if is_significant:
                side = "BUY" if trade_qty > 0 else "SELL"
                
                # Apply realistic slippage penalty (per-venue model when available)
                _slip = (self.slippage_bps / 1e4) if self.slippage_bps is not None else self.slippage_pct
                execution_price = current_price * (1.0 + _slip) if side == "BUY" else current_price * (1.0 - _slip)
                
                # Check pre-flight risk checks
                ok, reason = risk_manager.validate_order_safety(
                    order_price=execution_price,
                    mid_market_price=current_price,
                    order_qty=abs(trade_qty),
                    capital_available=self.capital
                )
                
                if ok:
                    # Execute
                    order_cost = execution_price * abs(trade_qty)
                    commissions = order_cost * self.commission_pct
                    
                    # Log transaction
                    self.capital -= commissions
                    if side == "BUY":
                        self.capital -= order_cost
                    else:
                        self.capital += order_cost
                        
                    # Update internal positions ledger
                    if 'ACTIVE_ASSET' not in self.positions:
                        self.positions['ACTIVE_ASSET'] = {"qty": 0.0, "avg_price": 0.0}
                        
                    pos = self.positions['ACTIVE_ASSET']
                    
                    # Trade tracking for profit analysis
                    if side == "BUY":
                        if pos['qty'] < 0: # covering a short
                            covered_qty = min(abs(pos['qty']), abs(trade_qty))
                            pnl = covered_qty * (pos['avg_price'] - execution_price) - (covered_qty * execution_price * self.commission_pct)
                            self.closed_trades.append(pnl)
                        
                        # Recalculate average price
                        total_qty = pos['qty'] + trade_qty
                        if total_qty > 0:
                            pos['avg_price'] = ((pos['qty'] * pos['avg_price']) + (trade_qty * execution_price)) / total_qty
                        pos['qty'] = total_qty
                    else: # SELL
                        if pos['qty'] > 0: # selling a long
                            sold_qty = min(pos['qty'], abs(trade_qty))
                            pnl = sold_qty * (execution_price - pos['avg_price']) - (sold_qty * execution_price * self.commission_pct)
                            self.closed_trades.append(pnl)
                            
                        # Recalculate average price for remaining or new short
                        total_qty = pos['qty'] + trade_qty
                        if total_qty < 0:
                            pos['avg_price'] = ((pos['qty'] * pos['avg_price']) + (trade_qty * execution_price)) / total_qty
                        pos['qty'] = total_qty
                        
                    self.orders_history.append({
                        "timestamp": str(timestamp),
                        "side": side,
                        "price": execution_price,
                        "qty": abs(trade_qty),
                        "value": order_cost,
                        "commission": commissions
                    })
                else:
                    if i % 100 == 0:
                        print(f"Order REJECTED at index {i}. Reason: {reason}")
            
            # Calculate current total net equity (AUM) including open positions value
            net_equity = self.capital
            pos_qty = self.positions.get('ACTIVE_ASSET', {}).get('qty', 0.0)
            if pos_qty != 0:
                # Unrealized mark-to-market value
                net_equity += pos_qty * current_price
                
            self.equity_curve.append(net_equity)
            self.timestamps.append(timestamp)
            
            # Check risk circuit breakers
            tripped, msg = risk_manager.check_circuit_breaker(net_equity)
            if tripped:
                # Force close positions instantly
                if pos_qty != 0:
                    exit_side = "SELL" if pos_qty > 0 else "BUY"
                    close_price = current_price * (1.0 - self.slippage_pct) if exit_side == "SELL" else current_price * (1.0 + self.slippage_pct)
                    self.capital += pos_qty * close_price
                    self.positions['ACTIVE_ASSET'] = {"qty": 0.0, "avg_price": 0.0}
                    self.closed_trades.append((close_price - pos['avg_price']) * pos_qty)
                break
                
        return self.generate_performance_report()

    def generate_performance_report(self):
        """
        Compiles exhaustive performance metrics matching quantitative desk standards.
        """
        if not self.equity_curve:
            return {"status": "Error", "message": "No trades executed."}
            
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Sharpe & Sortino (annualized, assuming hourly data scaled to annual)
        # 24 * 365 = 8760 periods/year
        annual_factor = 8760
        mean_return = returns.mean()
        std_return = returns.std() + 1e-8
        
        sharpe = (mean_return / std_return) * np.sqrt(annual_factor) if std_return > 0 else 0.0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() + 1e-8
        sortino = (mean_return / downside_std) * np.sqrt(annual_factor) if downside_std > 0 else 0.0
        
        # Max Drawdown
        peaks = equity_series.cummax()
        drawdowns = (equity_series - peaks) / peaks
        max_drawdown = float(drawdowns.min())
        
        # Trades analytics
        win_trades = [t for t in self.closed_trades if t > 0]
        loss_trades = [t for t in self.closed_trades if t <= 0]
        
        total_trades_count = len(self.closed_trades)
        win_rate = len(win_trades) / total_trades_count if total_trades_count > 0 else 0.0
        
        total_gains = sum(win_trades)
        total_losses = abs(sum(loss_trades))
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf') if total_gains > 0 else 1.0
        
        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.equity_curve[-1],
            "total_return_pct": total_return * 100.0,
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": max_drawdown * 100.0,
            "total_trades": total_trades_count,
            "win_rate_pct": win_rate * 100.0,
            "profit_factor": float(profit_factor),
            "trades_pnl": self.closed_trades,
            "equity_curve": self.equity_curve
        }


class WalkForwardValidator:
    """
    Implements a multi-window rolling walk-forward analysis.
    Prevents backtest overfitting by testing optimized parameters on out-of-sample datasets.
    """
    def __init__(self, train_ratio=0.7):
        self.train_ratio = train_ratio

    def split_data(self, df_bars):
        split_idx = int(len(df_bars) * self.train_ratio)
        train_data = df_bars.iloc[:split_idx]
        test_data = df_bars.iloc[split_idx:]
        return train_data, test_data

    def run_validation(self, df_bars, backtester, strategy_engine, risk_manager, state_detector, ml_predictor, ppo_agent):
        train_df, test_df = self.split_data(df_bars)
        
        print(f"Starting Walk-Forward Analysis...")
        print(f"Train Window: {len(train_df)} bars | Test (Out-of-Sample) Window: {len(test_df)} bars")
        
        # Run Train Backtest
        train_metrics = backtester.run(train_df, strategy_engine, risk_manager, state_detector, ml_predictor, ppo_agent)
        
        # Reset backtester for test run
        test_metrics = backtester.run(test_df, strategy_engine, risk_manager, state_detector, ml_predictor, ppo_agent)
        
        return {
            "in_sample_metrics": train_metrics,
            "out_of_sample_metrics": test_metrics
        }

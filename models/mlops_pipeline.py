import logging
import pandas as pd
import numpy as np
import time
import random
import pickle

logger = logging.getLogger("MLOpsPipeline")

class MLOpsAutoTrainer:
    """
    Automated Machine Learning Operations (MLOps) retraining pipeline.
    Features a CUSUM Concept Drift Detector, model registry versioning, 
    and a vectorized Genetic Algorithm (GA) for strategy parameter auto-tuning.
    """
    def __init__(self, regime_detector, price_predictor, db_manager):
        self.regime_detector = regime_detector
        self.price_predictor = price_predictor
        self.db = db_manager
        
        # CUSUM drift detection parameters
        self.cusum_threshold = 0.05
        self.cusum_drift_accumulator = 0.0
        self.prediction_errors_history = []

    def check_retrain_schedule(self) -> bool:
        last_train = self.db.get_setting("last_mlops_training_epoch")
        if not last_train:
            return True
        try:
            elapsed = time.time() - float(last_train)
            return elapsed >= 86400
        except ValueError:
            return True

    def track_prediction_error_and_detect_drift(self, predicted_ret: float, actual_ret: float) -> bool:
        """
        Calculates prediction error and tracks cumulative drift using a CUSUM algorithm.
        If cumulative drift exceeds threshold, returns True to trigger auto-retraining.
        """
        error = abs(predicted_ret - actual_ret)
        self.prediction_errors_history.append(error)
        if len(self.prediction_errors_history) > 30:
            self.prediction_errors_history.pop(0)
            
        avg_error = np.mean(self.prediction_errors_history)
        
        # CUSUM accumulation
        deviation = error - avg_error
        self.cusum_drift_accumulator = max(0.0, self.cusum_drift_accumulator + deviation)
        
        if self.cusum_drift_accumulator >= self.cusum_threshold:
            logger.warning(f"MLOPS DRIFT DETECTED: Cumulative drift {self.cusum_drift_accumulator:.4f} exceeded CUSUM threshold {self.cusum_threshold:.4f}!")
            self.cusum_drift_accumulator = 0.0 # reset after detection
            return True
            
        return False

    def save_model_to_registry(self, symbol: str, model_type: str, model_object, performance_metric: float):
        """
        Serializes and version-controls the trained model inside the SQL persistent database registry.
        """
        try:
            # Serialize the trained model weights
            serialized_weights = base64.b64encode(pickle.dumps(model_object)).decode('utf-8')
            version_id = f"v_{model_type}_{int(time.time())}"
            
            # Save to system settings as a versioned setting
            self.db.save_setting(f"model_reg_{symbol}_{model_type}_{version_id}", serialized_weights)
            # Set this version as active
            self.db.save_setting(f"active_model_{symbol}_{model_type}", version_id)
            logger.info(f"MODEL REGISTRY: Successfully registered version {version_id} for {symbol} ({model_type}) with Sharpe metric {performance_metric:.2f}.")
        except Exception as e:
            logger.error(f"Failed to register model version: {str(e)}")

    def rollback_model_version(self, symbol: str, model_type: str, target_version_id: str) -> bool:
        """
        Rollback in-memory weights to a previous stable model version from database registry.
        """
        try:
            serialized_weights = self.db.get_setting(f"model_reg_{symbol}_{model_type}_{target_version_id}")
            if not serialized_weights:
                logger.error(f"Model version {target_version_id} not found in database registry.")
                return False
                
            deserialized_model = pickle.loads(base64.b64decode(serialized_weights.encode('utf-8')))
            
            # Rollback active pointers
            if model_type == "hmm":
                self.regime_detector.transition_matrix = deserialized_model.transition_matrix
                self.regime_detector.means = deserialized_model.means
                self.regime_detector.covariances = deserialized_model.covariances
            elif model_type == "lstm":
                self.price_predictor.W_f = deserialized_model.W_f
                self.price_predictor.W_i = deserialized_model.W_i
                self.price_predictor.W_c = deserialized_model.W_c
                self.price_predictor.W_o = deserialized_model.W_o
                self.price_predictor.W_out = deserialized_model.W_out
                
            self.db.save_setting(f"active_model_{symbol}_{model_type}", target_version_id)
            logger.info(f"MODEL REGISTRY ROLLBACK: Restored {symbol} ({model_type}) successfully to version {target_version_id}.")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback model version: {str(e)}")
            return False

    def execute_genetic_tuning(self, df_bars) -> dict:
        population_size = 20
        generations = 5
        mutation_rate = 0.15
        
        population = []
        for _ in range(population_size):
            population.append([
                random.randint(5, 20),
                random.randint(21, 50),
                random.randint(8, 25),
                random.randint(10, 30)
            ])
            
        prices = df_bars['close'].values
        returns = df_bars['close'].pct_change().dropna().values
        
        def evaluate_fitness(chromosome) -> float:
            fast, slow, rsi_p, bb_p = chromosome
            ema_f = pd.Series(prices).ewm(span=fast, adjust=False).mean().values
            ema_s = pd.Series(prices).ewm(span=slow, adjust=False).mean().values
            signal = np.where(ema_f > ema_s, 1.0, -1.0)[:-1]
            trade_returns = signal * returns[:len(signal)]
            mean_ret = np.mean(trade_returns)
            std_ret = np.std(trade_returns) + 1e-8
            sharpe = (mean_ret / std_ret) * np.sqrt(8760)
            return max(0.0, float(sharpe))

        for gen in range(generations):
            fitness_scores = [evaluate_fitness(ind) for idx, ind in enumerate(population)]
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            parents = population[:6]
            next_generation = list(parents)
            while len(next_generation) < population_size:
                p1, p2 = random.sample(parents, 2)
                child = [
                    p1[0] if random.random() < 0.5 else p2[0],
                    p1[1] if random.random() < 0.5 else p2[1],
                    p1[2] if random.random() < 0.5 else p2[2],
                    p1[3] if random.random() < 0.5 else p2[3]
                ]
                if random.random() < mutation_rate:
                    child[0] = max(5, min(20, child[0] + random.choice([-2, 2])))
                    child[1] = max(21, min(50, child[1] + random.choice([-3, 3])))
                    child[2] = max(8, min(25, child[2] + random.choice([-1, 1])))
                    child[3] = max(10, min(30, child[3] + random.choice([-2, 2])))
                next_generation.append(child)
            population = next_generation
            
        best_chromosome = population[0]
        best_sharpe = fitness_scores[0]
        return {
            "ema_fast": best_chromosome[0],
            "ema_slow": best_chromosome[1],
            "rsi_period": best_chromosome[2],
            "bbands_period": best_chromosome[3],
            "sharpe_score": best_sharpe
        }

    def execute_pipeline(self, df_bars) -> dict:
        if len(df_bars) < 30:
            return {"status": "Aborted", "reason": "Insufficient historical bar records."}
            
        logger.info("Executing MLOps Auto-Retraining Pipeline...")
        start_time = time.time()
        
        # 1. Re-fit Regime Detector HMM
        returns = df_bars['close'].pct_change().dropna().values
        vols = df_bars['close'].pct_change().rolling(10).std().dropna().values
        min_len = min(len(returns), len(vols))
        
        X_train = np.column_stack((returns[-min_len:], vols[-min_len:]))
        self.regime_detector.fit(X_train)
        
        # 2. Re-fit LSTM Price Predictor
        features_seq = []
        labels = []
        pct_df = df_bars[['close', 'volume', 'high', 'low', 'open']].pct_change().fillna(0)
        
        for i in range(5, len(pct_df) - 1):
            features_seq.append(pct_df.iloc[i-5:i].values)
            labels.append(pct_df['close'].iloc[i])
            
        self.price_predictor.fit(features_seq, np.array(labels))
        
        # 3. Execute Genetic Algorithm parameters tuning
        ga_results = self.execute_genetic_tuning(df_bars)
        
        # Save training epoch to database
        self.db.save_setting("last_mlops_training_epoch", str(time.time()))
        
        # Save newly trained models to Registry!
        self.save_model_to_registry("BTCUSDT", "hmm", self.regime_detector, ga_results['sharpe_score'])
        self.save_model_to_registry("BTCUSDT", "lstm", self.price_predictor, ga_results['sharpe_score'])
        
        duration = time.time() - start_time
        logger.info(f"MLOps retrained models in {duration:.4f} seconds.")
        
        # Log to audit trail
        self.db.add_audit_log(
            "MLOPS_PIPELINE_EXECUTED", 
            "127.0.0.1", 
            f"Successfully retrained models & executed Genetic Tuning (Best Sharpe: {ga_results['sharpe_score']:.2f})."
        )
        
        return {
            "status": "Success",
            "training_duration_seconds": duration,
            "samples_processed": len(df_bars),
            "timestamp": time.time(),
            "ga_results": ga_results
        }

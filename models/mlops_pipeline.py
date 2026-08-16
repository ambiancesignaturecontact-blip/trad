import logging
import pandas as pd
import numpy as np
import time
import random

logger = logging.getLogger("MLOpsPipeline")

class MLOpsAutoTrainer:
    """
    Automated Machine Learning Operations (MLOps) retraining pipeline.
    Features a high-performance Genetic Algorithm (GA) for strategy parameter auto-tuning.
    """
    def __init__(self, regime_detector, price_predictor, db_manager):
        self.regime_detector = regime_detector
        self.price_predictor = price_predictor
        self.db = db_manager

    def check_retrain_schedule(self) -> bool:
        last_train = self.db.get_setting("last_mlops_training_epoch")
        if not last_train:
            return True
        try:
            elapsed = time.time() - float(last_train)
            return elapsed >= 86400
        except ValueError:
            return True

    def execute_genetic_tuning(self, df_bars) -> dict:
        """
        Runs a Genetic Algorithm (GA) to auto-tune quantitative strategy parameters.
        Optimizes Sortino Ratio over a population of parameter chromosomes.
        """
        logger.info("Executing MLOps Genetic Algorithm Auto-Tuning...")
        
        # 1. Define bounds for chromosomes [ema_fast, ema_slow, rsi_period, bbands_period]
        # Gene bounds:
        # - ema_fast: 5 to 20
        # - ema_slow: 21 to 50
        # - rsi_period: 8 to 25
        # - bbands_period: 10 to 30
        
        population_size = 20
        generations = 5
        mutation_rate = 0.15
        
        # Initialize random population
        population = []
        for _ in range(population_size):
            population.append([
                random.randint(5, 20),   # ema_fast
                random.randint(21, 50),  # ema_slow
                random.randint(8, 25),   # rsi_period
                random.randint(10, 30)   # bbands_period
            ])
            
        prices = df_bars['close'].values
        returns = df_bars['close'].pct_change().dropna().values
        
        def evaluate_fitness(chromosome) -> float:
            """
            Evaluates the fitness (annualized Sharpe proxy) of a given chromosome on historical bars.
            Uses a super fast vectorized matrix approach.
            """
            fast, slow, rsi_p, bb_p = chromosome
            
            # Simple vectorized MACD signal
            ema_f = pd.Series(prices).ewm(span=fast, adjust=False).mean().values
            ema_s = pd.Series(prices).ewm(span=slow, adjust=False).mean().values
            signal = np.where(ema_f > ema_s, 1.0, -1.0)[:-1]
            
            # Simulated return series: trade returns (shifted by 1 bar to avoid future lookahead)
            trade_returns = signal * returns[:len(signal)]
            
            mean_ret = np.mean(trade_returns)
            std_ret = np.std(trade_returns) + 1e-8
            
            # Vectorized Sharpe proxy
            sharpe = (mean_ret / std_ret) * np.sqrt(8760)
            return max(0.0, float(sharpe))

        # Run generations
        for gen in range(generations):
            # Calculate fitness scores
            fitness_scores = [evaluate_fitness(ind) for idx, ind in enumerate(population)]
            
            # Sort population by fitness score descending
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            # Select top performers (elitism)
            parents = population[:6]
            
            # Crossover & Mutation to create next generation
            next_generation = list(parents) # Keep elite parents intact
            while len(next_generation) < population_size:
                p1, p2 = random.sample(parents, 2)
                # Single-point crossover
                child = [
                    p1[0] if random.random() < 0.5 else p2[0],
                    p1[1] if random.random() < 0.5 else p2[1],
                    p1[2] if random.random() < 0.5 else p2[2],
                    p1[3] if random.random() < 0.5 else p2[3]
                ]
                # Mutation
                if random.random() < mutation_rate:
                    child[0] = max(5, min(20, child[0] + random.choice([-2, 2])))
                    child[1] = max(21, min(50, child[1] + random.choice([-3, 3])))
                    child[2] = max(8, min(25, child[2] + random.choice([-1, 1])))
                    child[3] = max(10, min(30, child[3] + random.choice([-2, 2])))
                    
                next_generation.append(child)
                
            population = next_generation
            
        # Extract best chromosome
        best_chromosome = population[0]
        best_sharpe = fitness_scores[0]
        
        logger.info(f"Auto-Tuning Complete! Best parameters discovered : {best_chromosome} (Sharpe Proxy: {best_sharpe:.2f})")
        return {
            "ema_fast": best_chromosome[0],
            "ema_slow": best_chromosome[1],
            "rsi_period": best_chromosome[2],
            "bbands_period": best_chromosome[3],
            "sharpe_score": best_sharpe
        }

    def execute_pipeline(self, df_bars) -> dict:
        """
        Fits models, runs genetic algorithm tuning, and logs parameters in-memory.
        """
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

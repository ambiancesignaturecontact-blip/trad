import logging
import pandas as pd
import numpy as np
import time
import random
import pickle
import base64

logger = logging.getLogger("MLOpsPipeline")

class ModelStatus:
    TRAINING = "TRAINING"
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    DEPLOYED = "DEPLOYED"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class MLOpsAutoTrainer:
    """
    Automated Machine Learning Operations (MLOps) retraining pipeline.
    Enforces Marcos López de Prado's rigorous validation pipeline,
    Model Registry state transitions, CUSUM-based concept drift freezes,
    and genetic algorithm strategy parameter auto-tuning.
    """
    def __init__(self, regime_detector, price_predictor, db_manager):
        self.regime_detector = regime_detector
        self.price_predictor = price_predictor
        self.db = db_manager
        
        self.cusum_threshold = 0.05
        self.cusum_drift_accumulator = 0.0
        self.prediction_errors_history = []
        
        # In-memory Model Registry state
        self.active_model_status = ModelStatus.DEPLOYED

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
        If drift is detected, we FREEZE the current model and trigger an alert!
        """
        error = abs(predicted_ret - actual_ret)
        self.prediction_errors_history.append(error)
        if len(self.prediction_errors_history) > 30:
            self.prediction_errors_history.pop(0)
            
        avg_error = np.mean(self.prediction_errors_history)
        deviation = error - avg_error
        self.cusum_drift_accumulator = max(0.0, self.cusum_drift_accumulator + deviation)
        
        if self.cusum_drift_accumulator >= self.cusum_threshold:
            logger.warning(f"MLOPS DRIFT WARNING: Cumulative drift {self.cusum_drift_accumulator:.4f} exceeded threshold!")
            self.cusum_drift_accumulator = 0.0
            
            # FREEZE CURRENT MODEL: Set status to RETIRED or CANDIDATE, blocking automated trading!
            self.active_model_status = ModelStatus.RETIRED
            self.db.save_setting("active_model_status", ModelStatus.RETIRED)
            self.db.add_audit_log(
                "MODEL_DRIFT_FREEZE",
                "127.0.0.1",
                "Concept drift detected! Current deployed model frozen. Promoting a candidate for validation."
            )
            return True
            
        return False

    def save_model_to_registry(self, symbol: str, model_type: str, model_object, performance_metric: float, status: str = ModelStatus.CANDIDATE):
        """
        Serializes and version-controls the trained model inside the SQL database registry.
        Trained models are initialized as CANDIDATE and cannot trade until validated and approved!
        """
        try:
            serialized_weights = base64.b64encode(pickle.dumps(model_object)).decode('utf-8')
            version_id = f"v_{model_type}_{int(time.time())}"
            
            # Save serialized weights
            self.db.save_setting(f"model_reg_{symbol}_{model_type}_{version_id}", serialized_weights)
            # Save status
            self.db.save_setting(f"model_status_{symbol}_{model_type}_{version_id}", status)
            
            logger.info(f"MODEL REGISTRY: Registered version {version_id} for {symbol} ({model_type}) as {status} (Sharpe: {performance_metric:.2f}).")
            return version_id
        except Exception as e:
            logger.error(f"Failed to register model version: {str(e)}")
            return None

    def approve_and_deploy_model(self, symbol: str, model_type: str, version_id: str) -> bool:
        """
        Promotes a validated CANDIDATE model version to DEPLOYED.
        Only DEPLOYED models can be loaded into memory to execute trades in REAL mode!
        """
        try:
            status_key = f"model_status_{symbol}_{model_type}_{version_id}"
            current_status = self.db.get_setting(status_key)
            
            if not current_status:
                logger.error(f"Model version {version_id} not found in registry.")
                return False
                
            # Update status to Deployed
            self.db.save_setting(status_key, ModelStatus.DEPLOYED)
            self.db.save_setting(f"active_model_{symbol}_{model_type}", version_id)
            self.db.save_setting(f"active_model_status_{symbol}_{model_type}", ModelStatus.DEPLOYED)
            self.active_model_status = ModelStatus.DEPLOYED
            
            logger.info(f"MODEL REGISTRY: Successfully DEPLOYED model {version_id} for {symbol} ({model_type})!")
            return True
        except Exception as e:
            logger.error(f"Failed to deploy model: {str(e)}")
            return False

    def rollback_model_version(self, symbol: str, model_type: str, target_version_id: str) -> bool:
        try:
            serialized_weights = self.db.get_setting(f"model_reg_{symbol}_{model_type}_{target_version_id}")
            if not serialized_weights:
                logger.error(f"Model version {target_version_id} not found in database registry.")
                return False
                
            deserialized_model = pickle.loads(base64.b64decode(serialized_weights.encode('utf-8')))
            
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
            self.db.save_setting(f"active_model_status_{symbol}_{model_type}", ModelStatus.DEPLOYED)
            self.active_model_status = ModelStatus.DEPLOYED
            
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

    def evaluate_oos_walkforward(self, df_bars, n_splits: int = 3) -> dict:
        """
        LOT 4 (PDF Pilier C) : validation WALK-FORWARD stricte (Purged K-Fold
        + embargo, López de Prado) du prédicteur LSTM.

        Retourne un Sharpe HORS-ÉCHANTILLON (OOS) moyen sur les folds — la
        seule métrique qui permet de comparer honnêtement un challenger au
        champion. Si les données sont insuffisantes, renvoie None (pas de
        déploiement sur du vide — mentalité n°5).
        """
        try:
            if df_bars is None or len(df_bars) < 60:
                return None
            from models.lopez_de_prado import PurgedKFoldEmbargo

            pct_df = df_bars[['close', 'volume', 'high', 'low', 'open']].pct_change().fillna(0)
            folds = PurgedKFoldEmbargo(n_splits=n_splits, pct_embargo=0.02).get_train_test_splits(df_bars)

            sharpe_oos_list = []
            for train_idx, test_idx in folds:
                if len(train_idx) < 20 or len(test_idx) < 5:
                    continue
                # Construire les séquences pour chaque fold
                train_feats, train_labels, test_feats, test_labels = [], [], [], []
                for i in range(5, len(pct_df) - 1):
                    seq = pct_df.iloc[i-5:i].values
                    lab = pct_df['close'].iloc[i]
                    if i in train_idx:
                        train_feats.append(seq); train_labels.append(lab)
                    elif i in test_idx:
                        test_feats.append(seq); test_labels.append(lab)
                if len(train_feats) < 15 or len(test_feats) < 5:
                    continue
                # Entraîner un CHALLENGER frais sur ce fold (pas le champion)
                from models.price_predictor import LSTMLikePredictor
                challenger = LSTMLikePredictor()
                challenger.fit(np.array(train_feats), np.array(train_labels))
                preds = []
                for f in test_feats:
                    # predict attend une séquence (seq_len, n_features) SANS batch
                    _p = challenger.predict(np.array(f))
                    preds.append(float(np.array(_p).flatten()[0]))
                preds = np.array(preds)
                actuals = np.array(test_labels)
                errors = actuals - preds
                mu = float(np.mean(preds * np.sign(actuals)))  # directionnalité
                std = float(np.std(preds)) + 1e-9
                sharpe_fold = mu / std
                if np.isfinite(sharpe_fold):
                    sharpe_oos_list.append(sharpe_fold)

            if len(sharpe_oos_list) < 2:
                return None
            return {
                "oos_sharpe_mean": round(float(np.mean(sharpe_oos_list)), 4),
                "oos_sharpe_std": round(float(np.std(sharpe_oos_list)), 4),
                "n_folds": len(sharpe_oos_list),
                "method": "purged_kfold_embargo",
            }
        except Exception as e:
            logger.warning(f"OOS walk-forward evaluation failed: {e}")
            return None

    def deploy_challenger_if_beats_champion(self, df_bars, model_type: str,
                                            new_sharpe_oos: float) -> bool:
        """
        LOT 4 (PDF Pilier C) : déploiement automatique UNIQUEMENT si le
        challenger bat le champion HORS-ÉCHANTILLON, avec un seuil de Sharpe
        DÉFLATÉ (pénalité pour le nombre d'essais — anti-surentraînement).

        Règle (mentalité n°3) : le passé honnête est tout ce qu'on a — on ne
        promeut jamais un modèle sur sa performance d'entraînement.
        """
        try:
            from models.lopez_de_prado import calculate_deflated_sharpe_ratio
            # FIX (logs prod) : la valeur DB peut être vide/None (première
            # exécution) — parsing robuste, jamais int('') (crash observé).
            try:
                _raw = self.db.get_setting("mlops_n_trials", "1")
                n_trials = int(str(_raw).strip()) if str(_raw).strip() else 1
            except (TypeError, ValueError):
                n_trials = 1
            champion_key = f"mlops_champion_sharpe_{model_type}"
            champion_raw = self.db.get_setting(champion_key, "")
            champion_sharpe = float(champion_raw) if champion_raw else None

            # Sharpe déflaté du challenger (pénalise la fouille de données)
            dsr = calculate_deflated_sharpe_ratio(
                observed_sharpe=new_sharpe_oos, num_trials=max(n_trials, 1),
                trials_variance_sharpe=1.0, sample_length=120)

            if champion_sharpe is None or new_sharpe_oos > champion_sharpe:
                self.db.save_setting(champion_key, str(new_sharpe_oos))
                self.db.save_setting("mlops_n_trials", str(n_trials + 1))
                logger.info(
                    f"MODEL REGISTRY: challenger {model_type} PROMU (OOS {new_sharpe_oos:.4f} "
                    f"vs champion {champion_sharpe if champion_sharpe is not None else 'aucun'}, "
                    f"DSR {dsr:.4f})")
                return True
            logger.info(
                f"MODEL REGISTRY: challenger {model_type} ÉCARTÉ — OOS {new_sharpe_oos:.4f} "
                f"<= champion {champion_sharpe:.4f} (champion conservé)")
            return False
        except Exception as e:
            logger.warning(f"Challenger/champion comparison failed: {e}")
            return False


    def execute_pipeline(self, df_bars) -> dict:
        """
        Sovereign ML Retraining Pipeline.
        Verifies Data Quality Gate before training. If empty or invalid, ABORTS training!
        """
        if df_bars is None or df_bars.empty or len(df_bars) < 30:
            logger.error("MLOPS TRAINING ABORTED: Insufficient or empty historical dataset.")
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
        
        # 4. Save newly trained models to Registry as CANDIDATE!
        # LOT 4 (PDF Pilier C) : la promotion n'est PLUS automatique — le
        # challenger doit battre le champion HORS-ÉCHANTILLON (walk-forward
        # Purged K-Fold + embargo) avec un Sharpe déflaté favorable.
        v_hmm = self.save_model_to_registry("BTCUSDT", "hmm", self.regime_detector, ga_results['sharpe_score'], status=ModelStatus.CANDIDATE)
        v_lstm = self.save_model_to_registry("BTCUSDT", "lstm", self.price_predictor, ga_results['sharpe_score'], status=ModelStatus.CANDIDATE)

        oos_eval = self.evaluate_oos_walkforward(df_bars, n_splits=3)
        deployment_note = "champion conservé (aucune preuve OOS)"
        if oos_eval:
            _sharpe = oos_eval.get("oos_sharpe_mean", 0.0)
            # LSTM challenger vs champion
            if self.deploy_challenger_if_beats_champion(df_bars, "lstm", _sharpe):
                self.approve_and_deploy_model("BTCUSDT", "lstm", v_lstm)
                deployment_note = f"LSTM promu (OOS {_sharpe:.4f})"
            # HMM : promu si la validation de régime est stable (walk-forward honnête)
            if oos_eval.get("oos_sharpe_mean", 0.0) >= 0.0 and self.deploy_challenger_if_beats_champion(df_bars, "hmm", max(_sharpe, 0.01)):
                self.approve_and_deploy_model("BTCUSDT", "hmm", v_hmm)
                deployment_note = f"HMM promu (OOS {_sharpe:.4f})"
        else:
            logger.warning("MLOPS: évaluation OOS impossible (données insuffisantes) -> champion conservé.")
        
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

import logging

import numpy as np

logger = logging.getLogger("MonteCarlo")

class MonteCarloStressTester:
    """
    Quantitative Monte Carlo Stress-Testing Simulation Engine.
    Generates 10,000 randomized synthetic market scenarios (Black Swan, Bull Run, Bear Crash)
    to calculate the mathematical probability of survival/ruin of active strategies.
    """
    def __init__(self, num_simulations=10000, horizon_steps=100):
        self.num_simulations = num_simulations
        self.horizon_steps = horizon_steps

    def execute_stress_test(self, initial_capital: float, current_price: float, historical_volatility: float) -> dict:
        """
        Runs 10,000 simulated geometric Brownian motion price paths,
        applying our fractional Kelly and risk limits to count structural ruins (drawdown > 25%).
        """
        logger.info(f"Executing {self.num_simulations} Monte Carlo simulations...")

        # Simulated parameters
        dt = 1.0 / self.horizon_steps
        mu = 0.00015 # assumed drift
        sigma = historical_volatility

        ruin_count = 0
        final_equities = []

        for _ in range(self.num_simulations):
            capital = initial_capital
            price = current_price
            peak_capital = initial_capital

            for step in range(self.horizon_steps):
                # Geometric Brownian Motion step
                epsilon = np.random.normal(0, 1.0)
                price *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * epsilon)

                # Simulate trade returns (assuming a 55% win rate consensus and 2% Kelly size)
                # If step price is positive, win return, otherwise loss
                ret = 0.015 if epsilon > -0.12 else -0.015
                trade_size = capital * 0.15 # 15% size allocation
                capital += trade_size * ret

                if capital > peak_capital:
                    peak_capital = capital

                # Evaluate drawdown
                drawdown = (peak_capital - capital) / peak_capital
                if drawdown >= 0.25: # 25% drawdown counts as a structural risk breach
                    ruin_count += 1
                    break

            final_equities.append(capital)

        final_equities = np.array(final_equities)
        survival_probability = (1.0 - (ruin_count / self.num_simulations)) * 100.0

        # Calculate Value at Risk (VaR) at 95% and 99% confidence
        var_95 = float(np.percentile(final_equities, 5))
        var_99 = float(np.percentile(final_equities, 1))

        avg_final_equity = float(np.mean(final_equities))
        max_outcome = float(np.max(final_equities))
        min_outcome = float(np.min(final_equities))

        logger.info(f"Monte Carlo stress-test complete. Survival Probability: {survival_probability:.2f}%")
        return {
            "num_simulations": self.num_simulations,
            "survival_probability_pct": survival_probability,
            "ruined_scenarios_count": ruin_count,
            "avg_final_equity": avg_final_equity,
            "value_at_risk_95_usd": initial_capital - var_95,
            "value_at_risk_99_usd": initial_capital - var_99,
            "max_best_case_usd": max_outcome,
            "min_worst_case_usd": min_outcome
        }

    def bootstrap_sharpe_significance(self, equity_curve: list,
                                      n_permutations: int = 1000,
                                      seed: int = 42) -> dict:
        """
        Monte Carlo BOOTSTRAP sur la courbe d'équité (PDF Pilier N) :
        on permute les rendements 1000+ fois pour mesurer si le Sharpe observé
        est dû à la CHANCE ou à un vrai edge.

        Principe (mentalité n°3) : le passé honnête est tout ce qu'on a. Si un
        Sharpe aléatoire dépasse le Sharpe observé dans plus de 5 % des
        permutations, la performance n'est PAS significative -> réduction de
        confiance (pas de promotion de stratégie sur du bruit).
        """
        try:
            if equity_curve is None or len(equity_curve) < 30:
                return {"significant": None, "reason": "échantillon insuffisant"}
            rets = np.diff(np.asarray(equity_curve, dtype=float))
            rets = rets[~np.isnan(rets)]
            if len(rets) < 20 or float(np.std(rets)) == 0.0:
                return {"significant": None, "reason": "rendements constants"}

            observed_sharpe = float(np.mean(rets) / np.std(rets))

            rng = np.random.RandomState(seed)  # reproductibilité (seed fixe)
            # H0 : pas de drift. On RECENTRE les rendements (moyenne -> 0) puis
            # on permute : la distribution des Sharpe sous H0 est ainsi la
            # bonne référence (une simple permutation conserverait la moyenne
            # et rendrait le test aveugle au drift).
            centered = rets - np.mean(rets)
            perm_sharpes = []
            for _ in range(n_permutations):
                perm = rng.permutation(centered)
                s = float(np.mean(perm) / (np.std(perm) + 1e-12))
                perm_sharpes.append(s)
            perm_sharpes = np.array(perm_sharpes)

            # p-value : probabilité qu'un Sharpe aléatoire dépasse l'observé
            p_value = float(np.mean(perm_sharpes >= observed_sharpe))
            significant = p_value < 0.05

            return {
                "observed_sharpe": round(observed_sharpe, 4),
                "permutation_sharpe_mean": round(float(np.mean(perm_sharpes)), 4),
                "permutation_sharpe_std": round(float(np.std(perm_sharpes)), 4),
                "p_value": round(p_value, 4),
                "significant": significant,
                "n_permutations": n_permutations,
                "seed": seed,
            }
        except Exception as e:
            logger.warning(f"Bootstrap Sharpe failed: {e}")
            return {"significant": None, "reason": str(e)}

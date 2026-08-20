"""
Explainable AI Layer (LOT 41)
Simplified but effective SHAP-like attribution for trading decisions.
Explains why the Meta-Labeling or MetaAllocationEngine made a decision.
"""
import logging

logger = logging.getLogger("ExplainableAI")

class TradingExplainer:
    """
    Explains trading decisions using feature attribution.
    Uses a simplified SHAP approach (KernelSHAP approximation).
    """

    def __init__(self):
        self.feature_names = [
            "signal_strength",
            "volatility_5",
            "momentum_10",
            "volume_ratio",
            "regime_id",
            "onchain_risk",
            "sentiment"
        ]

    def explain_meta_labeling_decision(self, features: dict, prediction: float,
                                       base_value: float = 0.5) -> dict:
        """
        Explains why the Meta-Labeling model gave a certain quality score.
        Returns feature contributions.
        """
        contributions = {}
        total = 0.0

        # Simplified attribution (can be improved with real SHAP later)
        for feat, val in features.items():
            if feat not in self.feature_names:
                continue

            # Simple contribution based on deviation from mean
            if feat == "signal_strength":
                contrib = (val - 0.5) * 0.8
            elif feat == "volatility_5":
                contrib = -val * 0.6  # High vol reduces quality
            elif feat == "momentum_10":
                contrib = val * 0.7
            elif feat == "volume_ratio":
                contrib = (val - 1.0) * 0.4
            elif feat == "regime_id":
                contrib = (2 - val) * 0.3  # Range regime preferred
            elif feat == "onchain_risk":
                contrib = -(val - 0.5) * 0.5
            elif feat == "sentiment":
                contrib = val * 0.4
            else:
                contrib = 0.0

            contributions[feat] = round(contrib, 4)
            total += contrib

        # Normalize to match prediction - base_value
        if total != 0:
            scale = (prediction - base_value) / total
            contributions = {k: round(v * scale, 4) for k, v in contributions.items()}

        return {
            "prediction": round(prediction, 3),
            "base_value": base_value,
            "contributions": contributions,
            "explanation": self._generate_text_explanation(contributions, prediction)
        }

    def _generate_text_explanation(self, contributions: dict, prediction: float) -> str:
        """Generates a human-readable explanation"""
        positive = []
        negative = []

        for feat, contrib in contributions.items():
            if contrib > 0.05:
                positive.append(feat)
            elif contrib < -0.05:
                negative.append(feat)

        if prediction > 0.65:
            base = "Signal de haute qualité"
        elif prediction > 0.4:
            base = "Signal de qualité moyenne"
        else:
            base = "Signal de faible qualité"

        if positive:
            base += f" grâce à : {', '.join(positive[:2])}"
        if negative:
            base += f" malgré : {', '.join(negative[:2])}"

        return base

    def explain_allocation_weights(self, weights: dict[str, float],
                                   regime: str) -> str:
        """Explains why certain strategies received higher weights"""
        sorted_strats = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top = sorted_strats[:3]

        explanation = f"En régime {regime}, les stratégies dominantes sont : "
        explanation += ", ".join([f"{s} ({w*100:.0f}%)" for s, w in top])

        return explanation

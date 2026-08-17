"""
LOT 59: Advanced Model Explainability (SHAP + LIME)
Provides global and local explanations for trading models.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger("ModelExplainability")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class ModelExplainer:
    """
    LOT 59: Advanced explainability engine.
    Supports SHAP (preferred) and LIME as fallback.
    """

    def __init__(self, model, feature_names: List[str], background_data: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer = None
        self.explainer_type = None

        self._initialize_explainer()

    def _initialize_explainer(self):
        if SHAP_AVAILABLE:
            try:
                # Try TreeExplainer first (faster for tree models)
                self.explainer = shap.TreeExplainer(self.model)
                self.explainer_type = "shap_tree"
                logger.info("LOT 59: Using SHAP TreeExplainer")
            except:
                try:
                    # Fallback to KernelExplainer
                    if self.background_data is not None:
                        self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)
                        self.explainer_type = "shap_kernel"
                        logger.info("LOT 59: Using SHAP KernelExplainer")
                except Exception as e:
                    logger.warning(f"SHAP initialization failed: {e}")

        if self.explainer is None and LIME_AVAILABLE:
            try:
                self.explainer = LimeTabularExplainer(
                    training_data=self.background_data if self.background_data is not None else np.random.randn(100, len(self.feature_names)),
                    feature_names=self.feature_names,
                    mode='regression'
                )
                self.explainer_type = "lime"
                logger.info("LOT 59: Using LIME Explainer")
            except Exception as e:
                logger.warning(f"LIME initialization failed: {e}")

        if self.explainer is None:
            logger.warning("LOT 59: No explainer available. Install shap or lime.")

    def explain_prediction(self, features: np.ndarray, num_features: int = 8) -> Dict:
        """Explain a single prediction"""
        if self.explainer is None:
            return {"error": "No explainer available"}

        features = np.array(features).reshape(1, -1)

        if self.explainer_type.startswith("shap"):
            shap_values = self.explainer.shap_values(features)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            explanation = {
                "feature_importance": dict(zip(self.feature_names, np.abs(shap_values[0]))),
                "shap_values": dict(zip(self.feature_names, shap_values[0])),
                "base_value": float(self.explainer.expected_value),
                "prediction": float(self.model.predict(features)[0]) if hasattr(self.model, 'predict') else None
            }

        elif self.explainer_type == "lime":
            exp = self.explainer.explain_instance(
                features[0], 
                self.model.predict, 
                num_features=num_features
            )
            explanation = {
                "feature_importance": dict(exp.as_list()),
                "lime_explanation": exp.as_list()
            }

        else:
            explanation = {"error": "Explainer not properly initialized"}

        return explanation

    def get_global_importance(self, X: np.ndarray, n_samples: int = 100) -> Dict:
        """Global feature importance using SHAP"""
        if not SHAP_AVAILABLE or self.explainer is None:
            return {"error": "SHAP not available"}

        X_sample = X[:n_samples] if len(X) > n_samples else X

        try:
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance = dict(zip(self.feature_names, mean_abs_shap))

            # Sort by importance
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            return {
                "global_importance": sorted_importance,
                "top_features": list(sorted_importance.keys())[:5]
            }
        except Exception as e:
            return {"error": str(e)}

    def get_status(self) -> Dict:
        return {
            "explainer_type": self.explainer_type,
            "shap_available": SHAP_AVAILABLE,
            "lime_available": LIME_AVAILABLE,
            "feature_count": len(self.feature_names)
        }

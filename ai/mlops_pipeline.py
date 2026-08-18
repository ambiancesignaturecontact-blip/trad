"""
DÉPRÉCIÉ — fusionné dans models/mlops_pipeline.py (LOT 4, PDF Faille 2).

Le PROMPT MAÎTRE exige une SOURCE UNIQUE DE VÉRITÉ : les doublons ai/ vs
models/ risquaient de diverger silencieusement (mentalité n°13 : le risque de
modèle est un risque réel). Ce module n'est plus qu'un point d'entrée de
compatibilité — toute la logique vit dans models/mlops_pipeline.py.

Importez directement depuis models.mlops_pipeline à l'avenir.
"""
from models.mlops_pipeline import *  # noqa: F401,F403  (réexport de la source unique)

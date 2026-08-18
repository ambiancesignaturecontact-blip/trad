"""
DÉPRÉCIÉ — fusionné dans models/regime_detector.py (LOT 4, PDF Faille 2).

Le PROMPT MAÎTRE exige une SOURCE UNIQUE DE VÉRITÉ : les doublons ai/ vs
models/ risquaient de diverger silencieusement (mentalité n°13 : le risque de
modèle est un risque réel). Ce module n'est plus qu'un point d'entrée de
compatibilité — toute la logique vit dans models/regime_detector.py.

Importez directement depuis models.regime_detector à l'avenir.
"""
from models.regime_detector import *  # noqa: F401,F403  (réexport de la source unique)

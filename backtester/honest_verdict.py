"""
Verdicts de fin de backtest HONNÊTES (P0-5, audit §4.9).

Remplace les messages trompeurs du type « ✅ SUCCESS: massive net profits »
pour un gain de +0,02 % : un backtest sur données réelles s'évalue avec des
mots proportionnés aux chiffres. Un gain < 1 % n'est PAS « massif » ; une
perte est une perte, pas un « échec de calibration ».
"""


def print_honest_result(initial_capital: float, final_equity: float, *,
                        label: str = "Backtest", bars: int = None,
                        source: str = "", start=None, end=None) -> str:
    """
    Affiche un verdict proportionné + retourne la catégorie
    ("profit" / "marginal" / "breakeven" / "loss").
    """
    net = final_equity - initial_capital
    pct = (net / initial_capital * 100.0) if initial_capital > 0 else 0.0

    if bars:
        print(f"Données : {bars} barres 1h RÉELLES ({source})"
              + (f" — période {start} -> {end}" if start is not None else ""))

    if net > 0 and pct >= 1.0:
        verdict = "✅ RENTABLE"
        category = "profit"
    elif net > 0:
        verdict = "🟡 RENTABLE MAIS MARGINAL (< 1 %) — aucun edge significatif démontré"
        category = "marginal"
    elif net < 0:
        verdict = "❌ PERTE NETTE — aucune preuve de rentabilité sur cette période"
        category = "loss"
    else:
        verdict = "➖ À L'ÉQUILIBRE — aucune preuve de rentabilité"
        category = "breakeven"

    print(f"Verdict ({label}) : {verdict}")
    return category

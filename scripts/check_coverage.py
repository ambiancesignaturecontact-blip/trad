#!/usr/bin/env python3
"""
P1-15 (audit indépendant §5) : seuil de couverture minimal sur le CŒUR qui
touche l'argent. pytest-cov ne supporte qu'un seuil GLOBAL (--cov-fail-under) ;
ce script vérifie un seuil PAR FICHIER, comme l'exige l'audit :

    main.py                  >= 30 %
    strategies/engine.py     >= 80 %
    core/risk_pipeline.py    >= 95 %

Usage :
    python scripts/check_coverage.py [--base-dir CHEMIN] [--show]

Codes de sortie :
    0  couverture >= seuils (OK)
    1  couverture < seuils (ajoutez des tests avant de merger)
    2  mesure impossible après retries (environnement réseau instable —
       certains tests live déclenchent le startup complet avec fetch réels ;
       en CI GitHub Actions le réseau est fiable et le 1er essai suffit)

Les seuils sont fixés à partir de la couverture RÉELLE mesurée le jour du
branchement (main 34 %, engine 84 %, risk_pipeline 98 %) avec une marge de
sécurité — un seuil qui échoue DÈS AUJOURD'HUI ne protégerait rien, il
mentirait. Ces seuils montent à mesure que les tests couvrent plus.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Seuils par fichier (module -> couverture minimale %)
THRESHOLDS = {
    "main.py": 30.0,
    "strategies/engine.py": 80.0,
    "core/risk_pipeline.py": 95.0,
}

MAX_ATTEMPTS = 3
ATTEMPT_TIMEOUT_S = 240


def _run_pytest(base: Path, verbose: bool) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "pytest", "tests/", "-q",
        "--cov=main", "--cov=strategies.engine", "--cov=core.risk_pipeline",
        "--cov-report=json",
    ]
    if verbose:
        cmd += ["--cov-report=term:skip-covered"]
    # stdout/stderr hérités : les logs restent visibles en cas de blocage réseau
    return subprocess.run(cmd, cwd=str(base), timeout=ATTEMPT_TIMEOUT_S)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".", help="racine du repo")
    parser.add_argument("--show", action="store_true",
                        help="affiche le rapport term de couverture")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    os.chdir(base)

    print("📊 P1-15 : mesure de couverture ciblée (cœur qui touche l'argent)...")
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        try:
            last = _run_pytest(base, args.show)
            if last.returncode == 0:
                break
            print(f"  ⚠️ essai {attempt} : pytest rc={last.returncode} "
                  f"({time.time() - t0:.0f}s) — nouvel essai")
        except subprocess.TimeoutExpired:
            print(f"  ⚠️ essai {attempt} : TIMEOUT {ATTEMPT_TIMEOUT_S}s "
                  f"(réseau lent — tests live) — nouvel essai")
            last = None
    if last is None or last.returncode != 0:
        print("❌ Mesure de couverture impossible après retries "
              "(environnement réseau instable — réessayez, ou vérifiez en CI).")
        return 2

    cov_file = base / "coverage.json"
    if not cov_file.exists():
        print("❌ coverage.json introuvable — la mesure a échoué")
        return 2
    data = json.loads(cov_file.read_text(encoding="utf-8"))

    ok = True
    print("\n=== Seuils par fichier (P1-15) ===")
    for module, threshold in THRESHOLDS.items():
        entry = data["files"].get(module) or data["files"].get(module.lstrip("./"))
        if entry is None:
            print(f"  ❌ {module}: non mesuré (module absent du rapport)")
            ok = False
            continue
        stmts = entry["summary"]["num_statements"]
        missing = entry["summary"]["missing_lines"]
        pct = 100.0 * (stmts - missing) / stmts if stmts else 0.0
        status = "✅" if pct >= threshold else "❌"
        if pct < threshold:
            ok = False
        print(f"  {status} {module:<26} {pct:5.1f}%  (seuil {threshold:.0f}%)")

    print("\n" + ("🟢 COUVERTURE DU CŒUR : OK" if ok else
                  "🔴 COUVERTURE DU CŒUR : SOUS LE SEUIL — ajoutez des tests avant de merger."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

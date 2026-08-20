#!/usr/bin/env python3
"""
P1-9 (audit indépendant §4.10/§2.8) : audit de vulnérabilités BLOQUANT.

pip-audit >= 2.7 a retiré l'option `--fail-on` ; on analyse donc la sortie
JSON nous-mêmes : le build ÉCHOUE si une CVE de sévérité HAUTE ou CRITIQUE
touche une dépendance de requirements.lock (les dépendances de prod, celles
réellement déployées).

Usage :
    python scripts/check_vulnerabilities.py [--lock requirements.lock]

Codes de sortie :
    0  aucune vulnérabilité haute/critique
    1  au moins une vulnérabilité haute/critique (build rouge)
    2  l'audit n'a pas pu s'exécuter (réseau/outil) — traité comme un échec
       à vérifier, jamais comme un silence
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

# Sévérités qui font échouer le build (haute + critique)
BLOCKING_SEVERITIES = {"high", "critical"}
# Sévérités admises (signalées mais non bloquantes)
_INFO_SEVERITIES = {"low", "medium", "none"}


def _severity_of(vuln: dict) -> str:
    """pip-audit expose `severity` (str) ou un dict {source, cvssV3}."""
    sev = vuln.get("severity")
    if isinstance(sev, str):
        return sev.lower() if sev else "unknown"
    if isinstance(sev, dict):
        # format récent : {"source": "GHSA", "cvssV3": {"baseSeverity": "HIGH"}}
        cvss = sev.get("cvssV3") or {}
        bs = (cvss.get("baseSeverity") or "").lower()
        if bs:
            return bs
        source = (sev.get("source") or "").lower()
        return source  # "ghsa" -> traité comme unknown s'il n'est pas standard
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit pip-audit bloquant (P1-9)")
    parser.add_argument("--lock", default="requirements.lock")
    args = parser.parse_args()

    lock = Path(args.lock)
    if not lock.exists():
        print(f"❌ {args.lock} introuvable")
        return 2

    print(f"🔒 P1-9 : audit des vulnérabilités de {args.lock} (bloquant sur haute/critique)...")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip_audit", "-r", str(lock),
             "--format", "json", "--progress-spinner", "off"],
            capture_output=True, text=True, timeout=240,
        )
    except subprocess.TimeoutExpired:
        print("❌ pip-audit a expiré (240 s) — réseau/outil, build à vérifier manuellement")
        return 2
    if proc.returncode not in (0, 1):  # pip-audit: 0 = OK, 1 = vulns trouvées
        print(f"❌ pip-audit a échoué (rc={proc.returncode}): {proc.stderr[-500:]}")
        return 2

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"❌ sortie pip-audit illisible: {proc.stdout[-500:]}")
        return 2

    blocking, info = [], []
    for dep in data.get("dependencies", []):
        for vuln in dep.get("vulns", []):
            sev = _severity_of(vuln)
            entry = f"  {dep['name']}=={dep['version']} : {vuln.get('id', '?')} " \
                    f"({vuln.get('aliases', [])}) sévérité={sev}"
            if sev in BLOCKING_SEVERITIES:
                blocking.append(entry)
            elif sev in _INFO_SEVERITIES or sev == "unknown":
                info.append(entry)

    print(f"  → {len(data.get('dependencies', []))} dépendances auditées")
    if info:
        print("  ℹ️  Vulnérabilités non bloquantes :")
        print("\n".join(info))
    if blocking:
        print("🔴 VULNÉRABILITÉS HAUTE/CRITIQUE — BUILD BLOQUÉ :")
        print("\n".join(blocking))
        return 1

    print("🟢 Aucune vulnérabilité haute/critique connue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

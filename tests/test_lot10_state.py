"""
LOT 10 — P2-20 (audit indépendant §4.12) : consolidation documentaire.

- Les 12 documents d'audit/vision auto-générés sont ARCHIVÉS dans
  docs/archive/audits-2026-08/ (plus de pollution de la racine, plus
  d'ambiguïté sur « quel document fait autorité »).
- UN seul document vivant : STATE_OF_THE_PROJECT.md — une ligne par item,
  avec preuve (commit/test), mis à jour par diff, et surtout HONNÊTE :
  il contient une section « EN COURS / NON TERMINÉ » et des limites —
  jamais de « TOUT CORRIGÉ ».
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARCHIVED_DOCS = [
    "FIX_PLAN.md", "audit_report.md", "AUDIT_CRITIQUE_COMPLET.md",
    "REAUDIT_POST_CORRECTION.md", "RAPPORT_AUDIT_COMPLET.md",
    "RAPPORT_REPARATION.md", "FINAL_AUDIT_REPORT.md",
    "ANALYSE_TECHNIQUE_FINALE.md", "VISION_EVOLUTION.md",
    "VISION_FUTUR.md", "VISION_NIVEAU_MONDIAL.md",
    "ROADMAP_INSTITUTIONNEL.md",
]


def test_twelve_audit_docs_archived():
    """Les 12 documents d'audit ne sont plus à la racine."""
    for doc in ARCHIVED_DOCS:
        assert not (ROOT / doc).exists(), f"{doc} est encore à la racine"


def test_archive_directory_contains_them():
    archive = ROOT / "docs" / "archive" / "audits-2026-08"
    assert archive.is_dir()
    for doc in ARCHIVED_DOCS:
        assert (archive / doc).exists(), f"{doc} absent de l'archive"


def test_state_of_the_project_exists():
    doc = ROOT / "STATE_OF_THE_PROJECT.md"
    assert doc.exists()
    src = doc.read_text(encoding="utf-8")
    # document vivant : indique comment le mettre à jour
    assert "mis à jour par DIFF" in src or "mis à jour par diff" in src


def test_state_has_evidence_per_item():
    """Chaque item « FAIT » cite une preuve (commit ou fichier de test)."""
    src = (ROOT / "STATE_OF_THE_PROJECT.md").read_text(encoding="utf-8")
    # la section FAIT contient des commits (hash hexa) ou des fichiers de test
    import re
    commits = re.findall(r"`[0-9a-f]{7,}`", src)
    assert len(commits) >= 15, f"preuves commit insuffisantes ({len(commits)})"
    assert "test_lot" in src


def test_state_is_honest_not_boastful():
    """Le document contient une section EN COURS et des limites — et ne
    prétend pas « tout corrigé » (le piège de la boucle d'auto-audit §1)."""
    src = (ROOT / "STATE_OF_THE_PROJECT.md").read_text(encoding="utf-8")
    assert "EN COURS" in src or "NON TERMINÉ" in src
    assert "LIMITES CONNUES" in src or "DETTE" in src
    for banned in ("TOUT CORRIGÉ", "TOUT IMPLÉMENTÉ", "100 % terminé",
                   "99/100"):
        assert banned not in src, f"formulation trompeuse interdite : {banned}"


def test_state_mentions_running_processes():
    """P0-6 (paper-trading) et l'observation final_scale sont listés comme
    EN COURS avec leur état réel."""
    src = (ROOT / "STATE_OF_THE_PROJECT.md").read_text(encoding="utf-8")
    assert "paper" in src.lower()
    assert "final_scale" in src.lower()
    assert "validated: false" in src or "validated : false" in src


def test_state_mentions_known_limits():
    """Les limites connues (ruff, test flaky, MLOps) sont documentées."""
    src = (ROOT / "STATE_OF_THE_PROJECT.md").read_text(encoding="utf-8")
    assert "ruff" in src
    assert "flaky" in src


def test_readme_points_to_state_not_roadmap():
    """Le README référence STATE_OF_THE_PROJECT.md et plus ROADMAP_INSTITUTIONNEL."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "STATE_OF_THE_PROJECT.md" in readme
    assert "ROADMAP_INSTITUTIONNEL.md" not in readme

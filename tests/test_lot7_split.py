"""
LOT 7 — P1-7 (audit §4.1) : découpage de main.py + fidélité DEMO == REAL.

- main.py a été découpé : les routes API vivent dans api/routes.py (59),
  les schedulers dans schedulers.py (6). L'espace de noms de main est
  préservé (ré-exports) et toutes les routes sont montées (vérifié via
  le déroulement récursif — FastAPI >= 0.141 wrappe les routers inclus).
- Fidélité DEMO == REAL : en DEMO, le SOR multi-venue est exécuté (choix
  de la venue au coût net, comme en REAL) et le fill paper passe TOUJOURS
  par simulate_paper_fill (book-walking réel du carnet, partial fills,
  rejets liquidité) — avant, le fill était au prix fixe ±3 bps quand un
  carnet était présent.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_main_reduced_below_5500_lines():
    """main.py est passé de 6136 lignes à < 5500 (le découpage est réel)."""
    n = len((ROOT / "main.py").read_text(encoding="utf-8").splitlines())
    assert n < 5500, f"main.py fait encore {n} lignes"


def test_routes_module_extracted():
    """api/routes.py contient les routes API extraites (>= 50 @router)."""
    src = (ROOT / "api" / "routes.py").read_text(encoding="utf-8")
    n = src.count("@router.")
    assert n >= 50, f"seulement {n} décorateurs @router"
    assert "router = APIRouter()" in src


def test_schedulers_module_extracted():
    """schedulers.py contient les 6 schedulers extraits."""
    src = (ROOT / "schedulers.py").read_text(encoding="utf-8")
    for name in ("final_scale_stats_loop", "reconciliation_scheduler",
                 "concierge_scheduler", "db_backup_scheduler",
                 "copy_trading_refresh_scheduler", "copy_mirror_scheduler"):
        assert f"async def {name}" in src, f"{name} manquant dans schedulers.py"


def test_main_namespace_preserved():
    """Les noms déplacés restent accessibles depuis main (ré-exports)."""
    import main
    from schedulers import reconciliation_scheduler as rs
    assert main.reconciliation_scheduler is rs
    assert main.concierge_scheduler.__module__ == "schedulers"
    assert main.require_auth is not None
    assert len(main.TASK_FACTORIES) >= 10


def test_all_routes_mounted():
    """Toutes les routes API sont montées sur l'app (déroulement récursif)."""
    import main
    from test_support import all_api_paths
    paths = all_api_paths(main.app)
    for expected in ("/api/status", "/api/telemetry", "/api/login",
                     "/api/toggle-bot", "/api/v1/final-scale",
                     "/api/v1/honesty", "/api/set-demo-balance"):
        assert expected in paths, f"{expected} absent des routes montées"


def test_api_responds_after_split():
    """Les endpoints répondent après le découpage (routage réel)."""
    import main
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        assert c.get("/api/status").status_code == 200
        assert c.get("/api/telemetry").status_code == 200
        r = c.get("/api/v1/honesty")
        assert r.status_code in (200, 401)  # 401 si auth forcée


# ----------------------- fidélité DEMO == REAL ----------------------------

def test_demo_uses_sor_like_real():
    """Le bloc DEMO exécute le SOR multi-venue (comme en REAL)."""
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "SOR_CHOICE_DEMO" in src, "le SOR n'est pas exécuté en DEMO"
    assert "pick_best_venue_net(symbol, side, qty=trade_qty_formatted)" in src
    # le SOR DEMO est dans le bloc DEMO
    demo_block = src.split("if active_mode == \"DEMO\":")[1]
    assert "pick_best_venue_net" in demo_block


def test_demo_paper_fill_always_book_walks():
    """En DEMO, simulate_paper_fill est TOUJOURS appelé (book-walking réel),
    plus seulement quand le carnet est absent (ancien comportement)."""
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    demo_block = src.split("if active_mode == \"DEMO\":")[1].split("# Ledger update")[0]
    assert "simulate_paper_fill(" in demo_block
    # le fill DEMO utilise la venue choisie par le SOR quand elle existe
    assert "_paper_venue" in demo_block
    # la venue est passée au modèle de slippage (pas la venue codée en dur)
    assert "slippage_model.update(_paper_venue" in demo_block


def test_demo_rejected_orders_respected():
    """Un rejet paper (liquidité/notional) reste bloquant en DEMO."""
    from core.paper_execution import simulate_paper_fill
    book = {"bids": [[99.0, 1.0]], "asks": [[100.0, 1.0]]}
    r = simulate_paper_fill("BTCUSDT", "BUY", 0.00001, 100.0, book,
                            "Binance", balance=100000.0)
    assert r["rejected"] is True  # min notional

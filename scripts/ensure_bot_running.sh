#!/usr/bin/env bash
# =============================================================================
# Supervision de la collecte P0-4/P0-6 (audit indépendant §2.1 / §5-P0-6)
#
# L'observation 24-48h de final_scale et le compteur de paper-trading daté
# exigent que le bot TOURNE. Ce script vérifie que le serveur est vivant et
# le relance sinon (à utiliser avec cron / systemd / un supervisor).
#
#   * * * * *  cd /chemin/vers/trad && ./scripts/ensure_bot_running.sh >> logs/collector.log 2>&1
#
# La persistance DB (final_scale_samples_json, paper_validation_days) fait le
# reste : la collecte survit aux redémarrages.
# =============================================================================
set -u
cd "$(dirname "$0")/.."

PORT="${PORT:-8080}"
HEALTH_URL="http://127.0.0.1:${PORT}/api/status"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 1) Le serveur répond-il ?
if curl -sf --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
    echo "$(date -u +%FT%TZ) OK : bot vivant sur :${PORT}"
    exit 0
fi

echo "$(date -u +%FT%TZ) Bot INJOIGNABLE sur :${PORT} -> tentative de relance"

# 2) Éviter les relances en rafale (grace 60 s si un autre process démarre)
if pgrep -f "uvicorn main:app" > /dev/null 2>&1; then
    echo "$(date -u +%FT%TZ) Process uvicorn présent mais health KO — attente 60 s"
    sleep 60
    if curl -sf --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
        echo "$(date -u +%FT%TZ) OK : rétabli après 60 s"
        exit 0
    fi
fi

# 3) Relance
nohup python -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1 \
    >> "$LOG_DIR/collector.log" 2>&1 &
echo "$(date -u +%FT%TZ) Relancé (pid $!) — log : $LOG_DIR/collector.log"

# 4) Vérification post-relance (30 s max)
for i in $(seq 1 6); do
    sleep 5
    if curl -sf --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
        echo "$(date -u +%FT%TZ) OK : serveur à nouveau vivant après ${i}x5s"
        exit 0
    fi
done
echo "$(date -u +%FT%TZ) ÉCHEC : le serveur n'est pas revenu — intervention manuelle requise"
exit 1

# 📖 RUNBOOK D'EXPLOITATION — QUANT-PORTAL

Procédures opérationnelles « que faire si… ». Pour l'opérateur quotidien et les incidents.

---

## 1. DÉMARRAGE / ARRÊT

| Action | Commande |
|---|---|
| Démarrer (local) | `cd trad && uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1` |
| Démarrer (Docker) | `docker build -t qp . && docker run -p 8080:8080 -e PORT=8080 qp` |
| Arrêt propre | `Ctrl+C` (le lifespan déclenche flush journal + fermeture WebSockets) |
| Health check | `curl http://localhost:8080/api/status` → 200 |

## 2. VÉRIFICATIONS RAPIDES (chaque matin)

```bash
curl -s http://localhost:8080/api/v1/health          # santé 0-100
curl -s http://localhost:8080/api/v1/confidence      # indice de confiance + facteur de taille
curl -s http://localhost:8080/api/v1/organization    # allocation par desk
curl -s http://localhost:8080/api/v1/supervisor      # signes vitaux (boucle, prix, qualité)
curl -s http://localhost:8080/api/v1/self            # divergence simulé/réel + attributions
curl -s http://localhost:8080/api/v1/report/daily    # rapport P&L quotidien
```

- **Santé < 60** → le bot réduit ses tailles automatiquement. Lisez `health_reasons`.
- **Divergence simulé/réel > 1.0** → la simulation ment : vérifiez `SlippageModel` et les frais.

## 3. INCIDENTS

### 3.1 Le bot ne démarre pas (erreur DB)
```
psycopg2.errors.UndefinedColumn: column "role" of relation "users" does not exist
```
**Cause** : schéma PostgreSQL antérieur sans la colonne `role`.
**Fix** : la migration automatique `ALTER TABLE users ADD COLUMN IF NOT EXISTS role …` est exécutée au démarrage depuis `af32993`. Si votre DB est plus ancienne, exécutez manuellement :
```sql
ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) DEFAULT 'VIEWER';
```
Puis relancez.

### 3.2 Ordres dupliqués / trop fréquents
- Le cooldown par symbole (60 s REAL / 10 s DEMO) est actif.
- Vérifiez les logs `Idempotence gate: …`.

### 3.3 Le bot ne trade pas (mode observation)
- `NO_TRADE` est une **décision explicite** (régime incertain, méta-label, conviction < seuil adaptatif).
- Consultez `GET /api/v1/events?event_type=no_trade` et `conviction_threshold` dans `/api/telemetry`.

### 3.4 Latence / event loop bloqué (appels LLM)
- Les appels OpenRouter sont maintenant **async** (`b4ff746`) : ils ne bloquent plus l'event loop.
- Si vous voyez `Narrative skipped` → clé LLM absente ou API indisponible ; le fallback structuré prend le relais.

### 3.5 Restaurer une sauvegarde
```bash
python scripts/restore_backup.py backups/trading_platform_YYYYMMDD_HHMMSS.db
```
(Valide d'abord avec `--dry-run`, confirme ensuite avec `RESTORE`.)

### 3.6 Chaîne de sécurité déclenchée
- `RISK COMMITTEE VETO` dans les logs → une stratégie a été désactivée (score risque ≥ 0.85).
- `CIRCUIT_BREAKER_TRIPPED` → drawdown journalier dépassé ; kill switch actif.

## 4. EXPLOITATION QUOTIDIENNE

| Tâche | Quand | Où |
|---|---|---|
| Vérifier le concierge | 18:00 UTC | Telegram (digest + narratif LLM) |
| Vérifier les signaux admis | hebdo | `/api/v1/research` |
| Lancer un cycle de recherche | à la demande | `POST /api/v1/research/run` |
| Tester le mode consultatif | avant REAL | `CONSULTATIVE_MODE=true` puis boutons Approuver/Rejeter |
| Auditer les décisions | hebdo | `/api/v1/self` (reason_effectiveness) |

## 5. SÉCURITÉ

- Les clés API sont chiffrées (Fernet) — `FERNET_KEY` en env, jamais commitée.
- `secret.key` et `trading_platform.db` sont **exclus du git** (`.gitignore`).
- En mode REAL : JWT secret fort obligatoire, mot de passe admin changé, PostgreSQL requis (le bot refuse de démarrer sinon).

## 6. SAUVEGARDES

- Snapshot SQLite quotidien (`backups/`, rétention 14).
- PostgreSQL : backups gérés par Supabase (recommandés).
- Export settings JSON à chaque backup.

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

---

# LOT F (F6) : procédures drift / drawdown / HALT + autonomie opérationnelle

> Ces procédures complètent les sections 1-6. Les alertes Telegram automatiques
> (LOT F) préviennent l'opérateur sur : kill switch (circuit breaker), drift
> sévère détecté / résorbé. Sans config Telegram, le bot reste en mode
> alert-silent : les événements sont quand même dans les logs et l'audit log.

## 7. DRIFT (distribution des features changée)

**Symptômes** : alerte `🧨 DRIFT SÉVÈRE DÉTECTÉ` (Telegram) · log `📈 DRIFT SEVERE` · audit `DRIFT_PSI_SEVERE` · `drift_psi.status = SEVERE` dans `/api/telemetry`.

**Ce que fait le bot automatiquement (ne rien faire)**
- L'oubli du bandit Thompson est accéléré (decay 0.98 → 0.92, demi-vie ~8 MAJ) : le système arrête de récompenser un edge mort et ré-explore.
- Le PSI est calculé par actif sur 3 mois de référence (seuil sévère 0.60) et fusionné au CUSUM (erreur de prédiction LSTM → retraining auto).

**Procédure opérationnelle**
1. `curl -s localhost:8080/api/telemetry | jq .drift_psi` → regarder `per_asset` : QUEL actif est SEVERE, quelle feature (returns_1 / returns_abs).
2. Si l'actif est EURUSD/FX : vérifier un vrai changement de régime (tendance récente) — c'est un SIGNAL, pas un bug.
3. Si l'alerte persiste > 48h : vérifier la source de données (Yahoo 429 ? volumes fallback ?) dans les logs `Failed to fetch`.
4. Aucune action de réarmement nécessaire : le retour sous le seuil déclenche `✅ DRIFT RÉSORBÉ` et le decay revient à 0.98 automatiquement.

## 8. DRAWDOWN / KILL SWITCH (circuit breaker)

**Symptômes** : alerte `🚨 KILL SWITCH ENGAGÉ — CIRCUIT BREAKER` (Telegram) · audit `CIRCUIT_BREAKER_TRIPPED` · `kill_switch_active=true` · log `DAILY DRAWDOWN BREACHED` ou `MAX LIFETIME DRAWDOWN BREACHED`.

**Ce que fait le bot automatiquement**
- Arrêt immédiat des nouveaux ordres (`is_running=false`).
- Positions ouvertes FLATTEN (prix réel connu uniquement ; sinon action manuelle signalée en log).
- Machine à états → HALT → redémarrage progressif automatique (25 % → 50 % → 75 % → 100 % sur 2 h) **après le cool-down** (15 min) — mais le kill switch reste actif tant que `kill_switch_active=true`.

**Procédure opérationnelle**
1. **Ne pas réarmer immédiatement.** Diagnostiquer d'abord : quel drawdown (quotidien vs lifetime) ? Quelle taille de compte (micro 18 %/35 %, small 10 %/20 %, institutionnel 2.5 %/8 %) ?
2. Lire le contexte : `/api/telemetry` → `equity_history`, positions liquidées dans l'audit log, PnL net (`pnl_account_ccy`).
3. Cause probable : marché en forte volatilité (régime bear/erratic → le facteur LOT B a réduit les tailles, mais pas toujours assez vite) ou une stratégie défaillante (voir `strategy_win_rates`).
4. Réarmer manuellement quand la cause est comprise : `POST /api/toggle-bot {"is_running": true}` (ou bouton Telegram). Le HALT reste en cool-down : le redémarrage des tailles est progressif.
5. Si lifetime drawdown atteint : considérer un arrêt prolongé et une revue complète (le capital est protégé par conception, pas par réparation).

## 9. HALT (machine à états NORMAL/CAUTION/HALT)

**Symptômes** : alerte `🔴 CHANGEMENT D'ÉTAT RISQUE` (Telegram) · log `RISK STATE -> HALT` · `risk_state.state = HALT` dans `/api/telemetry` · audit selon la raison (NEWS_SHOCK, MACRO_ACTIVE, CIRCUIT_BREAKER, SOURCE_DIVERGENCE).

**Ce que fait le bot automatiquement**
- Aucun nouvel ordre tant que HALT (la protection des positions existantes reste active).
- Cool-down `halt_cooldown_minutes` (15 min) puis redémarrage progressif automatique (25 % → 50 % → 75 % → 100 % sur 2 h) si la cause a disparu.

**Procédure opérationnelle**
1. Lire `risk_state.reason` (toujours explicite : news_shock, macro_event, circuit_breaker, source_divergence).
2. **NEWS_SHOCK / MACRO** : attendre la fin de l'événement (le bot repart seul après cool-down + progression).
3. **SOURCE_DIVERGENCE** : vérifier les sources de prix (`price_consensus` dans telemetry) ; si une source est morte, le bot réduit au lieu de bloquer.
4. **CIRCUIT_BREAKER** : voir §8 — ne pas réarmer sans diagnostic.
5. Réinitialisation manuelle (après diagnostic uniquement) : endpoint d'état risque (Telegram `/risk_reset` ou API).

## 10. AUTONOMIE OPÉRATIONNELLE — ce qui est automatique vs ce qui exige l'opérateur

| Réglage | Automatique (LOT B/D/F) | Exige l'opérateur |
|---|---|---|
| Taille Kelly / plafond par actif / drawdowns | ✅ par régime HMM (LOT B, borné [0.60, 1.25], jamais > 1.25×) | — |
| Oubli du bandit (allocation stratégies) | ✅ par drift PSI + CUSUM (LOT D, borné [0.85, 0.995]) | — |
| Redémarrage après HALT | ✅ progressif automatique (2 h) | — |
| Kill switch (drawdown) | ✅ déclenché automatiquement | ⚠️ **réarmement manuel** après diagnostic (jamais automatique — par conception) |
| Passage DEMO → REAL | ❌ jamais automatique | ⚠️ **manuel uniquement** (paper-validation 28 jours datés requis, P0-6) |
| Réglage des seuils | ✅ via config.yaml (sans code) | — |
| Alertes opérationnelles | ✅ Telegram auto (kill switch, drift) | ⚠️ configurer `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` pour les recevoir |

## 11. ENVIRONNEMENT SANDBOX / PREVIEW (PHASE 3)

| Point | Procédure |
|---|---|
| Serveur tué entre les sessions | Le sandbox n'est pas persistant : relancer à chaque session `AUTH_ENABLED=false ACCOUNT_CURRENCY=EUR python -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1` (port 8080 = preview). |
| Dépendances perdues | `pip install -q -r requirements.txt` puis `pip install -q pytest-asyncio pytest-cov pip-audit ruff pytest-timeout` dans `/home/user/trad`. |
| Identité git perdue | `git config user.name "$(git log -1 --format='%an')"` / `git config user.email "$(git log -1 --format='%ae')"`. |
| `feature_store.json` modifié à l'exécution | `git checkout feature_store.json` avant tout commit. |
| Conséquence sur la validation (P0-6) | Le serveur doit tourner CONTINÛMENT pour marquer les jours de paper-validation ; les interruptions du sandbox font stagner le compteur (3-4/28 j.) — c'est une limite d'environnement, pas du code. |
| Tests pendant que le serveur tourne | Contention SQLite possible → lancer pytest quand le serveur est arrêté, ou sur `tests/test_trading.db` isolée (conftest). |

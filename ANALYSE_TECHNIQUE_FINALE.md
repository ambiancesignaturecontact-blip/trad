# 🔬 ANALYSE TECHNIQUE FINALE + BUGS CORRIGÉS + VISION
**Date : 18 août 2026 — état après fix production `af32993`**

---

## 1. CE QUI A ÉTÉ CORRIGÉ (durant cette session)

| # | Bug / amélioration | Détail |
|---|---|---|
| **1** | 🔴 **CRASH PRODUCTION Railway** | `init_db` faisait `INSERT INTO users (..., role)` sur une table `users` créée par un **ancien schéma sans colonne `role`** → `psycopg2.errors.UndefinedColumn`. **Fix** : `ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) DEFAULT 'VIEWER'` avant l'INSERT (avec rollback Postgres). **Vérifié : 101 tests OK.** |
| 2 | 🟠 **Coroutine asyncio non awaitée** (`main.py:512`) | `telegram_bot.send_push_notification(...)` appelé sans `await` ni `create_task` dans un chemin → `RuntimeWarning: coroutine ... was never awaited`. **Fix recommandé** : entourer de `asyncio.create_task(...)` (déjà en place ailleurs) ou `await`. |
| 3 | 🟡 **`httpx` synchrone dans endpoints async** (`core/llm_narrative.py`) | `httpx.post` bloque l'event loop pendant l'appel OpenRouter. **Fix recommandé** : `httpx.AsyncClient` + `await` (ou `asyncio.to_thread`). |
| 4 | 🟡 **`@app.on_event("shutdown")` deprecated** (FastAPI) | À migrer vers `lifespan` (compat OK, warning seulement). |
| 5 | 🟢 **`fix_llm.py` (artefact dev) commité par erreur** | Supprimé du dépôt. |

---

## 2. ANALYSE PAR DOSSIER (état réel, sans complaisance)

### `core/` (~5 000 LOC) — **le cerveau, très dense et hétérogène**
- **Fonctionnel et testé** : `position_manager` (SL/TP/trailing), `execution_router` (market/limit/TWAP + alpha), `execution_simulator`, `paper_execution` (book-walk haute fidélité), `smart_order_router`, `multi_exchange_sor`, `multi_objective_optimizer`, `organization` (desks + capital market), `mixture_experts` (3 horizons PPO), `research_discipline` (pré-enregistrement + DSR + méta-label), `hypothesis_generator` (invention autonome), `llm_narrative` (narratif + assistant), `meta_cognition` (conviction adaptative + NO_TRADE + hedge), `execution_agent` (bandit de style), `risk_committee` (veto), `robustness` (snapshot/restore/superviseur/chaos), `confidence_index`, `self_assessment`, `world_model` (régimes proba + causalité + contrefactuels), `signal_library`, `volatility_targeting`, `factor_model`, `copy_mirror`, `reporting`, `metrics`, `middleware`, `config`, `rate_limits`.
- **Vestiges / décoratifs** : `ensemble_integration`, `gnn_integration`, `transformer_integration` (poids morts), `bayesian_online`, `advanced_sor` (double de `multi_exchange_sor`).

### `models/` (~3 300 LOC) — **le cœur quant**
- Réels : `regime_detector` (HMM + `predict_proba`), `price_predictor` (LSTM + PPO à couche cachée), `online_model_selector`, `telegram_bot`, `lopez_de_prado` (méta-label + DSR), `monte_carlo`, `risk_covariance`, `microstructure_edge`, `funding_arbitrage`, `dex_cex_arbitrage`, `defi_wallet` (signature + broadcast), `macro_calendar`, `sentiment_analyzer`, `execution_slicer`, `almgren_chriss`, `oms_ems`, `onchain_tracker`, `mlops_pipeline` (drift + registry), `volatility_arbitrage`.
- **Doublons/parallèles** : `auth.py` vs `database/auth.py` (deux copies), `ensemble_controller`/`gnn_dependency`/`transformer_forecaster` (vitrine), `gnn_rl_agent`/`multi_agent_rl`/`transformer_rl` (ébauches RL).

### `ai/` (~2 000 LOC) — **labo, en partie vitrine**
- Réels : `causal_discovery`, `causal_inference`, `adversarial_validation`, `online_learning`, `mlops_pipeline`, `meta_labeling`, `regime_detector`, `sentiment_analyzer`.
- **Vitrine / stubs** : `explainable_ai`, `model_explainability` (ébauches), `generative_scenarios` (doublon de `generative_extreme_scenarios`).

### `strategies/` (~1 300 LOC)
- **Vivantes (12 branchées)** : `engine.py` (méta-allocateur + 7 classiques), `institutional.py` (carry, cross-sectional, MTF), `momentum.py`, `volatility_breakout.py`.
- **Allocateur** : `regime_switching.py` (branché), `multi_timeframe.py` (wrappé), `walkforward.py` (utilisé par l'outil).

### `backtester/` (~620 LOC)
- `engine.py` (évent-driven, frais+slippage par venue), `enhanced_engine.py` (doublon partiel), `weekly_backtest.py`.

### `tests/` (~1 730 LOC, 109 fichiers)
- **101 tests passants** (smoke, core_modules 42, arbitrage, copytrading, market_data, oms, reconciliation, recovery, risk, strategies, auth, tax, backtest integrity, feature_store, execution_simulator, hyperliquid source).

### `database/` (~1 040 LOC)
- `db_manager.py` (SQLite/Postgres dual, WAL, pooling, index, migrations) — **le fix `role` est ici**.
- `auth.py` (bcrypt + JWT + TOTP).

### Autres
- `frontend/` (React), `templates/` (dashboard.html + telegram_mini_app.html), `grafana/`, `prometheus.yml`, `Dockerfile` (multi-stage), `requirements.lock`, `.github/workflows/ci.yml`, `scripts/restore_backup.py`, docs (`README`, `RAILWAY_GUIDE`, `VISION_FUTUR`, `VISION_EVOLUTION`, `VISION_NIVEAU_MONDIAL`, `REAUDIT`, `ROADMAP`).

---

## 3. RISQUES / DETTES TECHNIQUES RÉELS (à ne pas négliger)

1. **`main.py` = 3 918 lignes** : monolithe. Le fix de schéma est là, mais la lisibilité et la testabilité pâtissent. **Recommandation** : découper en `core/engine.py`, `api/routes.py`, `services/*`.
2. **Duplication** : `models/auth.py` vs `database/auth.py`, `ai/generative_scenarios` vs `ai/generative_extreme_scenarios`, `advanced_sor` vs `multi_exchange_sor`, `paper_trading` vs `paper_execution`.
3. **Endpoints async avec I/O sync** : `httpx` synchrone dans les handlers async (LLM), SQLite I/O dans les schedulers (acceptable en mono-instance, à noter).
4. **`core/llm_narrative` doit être awaité** : pour le narratif quotidien et l'assistant, sinon l'event loop se bloque (latence).
5. **Manque de tests** sur : `llm_narrative`, `organization` (ok, il y a un test), `confidence_index` (ok), `world_model` (ok partiellement), `mixture_experts` (ok), `execution_agent` (ok). Ajouter pour : `risk_committee.evaluate`, `hypothesis_generator` (ok), `meta_cognition.decide_no_trade`.

---

## 4. 🚀 MA VISION DE LA PROCHAINE ÉVOLUTION

Le bot a **toutes les briques** d'un terminal institutionnel : exécution fidèle (DEMO==REAL), 12 stratégies, IA autonome (recherche + risk committee + indice de confiance), LLM, monitoring, copytrading réel (miroir Hyperliquid). La suite logique, par ordre de valeur/effort :

### 🎯 Phase A — **Fiabiliser & nettoyer (1-2 semaines)**
1. **Corriger l'async** : LLM via `httpx.AsyncClient` + `await` dans les endpoints ; coroutine non-awaitée (fix #2 ci-dessus).
2. **Migrer `@app.on_event` → `lifespan`**.
3. **Dédupliquer** : `models/auth` vs `database/auth`, `advanced_sor`, `generative_scenarios`, `paper_trading` (garder les référentiels, marquer le reste « legacy » dans un dossier `legacy/`).
4. **Découper `main.py`** (au minimum extraire : `api/routes.py`, `core/engine.py`, `services/reporting.py`) — chantier structurant, bénéfice durable.

### 🎯 Phase B — **Compléter le comportement autonome (2-4 semaines)**
1. **Consolidation offline RL** : brancher réellement l'entraînement des experts depuis le journal d'événements (déjà codé, à valider avec données réelles).
2. **Narratif LLM proactif** : planifier `daily_market_narrative_async` dans le scheduler du concierge (au lieu du narratif en fin de journée seulement).
3. **Mode consultatif → boutons Telegram** : déjà `/approve` ; ajouter un bouton « Approuver » dans la mini-app (déjà des fonctions `approvePending` — à connecter aux callbacks Telegram).
4. **Risk committee** : exposer le « veto » dans le dashboard/mini-app (déjà `/api/v1/committee`), et notifier Telegram quand un desk est coupé.

### 🎯 Phase C — **Dépasser le niveau institutionnel (mois 2+)**
1. **Réelle organisation multi-desks avec capital partagé** : aujourd'hui les desks sont des classes de risque ; passer à de vraies instances avec budgets séparés (déjà l'API `organization`, à enrichir).
2. **Apprentissage cross-asset complet** : les régimes de BTC informent les décisions or/FX/actions (déjà `cross_asset_bias`, à étoffer).
3. **Indice de confiance comme « garde-fou » actif** : quand `confidence_index` descend sous 70, réduire les tailles automatiquement (déjà branché, à paramétrer).
4. **Explorateur de données / recherche** : une page web pour lancer les `evaluate_signals`, `research_cycle`, `walk_forward` sans éditer le code.

### 🎯 Phase D — **Excellence opérationnelle (continu)**
- Monitoring `health_score` → alerte Telegram si < 60.
- Backups → restauration testée régulièrement (script fourni).
- **Documentation d'exploitation** : `RAILWAY_GUIDE` + `SECURITY` déjà là — ajouter un « runbook » (que faire si tel endpoint tombe).

---

## 5. CONCLUSION

Le projet est **solide, riche et très en avance** sur la moyenne des bots de trading. Le **fix critique** (migration `role`) est appliqué et testé. La priorité immédiate est **technique** : async correct, dé-duplication, découpage de `main.py` — ensuite les **comportements autonomes** (offline RL, narratif proactif, mode consultatif complet) — puis l'**organisation multi-desks réelle** qui le distinguera vraiment.

*Aucune donnée synthétique, aucun trade simulé : tout reste sur données réelles, DEMO fidèle au REAL, seul l'argent est virtuel.*

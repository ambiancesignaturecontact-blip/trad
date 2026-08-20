# 🔬 AUDIT CRITIQUE COMPLET & AUTONOME — QUANT-PORTAL
**Date :** 17 août 2026 — **Méthode :** inspection exhaustive du code (backend, frontend, DB, tests, déploiement, dépendances) + vérifications en exécution réelle. Aucune fonctionnalité n'est considérée comme « terminée » parce qu'elle existe dans l'interface : chaque brique a été tracée jusqu'à son code d'exécution.

> ⚠️ Ce rapport est volontairement **sévère**. Le projet est déjà très complet et fonctionnel
> (58 tests verts, tout démarre, données réelles) — mais « complet dans l'interface » ≠
> « complet dans l'exécution ». Voici l'écart, honnêtement.

---

## A. CE QUI EST RÉELLEMENT SOLIDE (vérifié dans le code, pas dans l'UI)

| Brique | Preuve |
|---|---|
| Données 100 % réelles | Aucun `np.random` dans le flux de trading (vérifié), drapeau `using_fallback_data` explicite |
| Sécurité des fonds | Idempotence (cooldown), confirmation de fill REAL, gates de sécurité, circuit breaker + kill switch qui liquide |
| Auth | JWT + 2FA TOTP, **forcée en mode REAL**, callbacks Telegram vérifiés par `chat_id` (vérifié ligne par ligne) |
| Monitoring | `/metrics` Prometheus complet (prix, equity, PnL, latence, ordres, erreurs) |
| Fiabilité | Boucle blindée (try/except par actif), rate-limiting outbound, graceful shutdown, backups DB quotidiens |
| IA | Pipeline autonome (PPO + MLOps + registry champion/challenger), zéro donnée synthétique |
| Copy trading | Source réelle publique (Hyperliquid ~42k traders), filtres anti-outliers |

---

## B. PROBLÈMES PAR DOMAINE
*(Chaque entrée : 🔴 critique / 🟠 majeur / 🟡 moyen / 🔵 mineur)*

### B1. Architecture & structure du code

| # | Problème | Pourquoi c'est un problème | Prio | Correctif |
|---|---|---|---|---|
| 1 | **`main.py` = 2 214 lignes, monolithe** : trading loop, 22 routes, IA, risk, arbitrage, télégram… tout dedans | Aucune modularité : impossible de tester unitairement la boucle, de la réutiliser, de la faire évoluer sans risque de régression. Chaque correctif devient un chantier. | 🔴 | Découper : `core/trading_engine.py` (boucle), `api/routes/*.py` (routes), `services/*.py`. La boucle devient une classe injectable et testable. |
| 2 | **`config.yaml` est mort** : rien ne le charge (vérifié : 0 référence) | Les vrais paramètres (exposition max, seuils de signal, notional min, drawdown) sont **codés en dur** dans 5 fichiers différents. Tuning = éditer du code + redéployer. | 🟠 | Charger `config.yaml` au démarrage (dataclasses), les constantes codées en dur lisent les valeurs avec défauts. |
| 3 | **`.env` jamais chargé** : aucun `load_dotenv` (vérifié) | En local, le `.env.example` est décoratif — seules les vraies variables d'env marchent. Développement local trompeur. | 🟠 | `python-dotenv` déjà dans requirements → `load_dotenv()` en tête de `main.py` et `db_manager.py`. |
| 4 | **Dossiers en double** : `market_data/` non branché sur la boucle live ; plusieurs implémentations parallèles historiques | Deux vérités sur la qualité des données ; le package testé (`market_data/quality.py`) n'est pas celui utilisé en prod. | 🟡 | Unifier : brancher `DataQualityGate` dans la boucle, supprimer les doublons restants. |
| 5 | Pas de couche **services** (retry, circuit breaker générique, cache) | Chaque appel API réinvente sa gestion d'erreur. | 🟡 | Créer `core/http_client.py` unique (retry tenacity + limiter + latence + cache). |

### B2. Backend & API

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **Aucune validation de bornes** sur les entrées : `SetBalanceRequest.balance: float` accepte -1e18 ; `RiskSettingsUpdate` accepte un drawdown négatif ; `StrategyToggle.name` n'importe quoi | Une requête malveillante (ou une erreur d'UI) peut mettre le solde à −∞, désactiver un risque, casser l'état. | 🔴 | `Field(gt=0, le=...)`, `Literal`/enum pour les noms, validation Pydantic stricte + tests. |
| 2 | **Aucun rate-limit sur les POST** `/api/*` (seuls les appels sortants sont limités) | Un bot peut marteler kill-switch/keys/set-balance → DoS, spam DB, log flooding. | 🟠 | Middleware rate-limit par IP (sliding window) sur les routes d'action. |
| 3 | **Aucun middleware** : pas de log des requêtes, pas d'en-têtes de sécurité, pas de gestion centralisée des exceptions | Impossible de tracer « qui a appelé quoi quand » (audit), pas de protection navigateur (clickjacking), erreurs 500 brutes. | 🟠 | Middleware `request_id` + logging structuré + SecurityHeaders + handler d'exceptions JSON propre. |
| 4 | **Pas de versioning API** (`/api/v1`) | Changer une réponse casse tous les clients (dashboard, mini-app, React). | 🟡 | Router `/api/v1`, alias `/api` temporaire. |
| 5 | **Aucune pagination** sur `/api/history` (120 points max par design) et `orders` tronqué à 15 sans pagination | Données perdues pour l'analyse. | 🟡 | `?limit&offset`, métadonnées `total`. |
| 6 | `POST /api/login` : pas de **rate-limit ni backoff** (bruteforce possible) | Compte admin protégé par un seul mot de passe. | 🔴 | Rate-limit login (5 essais/min/IP) + verrouillage progressif + log des tentatives. |
| 7 | **Pas de reconnexion WebSocket côté client** dans `dashboard.html` (une déconnexion = terminal figé jusqu'au refresh) | Fiabilité UX temps réel. | 🟡 | Auto-reconnect avec backoff + état « reconnexion ». |

### B3. Sécurité

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **Aucun en-tête de sécurité** (CSP, X-Frame-Options, HSTS, nosniff) — vérifié : 0 référence | Le dashboard charge des CDN (Tailwind, Chart.js) → CSP difficile, mais X-Frame-Options/HSTS sont faciles et nécessaires. | 🟠 | Middleware SecurityHeaders (X-Frame-Options DENY sauf Telegram, HSTS en prod). |
| 2 | **CORS non configuré** | OK aujourd'hui (même origine) mais le jour où le terminal est appelé depuis un autre domaine (outil externe, React buildé séparément), tout bloque sans avertissement clair. | 🟡 | `CORSMiddleware` configurable via env `ALLOWED_ORIGINS`. |
| 3 | **Clé JWT par défaut en dur** dans `database/auth.py` (`quant_portal_super_secret_jwt_key_9988`) | Si `JWT_SECRET_KEY` n'est pas définie en prod, n'importe qui peut forger des tokens admin. | 🔴 | Refuser le démarrage en mode REAL / `AUTH_ENABLED` sans `JWT_SECRET_KEY` forte (checklist LOT 62) + génération auto en dev. |
| 4 | **`ADMIN_PASSWORD` par défaut** (`ChangeMe!Institutionnel2026`) | Risque si déployé tel quel. | 🟠 | Bloquer le démarrage avec le mot de passe par défaut en prod + avertissement en dev. |
| 5 | WebSocket `/ws` **sans auth ni limite** (télémétrie publique) | Lecture seule = acceptable ; mais aucune limite par client → un client peut ouvrir 1 000 sockets et faire flood des broadcasts. | 🟡 | Max 1 socket/IP + cap de clients + (option) token optionnel. |
| 6 | **Pas de rotation de clés API** côté stockage (les clés Binance restent chiffrées à vie) | Compromission longue durée indétectable. | 🔵 | Date de dernière rotation + alerte à +90 jours. |

### B4. Base de données

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **Connexion ouverte à chaque appel** : `get_connection()` fait un `psycopg2.connect` neuf par requête (vérifié) | En prod PostgreSQL : latence réseau + overhead par requête, pas de pooling → la boucle 2,5 s fait des dizaines de connexions/min. | 🔴 | Pool psycopg2 (`psycopg2.pool.ThreadedConnectionPool`) ou SQLAlchemy engine ; SQLite : `check_same_thread=False` + WAL. |
| 2 | **Aucun `PRAGMA` SQLite** (WAL, busy_timeout) — vérifié | Risque de `database is locked` si plusieurs writers (tâches de fond + boucle). | 🟠 | `PRAGMA journal_mode=WAL; busy_timeout=5000;` à l'init. |
| 3 | **Migrations maison** (ALTER TABLE à la volée à chaque démarrage) | Fragile, pas de versionnage, pas de rollback. | 🟡 | `alembic` + schéma versionné. |
| 4 | **Pas d'index explicites** sur `orders`, `audit_logs`, `market_candles` (sauf implicites) | Les requêtes `get_all_orders`, `load_candles` ralentissent avec le volume. | 🟡 | Index sur (symbol, timestamp), (mode, timestamp), (user_id). |
| 5 | **Backups sans rétention** : les fichiers s'accumulent indéfiniment | Disque plein à terme sur Railway. | 🟡 | Rotation : garder 14 snapshots, supprimer les plus vieux. |
| 6 | **Pas de procédure de restauration documentée** | Un backup inutile sans procédure. | 🔵 | Script `restore_backup.py` + doc. |

### B5. WebSockets

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **Broadcast non-régulé par client** : `broadcast_telemetry` envoie à tous les clients à chaque tick (200 ms min), sans détection de clients lents | Un client lent bloque/sature ; le serveur retravaille la payload à chaque envoi. | 🟠 | Broadcast asynchrone par client avec file + drop des clients lents + payload mise en cache. |
| 2 | **Pas de heartbeat applicatif** (le ping websockets existe mais pas de message « alive ») | Déconnexions silencieuses des clients mobiles Telegram. | 🔵 | Ping JSON toutes les 30 s + compteur. |

### B6. Market data

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **Yahoo réinterrogé toutes les 2,5 s par actif** (7 actifs = ~2,8 req/s, limité à 3/s — vérifié dans les logs : le limiter sature) | Latence ajoutée, dépendance à un service tiers gratuit, risque de blocage IP. | 🟠 | Cache TTL court (10–30 s) + WebSocket comme source primaire ; Yahoo en secours seulement. |
| 2 | **`market_data/quality.py` (DataQualityGate testé) non branché sur la boucle** | La qualité des données est vérifiée par du code non exécuté. | 🟠 | Brancher le gate dans `process_symbol` + signaler `INVALID` → HALT. |
| 3 | **Candles de repli ±0,05 % générées** quand la source échoue (flag `using_fallback_data`) | Acceptable en DEMO, dangereux en REAL si mal surveillé. | 🟠 | En REAL : refuser de trader un actif en mode repli (le flag existe, le forcer dans la gate REAL). |

### B7. Trading & exécution

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **AUCUN stop-loss / take-profit / trailing** dans le flux d'exécution (vérifié : zéro occurrence hors modèle méta-labeling non utilisé) | C'est LA faille majeure : le bot n'a aucun mécanisme de sortie de protection. Une position peut aller à −50 % sans réaction ; la seule protection est le circuit breaker global (drawdown). | 🔴🔴 | Gestionnaire de positions : SL/TP par position (ATR × k ou % fixe), trailing, vérifié à chaque tick AVANT les nouveaux signaux, ordres SL/TP natifs sur l'exchange en REAL. |
| 2 | **Aucune annulation d'ordres ouverts** : `cancel_order` existe dans les adapters mais n'est jamais appelé par la boucle (vérifié) | En REAL, un ordre market est immédiat, mais toute évolution vers des ordres limites (SOR, slicing) exigera l'annulation. | 🟡 | Boucle de gestion d'ordres (fetch_open_orders → cancel si périmé). |
| 3 | **Frais modélisés en fixe 0,1 % + slippage fixe 0,03 %** en live | Les frais réels varient (maker/taker, VIP), le slippage dépend de la liquidité. | 🟡 | Tarifs par exchange + estimation slippage par volume/carnet (book-walking existe dans les adapters, non utilisé en live). |
| 4 | **`ExecutionManagementSystem` / `OrderManagementSystem` instanciés mais PAS utilisés** : la boucle appelle `client.create_order` directement (vérifié) | Tout le travail OMS/EMS (routing, retry, réconciliation) est décoratif — la réconciliation n'est jamais exécutée en live. | 🟠 | Router l'exécution live à travers `oms.execute_order(...)` + `reconciler.reconcile()` périodique (au moins 1×/min en REAL). |
| 5 | **Le « portefeuille » réel est suivi en mémoire + DB, jamais réconcilié avec l'exchange** en continu | En REAL, une exécution partielle/manuelle côté exchange crée un écart silencieux. | 🟠 | `fetch_balance` + `fetch_positions` périodique + alerte d'écart (la brique `reconciliation/` existe, la brancher). |

### B8. Stratégies & quant

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **Stratégies classiques** : EMA/MACD (trend), RSI/BB (mean-reversion), MM à spread fixe, grid, scalping — toutes des TA classiques | Rien de discriminant ; un bot « élite » doit au moins ajouter : momentum multi-timeframe, carry/funding, microstructure (VPIN/Kyle déjà calculés mais **non utilisés dans les signaux**), on-chain (calculé mais non utilisé comme alpha). | 🟠 | Promouvoir VPIN/Kyle/on-chain de « mesures loggées » à « features de signal » ; ajouter un filtre de régime par stratégie. |
| 2 | **VPIN & Kyle's Lambda calculés mais jamais consommés** (vérifié : loggés seulement) | Travail de microstructure perdu. | 🟡 | Les intégrer dans le scoring de confiance des stratégies. |
| 3 | **Pas de backtest par stratégie individuelle exposé** (le backtest teste le méta-ensemble) | Impossible de savoir quelle stratégie contribue positivement. | 🟡 | Rapport d'attribution par stratégie sur le backtest (le journal a `strategy` depuis le tour 3 — l'utiliser dans le backtest). |
| 4 | **Walk-forward : un seul actif (BTC)** dans le pipeline autonome | Optimisation surreprésentée par un actif. | 🟡 | Étendre à ETH/SOL/or/actions. |

### B9. IA / ML

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **« LSTM-like » = petit LSTM numpy (hidden=8)** ; **PPO = politique linéaire** (pas de couche cachée — vérifié) ; « transformer/GNN » = noms de LOT sans implémentation réelle dans l'ensemble LOT 46 | Le marketing dépasse la réalité : les modèles sont des jouets pédagogiques, pas des modèles institutionnels. Les 5 « modèles » de l'ensemble reçoivent le PnL du même actif. | 🟠 | Honnêteté + montée en gamme progressive : vraie couche cachée PPO, LSTM plus profond, et **un vrai modèle par type** (ou renommer l'ensemble « régimes »). |
| 2 | **Pas de suivi prédiction vs réalité** pour chaque modèle (seul le prédicteur de prix l'est) | Impossible de détecter la dérive d'un modèle spécifique. | 🟡 | Log pipeline `prediction_log` (modèle, prédiction, réel, timestamp) + drift par modèle. |
| 3 | **Pas de métrique de performance IA exposée** (le Sharpe walk-forward est stocké mais pas dans `/metrics` ni le dashboard) | L'opérateur ne voit pas si l'IA s'améliore. | 🟡 | Gauges `quant_ai_oos_sharpe`, `quant_ai_ppo_episodes` + carte dashboard. |
| 4 | **GAN/RLHF (torch) instanciés mais jamais entraînés/utilisés dans le flux** (vérifié : uniquement à l'init) | Deux moteurs « institutionnels » qui ne servent à rien en l'état. | 🟡 | Soit les brancher (GAN → scénarios de stress du sizing ; RLHF → calibration du scaling des positions), soit les retirer de la vitrine. |

### B10. Risk management

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **Pas de limite par position/actif en valeur** (seulement % de capital global via sizing) | Une seule position peut concentrer le risque (le test micro autorise ~80 % d'exposition !). | 🟠 | Max par actif (ex. 25 %) + max corrélés (la matrice existe, l'utiliser en contrainte dure). |
| 2 | **Pas de stress-test périodique** : le Monte-Carlo (10 000 sims) n'est exécuté que sur clic API | Le risque de queue n'est jamais mesuré en continu. | 🟡 | Stress-test quotidien automatique (LOT déjà possible : brancher au scheduler autonome). |
| 3 | **Pas de gestion du levier/marge** en REAL (futures) | Aucun calcul de marge utilisée, liquidation price. | 🟡 | `fetch_positions` avec `liquidationPrice` → alerte avant liquidation. |

### B11. OMS/EMS & réconciliation — voir B7-4/5 (décoratif en live).

### B12. Arbitrage

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **Funding arbitrage : exécution non implémentée** — le module logge l'entrée mais `ENTER_ARBITRAGE` n'ouvre aucune position réelle (vérifié : il crée juste une entrée dans `active_arbitrages` et un audit log) | L'arbitrage « delta-neutre » affiché ne fait rien. | 🟠 | Implémenter l'exécution (spot + perp), ou le déclarer comme « signal uniquement ». |
| 2 | **DEX-CEX : `sign_dex_swap_transaction` signe mais n'envoie jamais** (pas de broadcast on-chain) | Transaction signée ≠ transaction exécutée. | 🟠 | Broadcast via web3 (`send_raw_transaction`) + confirmation. |
| 3 | Arbitrage volatilité/options : **strats purement calculatoires** (IV fixe), aucune exécution possible sans broker d'options | Vitrine. | 🟡 | Étiqueter « simulateur » ou intégrer un broker d'options. |

### B13. Copy trading

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **`start_copying` n'exécute AUCUNE copie** : il enregistre une allocation (vérifié) | Le module affiche des traders réels mais ne copie pas un seul trade. | 🟠 | Deux options honnêtes : (a) afficher « suivi seulement » (le plus sûr), (b) implémenter le mirroring via API dédiée de l'exchange cible (Hyperliquid exchange API : follow orders par trader) — chantier moyen. |
| 2 | Aucun suivi de performance des allocations copiées | Impossible de mesurer l'intérêt. | 🔵 | P&L suivi par allocation. |

### B14. Frontend & UX/UI

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **Deux UIs parallèles** : `templates/dashboard.html` (servi, Tailwind CDN) + `frontend/` React (buildable, non servi — vérifié : aucune route ni montage static) | Duplication, confusion, effort doublé. | 🟠 | Choisir : soit servir le build React à `/app` (définitif), soit supprimer le dossier React. |
| 2 | **Dashboard chargé en CDN** (Tailwind, Chart.js, polices, Telegram SDK) — aucune version épinglée | Dépendance au réseau + rupture possible (un CDN qui change). | 🟡 | Épingler les versions CDN (integrity SRI) ou bundle. |
| 3 | **Aucune accessibilité** : contrastes, aria-labels, focus, réduit-motion | Terminal « élite » devrait l'être. | 🔵 | Passée d'accessibilité + thème clair/sombre. |
| 4 | **Langue française codée en dur** partout | Pas d'i18n. | 🔵 | i18n minimal (fr/en) via fichiers de traduction. |
| 5 | Mini-app : les boutons d'action affichent un succès **local** sans vérifier la réponse serveur (pause/kill set l'état UI puis fetch sans attendre) | L'utilisateur croit que l'action a réussi alors que le serveur a pu la refuser (401, erreur). | 🟠 | Attendre la réponse, afficher l'erreur, re-synchroniser via telemetry. |

### B15. Déploiement & ops

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **Dockerfile mono-stage** : l'image finale contient compilateurs/build-essential (dev deps) | Image gonflée (~1 Go+ avec torch), surface d'attaque. | 🟡 | Multi-stage (builder → runtime slim), `--no-install-recommends`. |
| 2 | **Dépendances non figées** (`>=`) : build non reproductible | Deux déploiements à des dates différentes = versions différentes = bugs fantômes. | 🟠 | `pip freeze > requirements.lock` (ou uv) + build CI avec le lock. |
| 3 | **Pas de gestion de secrets propre côté app** (clés API dans la DB chiffrée avec Fernet, clé elle-même en env) | OK, mais la clé Fernet par défaut est générée et stockée dans le repo de travail si `SECRET_KEY_PATH` par défaut — risque en dev partagé. | 🟡 | Documenter `FERNET_KEY` obligatoire en prod (déjà dans .env.example) + warning LOT 62 si absent. |
| 4 | **Pas de `Procfile` worker séparé pour les tâches de fond** (tout tourne dans le process web) | Les schedulers consomment le même process ; un crash du web tue tout (acceptable en 1 worker, mais pas scalable). | 🔵 | Documenter / scinder en worker dédié. |
| 5 | **Pas de CI de build Docker** (le CI teste pytest mais pas l'image) | L'image peut casser sans être détectée. | 🟡 | Job CI `docker build` + smoke test de l'image. |

### B16. Tests & qualité logicielle

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **58 tests mais des modules critiques SANS test** (vérifié) : `strategies/engine.py`, `risk_engine.py`, `smart_order_router.py`, `multi_exchange_sor.py`, `dynamic_hedging.py`, `correlation_risk.py`, `models/oms_ems.py`, `telegram_bot.py`, `online_model_selector.py`, `adaptive_ensemble_agent.py` | Le cœur stratégie/risque/ordre n'est pas protégé contre les régressions. | 🔴 | Tests unitaires sur ces 10 modules (les plus critiques d'abord : engine, risk, oms). |
| 2 | **Aucun test de la boucle de trading** (la logique décision → ordre est 100 % non testée) | C'est LE code qui touche l'argent. | 🔴 | Extraire `process_symbol()` en fonction pure testable (décision → action) + tests avec fixtures de marché. |
| 3 | **Aucun test d'exécution REAL simulée** (fill/partial/retry) | Les chemins d'erreur exchange ne sont jamais exercés. | 🟠 | Mocks CCXT (fake client) pour ordres réussis/échoués/partiels. |
| 4 | **Pas de linting ni de typage** (ruff/mypy absents, pas de pyproject) | Qualité inégale, bugs de type silencieux. | 🟡 | `ruff check` + `mypy` (léger) dans le CI. |
| 5 | **Pas de tests de charge / benchmark** de la boucle | « Fonctionne » ≠ « tient la charge ». | 🔵 | Benchmark simple : 100 ticks simulés < X ms. |

### B17. Dépendances & environnement

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **`python-multipart` manquant** (requis par FastAPI pour les formulaires) — non requis actuellement, mais ajouté au moindre upload | Préparer. | 🔵 | L'ajouter (léger). |
| 2 | Dépendances inutilisées potentielles (statsmodels, sklearn, web3, ccxt utilisés ; vérifier `aiolimiter` → utilisé) | — | 🔵 | `pip-audit` / `pip check` dans le CI (vulnérabilités). |
| 3 | **Pas d'outil de gestion d'env Python** documenté (venv/uv) | Onboarding fragile. | 🔵 | Documenter `uv` + `uv.lock`. |

### B18. Docs & conformité

| # | Problème | Pourquoi | Prio | Correctif |
|---|---|---|---|---|
| 1 | **Pas de LICENSE, SECURITY.md, CHANGELOG, CONTRIBUTING** | Dépôt public sans licence = juridiquement non réutilisable ; pas de canal de signalement de vulnérabilité. | 🟡 | Ajouter MIT + SECURITY.md (signalement) + CHANGELOG. |
| 2 | README existe ✅ mais pas de diagramme d'architecture | Onboarding difficile. | 🔵 | Diagramme Mermaid (structure + flux de données). |
| 3 | **Aucun disclaimer réglementaire** dans le dépôt (le README en a un ✅) — ajouter dans l'app | Conformité. | 🔵 | Bandeau dans le dashboard. |

---

## C. FONCTIONNALITÉS MANQUANTES (hors bugs)

| # | Fonction | Priorité | Pourquoi |
|---|---|---|---|
| 1 | **Gestionnaire de stop-loss / take-profit / trailing par position** | 🔴 | Protection de capital élémentaire, absente (cf. B7-1) |
| 2 | **Rapport P&L quotidien automatisé** (par stratégie, par actif, par mode) + export CSV/PDF | 🟠 | L'opérateur n'a aucun « relevé de compte » |
| 3 | **Alertes de prix / signaux** configurables (seuils, push Telegram) | 🟠 | Le bot notifie mais ne laisse pas l'utilisateur se fixer des alertes |
| 4 | **Mode paper-trading séparé avec réconciliation** (frais réels simulés) | 🟠 | La « preuve » de rentabilité sans risque |
| 5 | **Historique/export des décisions** (pourquoi le bot a acheté) consultable | 🟠 | Auditabilité (le journal existe, pas d'UI de consultation) |
| 6 | **Dashboard de santé** (latence, erreurs, dérive, backups) | 🟡 | Les métriques existent ; aucune UI ne les montre |
| 7 | **Multi-utilisateurs / rôles** (viewer/trader/admin — les rôles existent dans auth.py, non exploités) | 🟡 | SaaS-ready annoncé, mono-utilisateur réel |
| 8 | **Webhooks** (TradingView, alertes externes → ordres) | 🟡 | Interopérabilité standard d'un terminal |
| 9 | **Backtesting multi-actifs simultané** | 🟡 | Évaluer la corrélation du portefeuille |
| 10 | **Market replay** (rejouer l'historique pour debug/validation) | 🔵 | Validation des décisions passées |

---

## D. MES IDÉES (features que tu n'as pas demandées)

1. **« Mode Autopilote » gradué** : un curseur DEMO → PAPER → REAL avec plafonds de taille progressifs (ex. REAL limité à $X/jour tant que la validation n'est pas passée). Le bot s'auto-gate : impossible de passer en REAL sans un « permis » (N jours de paper profitable + seuils).
2. **« Concierge de risque » Telegram** : chaque soir, un résumé structuré (P&L du jour par stratégie, positions ouvertes + SL/TP proposés, CVaR, alertes) + un bouton « clôturer tout ».
3. **Carnet d'ordres mental (reasoning log)** : à chaque décision, le bot enregistre les 3 raisons principales (signal, régime, risque) — rendu dans un timeline UI. C'est LA différence entre un bot et un terminal « professionnel » (auditabilité).
4. **Système de « régimes de marché » visibles** : carte du régime actuel (HMM) + probabilité, avec surimpression sur le graphique de prix.
5. **Score de santé du bot** (0–100) : composite (connexions, qualité données, dérive IA, drawdown, erreurs) affiché en gros dans le header — un opérateur sait en 1 seconde si tout va bien.
6. **Test A/B de stratégies en paper** : deux stratégies identiques sur deux budgets virtuels, le bot compare et promeut la gagnante (champion/challenger déjà en place pour les modèles — l'étendre aux stratégies).
7. **Export « rapport d'audit réglementaire »** : un PDF généré (positions, P&L, conformité FIFO) prêt pour un comptable — le moteur fiscal existe déjà.
8. **Recommandations de rééquilibrage** : le portefeuille suggère des actions (acheter X pour revenir à l'allocation cible) au lieu de trader en continu — moins de frais, plus institutionnel.
9. **Détection d'anomalie de marché** (flash crash, manipulation) via VPIN extrême → bascule automatique en mode défensif (réduction d'exposition) — le VPIN est déjà calculé, il suffit de le brancher.
10. **Journal de bords versionné** : chaque changement de paramètre (risque, stratégie, mode) horodaté et lié à l'opérateur — traçabilité complète (les audit-logs existent, les enrichir avec les changements de config).

---

## E. SYNTHÈSE FINALE

### 1. Problèmes critiques (à corriger avant toute mise en prod REAL)
| # | Problème | Correction |
|---|---|---|
| C1 | **Aucun stop-loss / take-profit** dans l'exécution | Gestionnaire de positions avec SL/TP/trailing (B7-1) |
| C2 | **OMS/EMS + réconciliation non branchés** sur l'exécution live | Router les ordres via OMS + reconcile périodique (B7-4/5) |
| C3 | **Validation d'entrées API absente** (balances négatives, risques invalides) | Pydantic stricte avec bornes (B2-1) |
| C4 | **Bruteforce login possible** | Rate-limit + backoff sur /api/login (B2-6) |
| C5 | **JWT / mot de passe admin par défaut acceptés** | Blocage en prod (B3-3/4) |
| C6 | **Pooling DB absent** (connexion par requête) | Pool psycopg2 / SQLite WAL (B4-1/2) |
| C7 | **Cœur trading non testé** (boucle, stratégies, risk, oms) | Tests unitaires + extraction `process_symbol` (B16-1/2) |

### 2. Fonctionnalités manquantes (par valeur)
SL/TP & gestion de positions → rapport P&L quotidien → paper-trading réconcilié → alertes configurables → UI santé → multi-utilisateurs → webhooks → backtest multi-actifs → reasoning log.

### 3. Améliorations prioritaires (impact / effort)
1. **Config externe** (config.yaml + .env réellement chargés) — petit effort, gros gain ops
2. **Tests du cœur** (engine, risk, boucle) — moyen effort, protège l'argent
3. **SL/TP + gestion de positions** — moyen effort, protège l'argent
4. **Pool DB + WAL** — petit effort, perf
5. **Dépendances figées + Docker multi-stage + CI Docker** — petit effort, fiabilité déploiement
6. **Rate-limit login + validation stricte** — petit effort, sécurité

### 4. Mes meilleures idées (top 5)
1. **Mode Autopilote gradué (DEMO→PAPER→REAL auto-gaté)** — transforme le bot en produit de confiance
2. **Reasoning log** (pourquoi chaque trade) — l'auditabilité qui fait la différence « élite »
3. **Score de santé du bot** (0–100) — l'opérateur sait tout en 1 seconde
4. **Concierge de risque Telegram quotidien** — le bot devient un assistant, pas un outil
5. **Détection de flash-crash via VPIN → mode défensif auto** — le microstructure déjà calculé devient un filet de sécurité

### 5. Roadmap ordonnée (dans l'ordre où je le ferais)

**Phase 1 — SÉCURISER (semaine 1)** : C7 (tests du cœur) → C3 (validation API) → C4/C5 (auth) → C6 (DB pooling/WAL) → C1 (SL/TP) → C2 (OMS/réconciliation branchés)

**Phase 2 — PROFESSIONNALISER (semaine 2)** : config.yaml + .env → dépendances figées + Docker multi-stage + CI image → rapport P&L quotidien → paper-trading réconcilié → index DB + rétention backups → pagination API

**Phase 3 — DIFFÉRENCIER (semaine 3-4)** : Autopilote gradué → reasoning log → score de santé → concierge Telegram → alertes configurables → régimes visibles

**Phase 4 — ÉLARGIR (mois 2)** : multi-utilisateurs/rôles → webhooks → backtest multi-actifs → vraie exécution des arbitrages (funding/DEX) ou étiquetage honnête « signal only » → market replay → i18n/accessibilité

**Phase 5 — VALIDER LA RENTABILITÉ (continu)** : backtest walk-forward multi-actifs ≥ 2 ans → paper-trading 4–8 semaines → n'appliquer le mode REAL qu'après passage des gates d'Autopilote.

---

*Chaque correctif des phases 1–2 est faisable dans ce dépôt tel quel ; les phases 3–5 sont des chantiers produits. La rentabilité finale ne se code pas : elle se prouve par la phase 5.*

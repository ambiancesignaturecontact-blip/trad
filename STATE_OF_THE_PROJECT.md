# STATE OF THE PROJECT — QUANT-PORTAL

> **Document VIVANT** — mis à jour par DIFF (une ligne par item + preuve),
> jamais réécrit de zéro à chaque session. Si vous ajoutez un item, ajoutez
> la preuve (commit, test, log) dans la même modification.
> Les 12 anciens documents d'audit/vision sont archivés dans
> `docs/archive/audits-2026-08/` (historique conservé, plus aucune ambiguïté
> sur « quel document fait autorité » : c'est celui-ci).

Dernière mise à jour : 2026-08-20 · Repo : `trad` · Langue : français

---

## 0. Preuves reproductibles (commandes)

| Preuve | Commande | Résultat au 2026-08-20 |
|---|---|---|
| Tests | `python -m pytest tests/ -q` | **459 passed** (~34 s) |
| Safety check | `python automated_production_safety_check.py` | ALL DEEP EDGE-CASE CHECKS SUCCESSFUL |
| Couverture cœur (P1-15) | `python scripts/check_coverage.py` | main 34 % / engine 84 % / risk_pipeline 98 % (seuils 30/80/95) |
| CVE (P1-9) | `python scripts/check_vulnerabilities.py` | 0 vulnérabilité haute/critique (80 dépendances) |

---

## 1. ✅ FAIT — items de l'audit indépendant (preuves)

| Item | Ce qui a été fait | Preuve |
|---|---|---|
| P0-1 | Fausse allégation « MevShield On-Chain active » retirée ; messages DEX-CEX honnêtes (signal-only / broadcast) | commit `58a0060` · test `test_lot1_p0_audit.py` |
| P0-2 | AUTH forcée automatiquement sur déploiement non-local (PORT/RAILWAY_*) ; bypass local DEMO préservé | `58a0060` |
| P0-3 | Mot de passe admin auto-généré JAMAIS loggé : livré par Telegram DM ou fichier 0600 | `58a0060` |
| P0-4 | Instrumentation `final_scale` (p10/p50/p90, persistance 5 min, facteur limitant) + **plancher anti-empilement 15 %** (0,8¹⁵≈3,5 % éliminé) | `d23144d`, `765abfb`, `f2aac61`, `1200d8b` · `test_lot2_p0_audit.py` |
| P0-5 | Scripts CLI alignés sur le live (hidden_dim 24, défaut constructeur 24, garde-fou `bias_audit` avec rejet) + **données 100 % réelles** (plus aucun fallback synthétique) | `d23144d`, `765abfb` · `run_*.py` |
| P1-7 | Découpage de `main.py` : **6136 → 4892 lignes** (59 routes API → `api/routes.py`, 6 schedulers → `schedulers.py`, extraction AST reproductible) | `3fdab5d` · `test_lot7_split.py` |
| P1-8 | `core/config.py` branché : **3 → 11 fichiers** (portefeuille, contrepartie, coûts, order flow, vol targeting, CVaR, drawdowns, multi-source) | `8f6b010` · `test_lot3_config.py` |
| P1-9 | CI sur `requirements.lock` + `requirements-dev.txt`, Python 3.11 aligné prod, **pip-audit bloquant** sur CVE haute/critique | `c4b3926` · `test_lot4_ci.py` |
| P1-10 | Corrélation des SIGNAUX inter-stratégies (facteur max(0,5 ; 1−corr)) — l'angle mort §4.4 est fermé | `30eb80f` · `test_lot5_engine_audit.py` |
| P1-11 | `update_pnl_attribution` par **Sharpe déflaté** (López de Prado) + plancher d'échantillon (±20 % sous 20 trades) | `30eb80f` |
| P1-12 | Bandit Thompson : **oubli 0,98** (non-stationnaire) + **tirage figé** par cycle de décision (60 s) | `30eb80f` |
| P1-13 | Slippage live par **book-walking** du carnet (sizing + SOR par venue) | `b3a57e5` · `test_lot6_slippage_exposure.py` |
| P1-14 | `max_exposure_normal` **0,25 → 0,75** (vs 0,25 par actif) ET branché réellement (constante morte corrigée) | `b3a57e5` |
| P1-15 | Seuils de couverture **par fichier** (main 30 / engine 80 / risk_pipeline 95) | `c4b3926` · `test_lot4_ci.py` |
| P2-16 | `market_data/macro_calendar.py` → shim de dépréciation vers `models/macro_calendar.py` (plus de FOMC simulé) | `b792eb3` · `test_lot8_cleanup.py` |
| P2-17 | `models/telegram_bot.py` → `bot/telegram_bot.py` (28 Ko, pas un modèle quantitatif) | `b792eb3` |
| P2-18 | Shim `db_manager.py` racine supprimé ; 5 importeurs → `database.db_manager` | `b792eb3` |
| P2-19 | **Violation réelle corrigée** : RLHF (ÉDUCATIF) alimentait le sizing → `rlhf_scale=1.0` constant, module non chargé en prod + garde-fou mécanique (mapping facteur→module, spy AST) | `fe43fe8` · `test_lot9_educational.py` |
| P2-20 | Ce document + archive des 12 documents d'audit | commit `ad50bea` · `test_lot10_state.py` |

### Fixes supplémentaires (hors liste P0/P1/P2)

| Fix | Détail | Preuve |
|---|---|---|
| VPIN aberrant (6 988 465) | Bug racine `bucket_size_volume` fixe → buckets de volume égal, VPIN borné [0,1] ; modulate de conviction sécurisé (hors [0,1] = neutre) | `1200d8b` · `test_lot2_p0_audit.py` |
| Fidélité DEMO == REAL | SOR multi-venue exécuté en DEMO + `simulate_paper_fill` TOUJOURS book-walké (avant : prix fixe ±3 bps si carnet présent) | `3fdab5d` · `test_lot7_split.py` |
| Collecte final_scale | Persistance DB (survit aux redémarrages, perte max 5 min), endpoint `/api/v1/final-scale`, `/api/v1/paper-validation` | `765abfb`, `f2aac61` |
| Pollution DB par les tests | Tests mockent la DB (leçon : le p50=11,45 % initial était des données de test) | `f2aac61` |

---

## 2. 🔄 EN COURS / NON TERMINÉ (processus, pas du code)

| Item | État réel au 2026-08-20 | Ce qui manque |
|---|---|---|
| **P0-6** : paper-trading daté et CONTINU avant REAL | Tracker opérationnel (`/api/v1/paper-validation`) : **2 / 28 jours** marqués, `validated: false` | 26 jours de fonctionnement réel continu (le bot doit tourner) |
| Observation `final_scale` 24-48 h (P0-4 diagnostic) | Collecte en cours (serveur live, persistance 5 min) : p50 = 0,15 (plancher actif), **facteur limitant = conviction** (|signal| ~0,10) | 24-48 h de données continues ; décision sur le seuil de conviction à trancher sur ces données |
| Preuve book-walking / SOR-DEMO en conditions réelles | Mécanisme testé + simulé ; en attente de trades DEMO réels (le pipeline trade rarement, signaux faibles) | Logs `SOR_CHOICE_DEMO` / `book_slippage_bps` sur vrais trades |
| Découpage `main.py` | Étape 1 faite (extraction routes + schedulers, −1244 lignes) | Découplage fin des dépendances (`from main import *`) — optionnel |

---

## 3. ⚠️ LIMITES CONNUES / DETTE ASSUMÉE

| Limite | Détail | Statut |
|---|---|---|
| `ruff` cassé | `.ruff.toml` contient une règle inconnue (`W503`) → ruff ne démarre pas ; CI non bloquant, assumé et documenté | Lot dédié nécessaire |
| Test flaky préexistant | `test_meta_allocation_dominance` dépend du tirage uniforme du bandit (~18 % d'échec théorique) | À corriger (seed ou seuil déterministe) |
| Warning MLOps | `Challenger/champion comparison failed: invalid literal for int() with base 10: ''` — attrapé, non bloquant | À investiguer (source ≠ `mlops_n_trials`, déjà robuste) |
| Torch absent du sandbox | GAN (LOT 54) et RLHF (LOT 55) en fallback neutre — cohérent avec l'étiquette ÉDUCATIF | Assumé |
| `main.py` encore ~4 900 lignes | Étape 1 du découpage seulement (le cœur `live_trading_loop` reste dans main) | Étape 2 optionnelle |
| Contention SQLite | Ne pas lancer `pytest` pendant que le serveur uvicorn écrit dans la même `trading_platform.db` (500 s+ vs 29 s) | Documenté |

---

## 4. Registre d'honnêteté (module_honesty)

- **PRODUCTION (16)** : multi_source_price, risk_pipeline, risk_state_machine, order_flow, execution_sor, position_lifecycle, portfolio_allocator, macro_calendar, sentiment, counterparty_risk, reconciliation, watchdog, cost_accounting, attribution, scenario_stress, bias_audit.
- **EXPÉRIMENTAL (4, gardés)** : regime_confidence, mlops_challenger, causal_discovery, meta_attribution.
- **ÉDUCATIF (4, JAMAIS dans le sizing — verrouillé par test P2-19)** : rlhf, gan_scenarios, options_volatility, llm_narrative.

---

## 5. Comment mettre à jour ce document

1. Une ligne par item, avec **preuve** (commit `xxxxxxx`, fichier de test, log réel).
2. Déplacer un item de « EN COURS » vers « FAIT » uniquement avec une preuve
   (test vert, log, chiffre) — jamais « parce que ça devrait marcher ».
3. Ne jamais ajouter de section de complétion globale : l'état réel du
   projet est celui de ce document, y compris ses limites.

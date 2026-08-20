# STATE OF THE PROJECT — QUANT-PORTAL

> **Document VIVANT** — mis à jour par DIFF (une ligne par item + preuve),
> jamais réécrit de zéro à chaque session. Si vous ajoutez un item, ajoutez
> la preuve (commit, test, log) dans la même modification.
> Les 12 anciens documents d'audit/vision sont archivés dans
> `docs/archive/audits-2026-08/` (historique conservé, plus aucune ambiguïté
> sur « quel document fait autorité » : c'est celui-ci).

Dernière mise à jour : 2026-08-20

## 🎓 Principes institutionnels appliqués (savoir public de référence)

| Principe | Source publique | Application dans QUANT-PORTAL |
|---|---|---|
| **Meta-labeling** : séparer la direction (side) du sizing (taille) — un modèle secondaire filtre les faux positifs et pilote la taille | López de Prado, *Advances in Financial Machine Learning* (2018) ; [Wikipedia Meta-Labeling](https://en.wikipedia.org/wiki/Meta-Labeling) | `meta_label_filter()` : filtre les signaux faibles (en DEMO : réduit la taille ; en REAL : bloque) — la direction vient du méta-moteur, la taille du Kelly + pipeline |
| **Fractional Kelly** : le Kelly plein maximise la croissance log mais produit des drawdowns sévères ; ½-¼ Kelly sacrifie ~25-56 % de croissance pour réduire massivement la volatilité | Thorp (2011), MacLean et al. (2010) ; pratiques professionnelles (Ernie Chan, AQR) | `KELLY_FRACTION = 0.15` (≈¼ Kelly) + win rate plancher/plafond 0.45-0.65 (estimation d'edge incertaine → réduire) |
| **Risk parity** : équilibrer le RISQUE entre classes d'actifs, pas le capital | Bridgewater All Weather (1996/2012) ; Bob Prince, "Risk Parity Is About Balance" | `risk_parity_weights()` dans `MetaAllocationEngine` + corrélation des signaux (P1-10) pour ne pas concentrer sur 2-3 facteurs latents |
| **Volatility targeting** : maintenir une volatilité cible constante, taille = cible/vol réalisée | Pratique institutionnelle (Man AHL, AQR) ; Kelly continu f* = μ/σ² | `volatility_scale_factor()` (cible tick 0.04 %, bornes 0.25-2.0) |
| **Base currency (multi-devise)** : le portefeuille se mesure dans UNE devise de référence ; chaque valeur étrangère convertie au taux du jour | IFRS (spot rate au jour de transaction) ; pratiques comptables multi-devises | `core/fx.py` : devise de compte configurable, conversion réelle er-api, affichage honnête si FX indisponible |
| **Triple-barrier labeling** : les sorties réelles (SL/TP/temps) définissent le label, pas un horizon fixe | López de Prado, AFML ch. 3 | `MetaLabelingTripleBarrier` (modèles/lopez_de_prado.py) — utilisé pour l'apprentissage |
| **Drawdown management** : circuit breakers quotidiens + lifetime, drawdowns par taille de compte | Pratique institutionnelle (risque de ruine) | `RiskManager.check_circuit_breaker()` : 2,5-18 %/jour, 8-35 % lifetime selon le capital |
| **Déflated Sharpe Ratio (DSR)** : corriger le Sharpe de la sélection/fouille de données | López de Prado, AFML ch. 8 | `calculate_deflated_sharpe_ratio` — promotion challenger OOS + attribution PnL (P1-11) |

> Ces principes sont des références publiques de qualité ; ils sont appliqués de façon critique :
> ce qui est robuste (fractional Kelly, DSR, triple-barrier) est en production ; ce qui est
> théorique (Kelly plein, optimisations fragiles) est volontairement écarté ou borné.
 · Repo : `trad` · Langue : français

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
| P2-20 | Ce document + archive des 12 documents d'audit | commit `643b106` · `test_lot10_state.py` |

### Fixes supplémentaires (hors liste P0/P1/P2)

| Fix | Détail | Preuve |
|---|---|---|
| VPIN aberrant (6 988 465) | Bug racine `bucket_size_volume` fixe → buckets de volume égal, VPIN borné [0,1] ; modulate de conviction sécurisé (hors [0,1] = neutre) | `1200d8b` · `test_lot2_p0_audit.py` |
| Fidélité DEMO == REAL | SOR multi-venue exécuté en DEMO + `simulate_paper_fill` TOUJOURS book-walké (avant : prix fixe ±3 bps si carnet présent) | `3fdab5d` · `test_lot7_split.py` |
| Collecte final_scale | Persistance DB (survit aux redémarrages, perte max 5 min), endpoint `/api/v1/final-scale`, `/api/v1/paper-validation` | `765abfb`, `f2aac61` |
| Pollution DB par les tests | Tests mockent la DB (leçon : le p50=11,45 % initial était des données de test) | `f2aac61` |
| Ruff réparé et bloquant | `W503` retiré (règle supprimée de ruff), ~2000 corrections sûres, 3 vrais bugs F821 corrigés (stress test du scheduler autonome ne s'exécutait JAMAIS), **`ruff check .` = 0 erreur**, CI bloquant | `359c13c` · `test_lot4_ci.py` |
| Test flaky corrigé | `test_meta_allocation_dominance` figé via cache de tirage Thompson (8/8 vertes, avant ~40 % d'échec) | `359c13c` |
| Warning MLOps éliminé | **Vrai bug trouvé** : `get_setting(key, "défaut")` passait le défaut en `user_id` → `int("")` ; appels corrigés + test de régression | `359c13c` · `test_lot9_educational.py` |
| Contention SQLite réglée | Tests isolés sur `tests/test_trading.db` (base fraîche par session, `.gitignore`) — pytest peut tourner pendant que le serveur écrit | `359c13c` · `tests/conftest.py` |
| Étape 2 du découpage | `telemetry.py` extrait (`serialize_helper`, `compile_telemetry_data`, `broadcast_telemetry`) — main.py 4918 → 4696 lignes | `359c13c` |
| Fix prod (logs Railway) | `/api/v1/health` 500 corrigé (imports directs des symboles perdus par le nettoyage ruff F401 — `compute_health_score` etc.) ; supervisor silencieux en pause volontaire ; **test : toutes les routes GET répondent sans NameError** | `ab251aa` · `test_routes_health.py` |
| Mini-app fiable (mandat) | **Cause trouvée** : pas de polling + état initial 100 % simulé (chiffres fictifs affichés !). Corrigé : polling REST 5 s (parité fallback dashboard), état initial honnête (—/chargement), indicateur de fraîcheur 🟢/🟡/🔴, erreurs visibles. Tests verrouillés | commit mandat · `test_miniapp_50usd.py` |
| Accès marchés vérifié | Crypto (Coinbase/Kraken/OKX/CoinGecko), Or XAUUSD (Yahoo GC=F + gold-api), Forex EURUSD (Yahoo + er-api), Actions AAPL/TSLA (Yahoo) — toutes les classes d'actifs alimentées par des sources RÉELLES (Binance/Bybit géobloqués en sandbox, OK en prod) | test live 2026-08-20 |
| Multi-devise (compte) | Devise de compte configurable (`account.currency` : USD/EUR/GBP/JPY/CHF/...) + conversion RÉELLE (open.er-api.com, cache 5 min) de balance/équité/PnL dans la devise du compte (principe base currency IFRS/GAAP) ; l'interne reste en USD ; UI mini-app + dashboard affichent la devise ; FX indisponible -> affichage USD honnête (jamais de taux inventé) | commit multi-devise · `test_multicurrency.py` (11 tests) |
| Intelligence Axe 1 | Conviction CALIBRÉE par le win rate réel (`calibrated_conviction`, interpolation 0.45→x0.60 … 0.65→x1.25, bornée, neutre sans historique) — le sizing reflète la probabilité calibrée de succès (meta-labeling, López de Prado) au lieu de \|signal\| brut | `18909c6` · `test_intelligence.py` |
| Intelligence Axe 2 | `RegimeSwitchingAllocator` ADAPTATIF : poids statiques = a priori, performance RÉELLE par (régime, stratégie) ajuste en ligne (EMA 0.2, min 5 obs, shift borné ±30 %) — non-stationnarité sans sur-réaction au bruit | `18909c6` |
| Intelligence Axe 3 | Risk budget CONDITIONNEL au régime : `regime_risk_scale` (Bear High Vol ×0.70, Erratic ×0.80, calme ×1.0, inconnu neutre) intégré à `total_risk_budget`/`rebalance` | `18909c6` |
| Trading 50 $ prouvé | 44 ordres FILLED sur la session (Grid Trading SOL, allers-retours ~20 s, ~3 $/trade — le min-notional remonté) ; l'impression « ne trade pas » vient de la petitesse/rapidité + NO_TRADE fréquents (régime Bear Trend High Vol, prudence par design) | journal DB 2026-08-20 |

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
| Torch absent du sandbox | GAN (LOT 54) et RLHF (LOT 55) en fallback neutre — cohérent avec l'étiquette ÉDUCATIF | Assumé |
| `main.py` encore ~4 700 lignes | Étape 2 partielle faite (télémétrie extraite) ; le cœur `live_trading_loop` reste dans main | Étape 3 optionnelle |
| `from main import *` dans les modules extraits | Pattern de l'étape 1 du découpage (F405/E402 ignorés per-fichier dans ruff) | À éliminer si découplage fin |

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

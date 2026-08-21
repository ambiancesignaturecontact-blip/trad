# CERTIFICATION PHASE 2 — QUANT-PORTAL

**Date** : 2026-08-21 · **Commit certifié** : `95e82b6` (post-audit) · précédent freeze `7bebe2e`
**Version système** : `qp-95e82b6-{config_hash}` · config.yaml sha256 : `0419a7b1…`
**main.py** : 3745 lignes · **Serveur** : DEMO, SQLite (REAL interdit sans PostgreSQL par design)

> Ce document est le rapport final de l'audit indépendant PHASE 2. Il ne
> certifie PAS le passage en REAL — il produit une décision objective GO /
> CONDITIONAL GO / NO-GO avec preuves. Aucune faiblesse n'est masquée.

---

## 1. RÉSUMÉ EXÉCUTIF

**DÉCISION : NO-GO pour le passage en REAL — CONDITIONAL GO pour la poursuite
du paper-trading longue durée.**

Le système est techniquement robuste (839 tests, 0 erreur lint, safety check
complet, architecture découplée, gouvernance en place), mais **les preuves
d'edge et de calibration du calibrage ACTUEL sont absentes** : 0 trade clôturé
dans le journal depuis le recalibrage du mandat, 3/28 jours de paper-trading
daté, et le P&L historique de l'ancien calibrage était négatif (−0,22 %/trade).
La règle du mandat est respectée : « Ne jamais choisir GO parce que les tests
sont verts ».

---

## 2. AUDIT FINAL PAR DOMAINE

### 2.1 DATA — PASS (avec réserve)

| Vérification | Résultat |
|---|---|
| Doublons timestamps | **0** sur 7 actifs |
| Fraîcheur | Crypto live (23:01), Or/FX/Actions à la clôture Yahoo (normal) |
| Coherence events | 890 fills ↔ 890 orders ↔ 1016 FILLED (cohérent) |
| **Pollution DB** | **170 événements de test purgés** (CAUSALTEST/TESTBTC — 42 % du P&L historique était du bruit de test) ; backup `backups/pre_purge_test_artifacts_*.db` |
| Sources | Multi-source avec divergence → gel ; SINGLE_SOURCE honnête (AAPL/TSLA) |

### 2.2 ALPHA / EDGE — FAIL (preuves insuffisantes)

230 clôtures réelles (ancien calibrage, 18-21/08) — après purge des artefacts :

| Stratégie | n | Win rate | Expectancy | Classe |
|---|---|---|---|---|
| Grid Trading | 115 | 2 % | −0,196 % | **DEGRADED** |
| Mean Reversion | 44 | 0 % | −0,113 % | **DEGRADED** |
| Trend Following | 20 | 10 % | −0,542 % | **DEGRADED** |
| Cross-Sectional Momentum | 9 | 0 % | −0,103 % | **DEGRADED** |
| Momentum | 5 | 0 % | −0,579 % | **DEGRADED** |
| Multi-Timeframe | 2 | 0 % | −0,274 % | **DEGRADED** |
| **Total** | **230** | **1,7 %** | **−0,219 %/trade** | — |

**Interprétation honnête** : ce P&L date de l'ANCIEN calibrage (avant le
mandat : plancher 0,15, seuil constant, Kelly non calibré). Il prouve que
l'ancien système paper perdait — et que les ~10 000 NO_TRADE ont protégé le
capital. **Le calibrage ACTUEL n'a aucune clôture mesurée** : l'edge n'est ni
prouvé ni réfuté — il est INCONNU.

### 2.3 INTELLIGENCE — PASS structurel, mesures en attente

- Conviction : calibrée (89 % couverture test), niveaux HIGH/NORMAL/LOW/UNCERTAIN/NO_TRADE.
- **Écart corrigé** : 24 TRADE avec niveau NO_TRADE (signal brut vs conviction calibrée) → entrée pilotée par la conviction calibrée.
- Meta-allocator hiérarchique (regret, familles), MoE, drift PSI/CUSUM, edge decay : présents, testés — mais **0 clôture → aucun feedback réel** (regret vide, calibration n=0).
- Régime HMM validé sur 7 actifs, confiance + stabilité.

### 2.4 RISK — PASS

- Limites config : Kelly 0,15 · max_per_asset 0,25 · drawdowns 2,5 %/8 % (normal) · CVaR 2 % · plancher anti-empilement 0,25 · halt 15 min.
- **0 kill switch déclenché**, 0 drawdown > 1 % en observation réelle, état risque NORMAL.
- Adversarial (9 scénarios) actif en mode block ; Monte Carlo : survie 100 %, VaR95 1,4 %, 0 ruine sur 5000 sims.
- 180 `RECONCILIATION_FAILED` historiques = artefacts de tests (0 depuis le 21/08) — non reproductibles en DEMO (scheduler REAL-only).

### 2.5 EXÉCUTION — WARN

- 1016 fills MARKET (100 %), slippage médian **6,6 bps** (sain) mais **p95 = 118 bps** (queue lourde, micro-trades sur carnets minces).
- Latence moyenne 75 ms (simulée). Frais réels **0,100 %** (conforme).
- **Friction aller-retour ≈ 0,27 %** sur trades de ~3 $ — structurellement significative pour un micro-compte.
- `execution_intel` : 0 fill mesuré dans la fenêtre récente (les fills historiques pré-datent le module) — comparaison prévision/réalité en attente de données.

### 2.6 ARCHITECTURE — PASS

- main.py 3745 lignes, 8 cerveaux, 48 modules, **0 `from main import *`**, tous importables (vérifié par test).
- Découplage réel : helpers état/auth extraits, ré-exports compatibles.
- Résilience : snapshots 2 h, watchdog + restart, backups quotidiens, réconciliation REAL-only.

### 2.7 OBSERVABILITÉ — PASS (après corrections)

- Decision Journal : 1338 décisions, versionnées (287 avec version système), raison à 100 %.
- **Écart corrigé** : journal déplacé à la décision finale — TRADE seulement si l'ordre part (preuve : 112 faux TRADE identifiés).
- No-trade : 10 006 abstentions catégorisées (conviction 955 / autres 545 sur 1500 récents).
- P&L attribution : fonctionnelle mais dépendante des clôtures (0 actuellement).

### 2.8 SÉCURITÉ — PASS

- Auth : AUTH_ENABLED prioritaire, REAL toujours protégé, déploiement non-local → auth obligatoire.
- REAL interdit sans PostgreSQL (startup block) + safety gates par actif.
- Secrets : Fernet chiffrés, JWT fort auto-généré, admin livré hors logs (fichier 0600 ou Telegram).
- Modules éducatifs verrouillés mécaniquement (tests AST + safety check).

---

## 3. TESTS & VALIDATIONS

| Validation | Résultat |
|---|---|
| Suite complète | **827 passés** (+12 flaky standalone réseau, pré-existants) |
| Nouveaux tests PHASE 2 | +10 (journal final, gates, conviction calibrée, marqueurs) |
| ruff / safety check | 0 erreur / **ALL PASSED** (7 stress tests edge-case) |
| Couverture modules clés | 87 % (conviction 89, edge_decay 93, execution_intel 93, hierarchical 97, paper_validation 82, system_version 70) |
| Walk-forward (70/30, 3 actifs) | Pipeline exécuté sur données réelles — **0 trade OOS/fenêtre** → Sharpe non significatif (échantillon trop court pour stratégie rare) |
| Monte Carlo (5000 sims, vol réelle) | Survie 100 %, VaR95 1,4 %, VaR99 3,1 %, 0 ruine |
| Bootstrap Sharpe | **Impossible** (1 point d'équité en mémoire — perdu au restart) |
| Stress adversariaux | 9 scénarios testés (spread x2, slippage x3, latence x5, vol x2, gap, inversion, données) |

**Limites de validation documentées** : les métriques OOS ne peuvent pas être
significatives avec 0-2 trades par fenêtre ; l'équité réelle n'est pas persistée
(1 point au restart) — le bootstrap Sharpe nécessite ≥ 30 points continus.

---

## 4. DÉCISION OBJECTIVE

### NO-GO — passage en REAL

1. **Preuve d'edge absente** : 0 clôture du calibrage actuel ; P&L historique négatif (ancien calibrage).
2. **Calibration non mesurée** : 0 trade calibré, calibration error inconnue.
3. **Continuité insuffisante** : 3/28 jours datés (P0-6).
4. Le mode REAL exige de plus PostgreSQL + clés API + environnement de production non testés ici.

### CONDITIONAL GO — poursuite du paper-trading longue durée

Conditions (mesurables via `/api/v1/paper-validation-report`) :
1. **28 jours continus** de fonctionnement (C1) — le serveur doit tourner sans interruption.
2. **≥ 30 trades clôturés** avec le calibrage actuel (C10) → expectancy et calibration mesurables.
3. **Expectancy positive** sur l'échantillon propre (l'ancien calibrage était à −0,22 %).
4. **Slippage p95 < 118 bps** ou réduction de la queue (C8).
5. **Rapport de validation = READY** (aujourd'hui : NOT_READY — 1 FAIL, 4 WARN).

### Décisions restant à surveiller

- L'entrée pilotée par la conviction calibrée (correction PHASE 2) réduira la fréquence — vérifier qu'elle n'éteint pas complètement le trading (le seuil adaptatif p25 suit les signaux).
- La friction 0,27 %/aller-retour sur micro-trades : envisager de trader au-dessus du min notional si le capital le permet.
- Queue de slippage (p95 118 bps) : surveiller le book-walking sur carnets minces.

---

## 5. CLASSIFICATION DES COMPOSANTS

| Composant | Classe | Preuve |
|---|---|---|
| Pipeline de risque, machine à états, SOR, position lifecycle, allocateur, macro, sentiment, contrepartie, réconciliation, watchdog, coûts, attribution, stress, bias-audit | **PRODUCTION** | tests + safety check |
| Conviction engine, edge decay, drift PSI, hierarchical allocator, adversarial, execution intel, journal, gouvernance | **PRODUCTION** (mécanisme) — **mesures en attente** | 87 % couverture, 0 clôture |
| GAN, RLHF, options vol, LLM narrative | **EDUCATIONAL** (verrouillés) | tests AST + registre |
| Stratégies (Grid, MeanRev, Trend, Momentum, XSM, MTF) | **DEGRADED** (historique) / **à re-mesurer** | P&L historique −0,22 % |

---

## 6. INCIDENTS & ARTEFACTS (traités)

| Événement | Nature | Traitement |
|---|---|---|
| 112 faux TRADE dans le journal (19 h sans fill) | Bug réel (journal avant gates) | Corrigé + tests |
| 24 TRADE à niveau NO_TRADE | Incohérence signal/conviction | Corrigé + tests |
| 170 événements de test dans la DB | Pollution de données (42 % du P&L) | Purgés + backup |
| 180 RECONCILIATION_FAILED (20/08) | Artefacts de tests (0 depuis) | Documenté, non reproductible |
| `live_reconciler.py` | Code mort | Documenté |

---

*Document vivant — mis à jour à chaque cycle MONITOR → DETECT → ANALYZE → VALIDATE → IMPROVE.*

# 🌍 VISION « NIVEAU MONDIAL » — CE QUI MANQUE ENCORE VRAIMENT
**Analyse experte indépendante — 18 août 2026**

> Ce document répond honnêtement à la question : *« que manque-t-il pour dépasser les
> niveaux institutionnels, voire mondiaux ? »* Il distingue ce qui est **faisable** dans
> ce projet, ce qui est **structurellement nécessaire** pour penser/trader comme un
> fonds de premier plan, et ce qui est **hors de portée** d'un bot mono-utilisateur
> (et pourquoi ce n'est pas grave).

---

## 1. LA VÉRITÉ D'ABORD

Ce qui fait la supériorité d'un Renaissance / Two Sigma / Jane Street n'est **pas** un
modèle magique. C'est un **processus de recherche systématique** + une **exécution
microscopique** + une **discipline de validation statistique** + une **infrastructure
d'expérimentation**. Le bot actuel a les briques *d'exécution et de surveillance* d'un
terminal sérieux. Ce qui lui manque vraiment, c'est le **cerveau de recherche** et la
**profondeur d'exécution**.

---

## 2. SA FAÇON DE PENSER (le processus de recherche)

| Ce que font les fonds mondiaux | État actuel | Ce qu'il faudrait |
|---|---|---|
| **Bibliothèque de signaux** : des centaines d'hypothèses testées, seules les meilleures survivent (correction multi-tests — *Deflated Sharpe Ratio* déjà dans le code mais jamais utilisé comme **porte** d'admission) | 7 stratégies TA classiques, fixes | Créer un **catalogue de signaux** (30-50) : chaque signal = fonction pure (features → score), évaluée en batch sur 2 ans, **admis uniquement si Sharpe défalté > seuil** avec probabilité de faux positif contrôlée |
| **Expérimentation tracée** : chaque idée = entrée dans un registre (hypothèse, données, résultat, verdict) | Aucun registre d'expériences | Table `experiments` + endpoint `/api/v1/experiments` ; le pipeline autonome y écrit chaque cycle |
| **Backtest honnête** : walk-forward + embargo + frais + slippage réaliste + *capacity* | Walk-forward multi-actifs ✅, frais fixes 0,1 % | Frais **par venue** (maker/taker/VIP), slippage **dépendant du carnet**, test de capacité (combien de $ avant que le signal s'effondre) |
| **Étude d'alpha par classe** : momentum, carry, value, volatilité, on-chain, cross-asset | Momentum/vol présent mais **non branché** (fichiers `strategies/momentum.py`, `volatility_breakout.py`, `multi_timeframe.py`, `regime_switching.py` existent mais 0 référence dans la boucle — vérifié) | **Brancher ces 4 stratégies** + ajouter **carry** (funding) et **cross-sectional momentum** (classer BTC/ETH/SOL, long top / short bottom) |

## 3. SA FAÇON DE TRADER (l'exécution)

| Niveau mondial | Actuel | À faire |
|---|---|---|
| **Exécution algorithmique** : TWAP/VWAP/POV, participation rate, arrival price | Almgren-Chriss + slicing (modules présents) mais la boucle envoie des **market orders directs** | Routeur d'exécution : choisir market / limit / TWAP selon liquidité + urgence du signal ; **mesurer l'alpha d'exécution** (prix réalisé vs arrivée) |
| **Slippage réel** : modélisé à partir du carnet, par venue, par taille | Fixe 0,03 % | Book-walking (déjà dans les adapters) branché dans le simulateur + recalibration quotidienne |
| **SOR avec carnets live** | SOR sur tickers publics | SOR sur **vrais carnets** (WebSocket depth par venue) + latence mesurée par venue |
| **Impact et queue position** (Jane Street) | — | Hors de portée sans infra colo/FIX — **à documenter comme limite assumée**, pas à imiter |

## 4. LA STRUCTURE DE L'IA (l'architecture cognitive)

L'IA actuelle est **un empilement linéaire** : signaux → méta-pondération → taille.
Un fonds mondial pense en **couches** :

```
Couche 0  Données brutes (tick, carnets, on-chain, macro, sentiment)
Couche 1  Signaux rapides (ms-min)  : microstructure, order flow, VPIN
Couche 2  Signaux lents (h-j)       : momentum, carry, on-chain, macro
Couche 3  Méta-modèle (régime-conditionné) : quel mélange de couches 1/2 selon le régime
Couche 4  Optimiseur de portefeuille : vol targeting, corrélation, CVaR, contraintes
Couche 5  Exécution : découpe, routage, impact
Couche 6  Apprentissage : récompense = PnL net d'impact, mise à jour par couche
```

**Ce qui manque concrètement :**
1. **Volatility targeting** (le standard institutionnel) : l'exposition cible est ajustée
   pour que la **vol du portefeuille** reste constante (ex. 15 % annualisé). Le bot a CVaR
   mais pas d'overlay de vol. C'est LE correctif le plus rentable et simple à ajouter.
2. **Séparation signal rapide / lent** : aujourd'hui tout est mélangé dans un consensus
   unique à 2,5 s. Distinguer horizon (scalping vs position) change la qualité.
3. **RL avec coût d'impact** : la récompense PPO est le rendement brut — ajouter une
   pénalité de slippage/impact (sinon le RL apprend à « tout acheter d'un coup »).
4. **Gates statistiques dans l'autopilote** : promouvoir un modèle **seulement si**
   Sharpe out-of-sample > X avec p-value (déflated Sharpe) — les outils existent
   (`lopez_de_prado.py`), il suffit de les brancher comme porte d'admission dans le
   cycle autonome.
5. **Explainability réelle** : le module `ai/explainable_ai.py` est un stub — produire
   au minimum un **top-5 des features** par décision (le reasoning log en profite).

## 5. STRATÉGIES MANQUANTES (par ordre de valeur/effort)

| Stratégie | Pourquoi | Effort |
|---|---|---|
| **Vol targeting (overlay)** | Réduit le drawdown structurellement, standard institutionnel, ~50 lignes | Faible — **priorité 1** |
| **Carry/funding (exécuté)** | L'arbitrage funding est signal-only ; le transformer en vraie stratégie delta-neutre (spot + perp) | Moyen — nécessite l'exécution bi-venue |
| **Cross-sectional momentum** | Classer les actifs entre eux (BTC vs ETH vs SOL vs or) : long les forts, short les faibles | Moyen |
| **Brancher momentum/multi-TF/vol breakout existants** | 4 fichiers de stratégies **morts** — les activer dans le méta-moteur | Faible |
| **Pairs/cointégration** (vraie StatArb) | Le StatArb actuel est un signal simple ; une vraie paire cointégrée (z-score) est plus robuste | Moyen |
| **On-chain comme alpha** (pas seulement risque) | Les flux d'exchanges (déjà suivis) sont un signal : accumulation/distribution | Faible |
| **Gamma/vol (options)** | Nécessite un broker d'options — **hors périmètre**, le marquer « simulateur » (fait) | — |

## 6. RISK & PORTFOLIO (le niveau « fonds »)

- ✅ Déjà : CVaR, corrélation, circuit breaker, SL/TP, plafond par actif, stress Monte-Carlo + GAN.
- ❌ Manque : **modèle de facteurs** (market/carry/momentum/vol) pour attribuer le risque et
  le P&L ; **limites de risque par facteur** ; **liquidité du portefeuille** (temps pour
  sortir X % sans impact) ; **budget de risque par stratégie** (allouer la vol, pas le capital).
- ❌ Manque : **coût de transaction dans le sizing** (Kelly net de frais — Kelly brut surestime).

## 7. INFRASTRUCTURE (ce qui fera la différence à long terme)

| Élément | Pourquoi | Effort |
|---|---|---|
| **Journal d'événements rejouable** (tous les ticks → parquet/DB) | Permet de rejouer n'importe quel jour avec le code actuel (debug, régression, A/B) — c'est LE fondement d'un fonds | Moyen-élevé |
| **Feature store avec backfill + lineage** | Reproductibilité totale des entraînements (le FeatureStore existe, l'étendre au backfill) | Moyen |
| **Séparation boucle/état** (event-driven au lieu du while 2,5 s) | Scalabilité, testabilité, multi-instance | Élevé — chantier d'architecture |
| **Métriques d'exécution** (slippage réalisé vs simulé, fill rate, latence) | Savoir si le modèle live = le modèle backtest | Faible |
| **A/B testing en paper** (deux budgets, deux configs) | Promouvoir une config seulement si elle bat l'autre en réel | Moyen |

## 8. MA PRIORISATION (si c'était mon projet, dans cet ordre)

1. **Brancher les 4 stratégies mortes + vol targeting** (2 jours) — impact immédiat, faible risque.
2. **Porte statistique dans l'autopilote** : deflated Sharpe comme condition de déploiement
   (1 jour) — transforme l'autonomie en *discipline*.
3. **Alpha d'exécution mesuré** : prix réalisé vs prix d'arrivée + slippage par venue
   (1-2 jours) — c'est ce qui dit si le bot « exécute bien ».
4. **Cross-sectional momentum + carry exécuté** (3-5 jours) — vraies sources d'alpha.
5. **Journal d'événements rejouable** (semaine) — le socle de tout le reste.
6. **Modèle de facteurs + budget de risque par stratégie** (semaine) — le niveau « fonds ».

## 9. CE QUI NE SERA JAMAIS « MONDIAL » ICI (et pourquoi c'est OK)

- **Latence microsecondes / colocation / FIX** : inutile pour un bot qui traite des barres
  de 1 min. Le HFT n'est pas le but — le **HFT n'est pas la rentabilité**, c'est une
  industrie à part.
- **Volume d'exécution** : sans capital important, l'impact n'est pas le problème central.
- **Données propriétaires** (tick vendors à 5 chiffres/mois) : les données publiques
  suffisent pour un edge de signal, pas pour un edge d'infrastructure.

**Conclusion :** le bot est déjà **fiable, protégé, testé et autonome** — un cran au-dessus
de 99 % des « bots de trading » du marché. Pour dépasser le niveau institutionnel, il ne
manque pas de fonctionnalités : il manque un **processus de recherche** (gates statistiques,
registre d'expériences), une **profondeur d'exécution** (slippage réel, alpha d'exécution)
et un **overlay de risque moderne** (vol targeting, budget de risque par stratégie).
C'est exactement la phase 3-4-5 de la roadmap — faisable, mesurable, et c'est là que
naît l'edge réel.

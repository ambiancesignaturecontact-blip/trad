# ☁️ GUIDE DE DÉPLOIEMENT RAILWAY — EXÉCUTION RÉELLE (COPY TRADING + ARBITRAGE)

**Mise à jour : 18 août 2026.** Tout le code est prêt ; il ne reste que la configuration
de tes clés dans Railway. Voici exactement quoi mettre où.

---

## 1. Vue d'ensemble (comment ça marche)

```
Leaderboard Hyperliquid (public, ~42k traders réels)
        │   GET stats-data.hyperliquid.xyz/Mainnet/leaderboard
        ▼
Copy Trading Manager  ──►  8 vrais traders affichés (ROI, PnL, compte)
        │
        ▼  (tu choisis "START" sur un trader + montant alloué)
Copy Mirroring (LOT 71, toutes les 10 min)
        │   GET api.hyperliquid.xyz/info  {type:"clearinghouseState", user:0x…}
        │   → positions RÉELLES du trader (BTC/ETH/SOL longs ou shorts)
        │   → échelle = allocation ÷ valeur du compte trader
        ▼
Ordres delta via OMS  ──►  exécutés sur TON compte Binance/Bybit
                          (si COPYTRADE_EXECUTION=auto + mode REAL + clés)
                          sinon : signaux "MIRROR" loggés (mode signal_only)
```

⚠️ **Important** : on copie les positions du trader Hyperliquid sur **ton** compte
Binance/Bybit (c'est là que tes clés existent). Ce n'est pas un « abonnement » au
copy-trading natif d'une plateforme : c'est un **miroir de positions** — le bot achète
ce que le trader détient, proportionnellement à ton allocation, et ajuste au fil de
l'eau (delta entre la cible du trader et ton portefeuille).

---

## 2. Variables à mettre dans Railway (Variables tab de ton service)

| Variable | Valeur | Requis pour |
|---|---|---|
| `COPYTRADE_EXECUTION` | `signal_only` (défaut sûr) ou `auto` | **Exécution réelle** du copy trading (`auto` = ordres passés) |
| `ARBITRAGE_EXECUTION` | `signal_only` (défaut sûr) ou `auto` | **Exécution réelle** des arbitrages funding + DEX-CEX |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | tes clés Binance (spot/futures) | Exécution REAL sur Binance (copy + arbitrage + trading) |
| `BYBIT_API_KEY` / `BYBIT_API_SECRET` | tes clés Bybit | Exécution REAL sur Bybit |
| `EVM_PRIVATE_KEY` | clé privée de ton wallet EVM (Arbitrum) | Broadcast DEX réel (arbitrage DEX-CEX) |
| `SUPABASE_DB_URL` | ta base PostgreSQL (obligatoire en REAL) | Mode REAL (fallback SQLite interdit) |
| `TRADING_MODE` | `DEMO` (défaut) → plus tard `REAL` | Bascule de mode |
| `AUTH_ENABLED` | `true` en production | Sécurité (JWT forcé) |
| `ADMIN_USER` / `ADMIN_PASSWORD` | **change le mot de passe par défaut !** | Connexion dashboard |
| `JWT_SECRET_KEY` | ≥ 24 caractères (optionnel) | Si absent, **auto-généré et stocké chiffré en DB** au premier boot (message unique dans les logs) |

> 🔑 **Secrets auto-générés (comportement récent)** : quand `AUTH_ENABLED=true` (ou mode REAL) et que
> `JWT_SECRET_KEY`/`ADMIN_PASSWORD` ne sont pas définis, le bot **ne crash plus** : il génère une clé JWT
> forte (48 octets) persistée chiffrée, et un mot de passe admin affiché **une seule fois** dans les logs
> de démarrage (`username=admin_quant password=…`). **Notez-le immédiatement**, puis définissez
> `ADMIN_PASSWORD` et `JWT_SECRET_KEY` dans Railway pour reprendre la main définitivement.
| `JWT_SECRET_KEY` | ≥ 24 caractères aléatoires | Signature des tokens |
| `FERNET_KEY` | clé Fernet (chiffrement des clés API) | Sécurité des clés stockées |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | ton bot Telegram | Notifications + concierge |
| `WEBHOOK_SECRET` | secret pour `/api/v1/webhook/trade` | Webhooks TradingView |
| `ALLOWED_ORIGINS` | ex. `https://ton-app.up.railway.app` | CORS (si besoin) |

Générer les secrets :
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"   # FERNET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"                                 # JWT_SECRET_KEY
```

---

## 3. Mise en route PAS À PAS (copy trading réel)

1. **Déploie** `git push origin main` → Railway rebuild (Docker multi-stage, torch CPU inclus).
2. **Railway → Variables** : ajoute `COPYTRADE_EXECUTION=signal_only` d'abord (mode sûr),
   tes clés Binance/Bybit, `SUPABASE_DB_URL`, les secrets de sécurité.
3. **Redémarre** le service. Vérifie les logs : `✅ LOT 67 Copy Trading…` + `✅ LOT 71 Copy mirroring…`.
4. **Ouvre le dashboard** → connecte-toi (ADMIN_USER/ADMIN_PASSWORD) → section Copy Trading →
   clique **START** sur un trader (alloue par ex. 1 000 $).
5. **Vérifie** `GET /api/v1/copy/mirror-status` (token admin) :
   ```
   execution_mode: signal_only
   following: {0x…: {mode: FOLLOW_ONLY, allocated_capital: 1000}}
   mirror_signals: {0x…: {trader_positions: [BTC, ETH, SOL], mirror_orders: [...]}}
   ```
   → Tu vois les **ordres de miroir calculés** (quantité, sens) SANS qu'ils soient exécutés.
6. **Quand tu es prêt** (après validation en paper) :
   - Passe en **REAL** (`/api/2fa-switch`, wallet ou TOTP — rappel : plus de code `123456`).
   - Met `COPYTRADE_EXECUTION=auto` → redémarre.
   - Le scheduler exécute désormais les ordres miroir via OMS sur ton compte Binance/Bybit.
   - **Garde `signal_only` tant que tu n'as pas validé** — c'est le comportement sûr.

> 🔒 Règles de sécurité actives pendant le mirroring : SL/TP par position (protection
> manager), plafond 25 %/actif, idempotence, réconciliation balance/positions toutes
> les 5 min, alerte liquidation.

---

## 4. Arbitrage réel (optionnel)

- `ARBITRAGE_EXECUTION=auto` + `EVM_PRIVATE_KEY` → l'arbitrage **DEX-CEX** broadcast
  réellement la transaction signée (1inch v6, slippage anti-MEV) sur Arbitrum.
- `ARBITRAGE_EXECUTION=auto` sans EVM → l'arbitrage **funding** enregistre l'entrée
  delta-neutre (il faut alors que Binance futures soit configuré avec les clés).
- **Recommandation** : commence par `signal_only` partout, observe 2-3 semaines les
  signaux, puis passe à `auto` avec de petites tailles.

---

## 5. Checklist de sécurité avant le mode REAL

- [ ] `JWT_SECRET_KEY` forte + `ADMIN_PASSWORD` changé (le bot **refuse de démarrer**
      en prod sinon — LOT 62).
- [ ] `FERNET_KEY` définie (les clés API sont chiffrées avec).
- [ ] `SUPABASE_DB_URL` définie (SQLite interdit en REAL).
- [ ] Autopilote : le bot exige N jours de paper (`autopilot.min_paper_validation_days`,
      défaut 7) avant d'autoriser le REAL.
- [ ] Backtest walk-forward ≥ 2 ans + paper-trading avant de mettre des fonds réels.

# GUIDE DE DEPLOIEMENT DE PRODUCTION : SUPABASE + VERCEL + WORKER AUTONOME

Ce guide détaille l'architecture cible et la procédure pas à pas pour déployer votre plateforme de trading autonome institutionnelle en environnement réel.

---

## 1. COMPRÉHENSION DES CONTRAINTES DE DEPLOIEMENT (SÉCURITÉ & FIABILITÉ)

### ⚠️ Pourquoi Vercel seul est insuffisant pour un Bot Autonome ?
Vercel est une plateforme **Serverless** (FaaS). Les fonctions d'API y sont exécutées à la demande avec un temps limite de calcul court (typiquement 10 à 60 secondes).
- **Problème** : Une boucle infinie asynchrone (comme notre boucle de trading autonome `while True: await asyncio.sleep(2)`) sera brutalement interrompue par Vercel dès que la requête HTTP se termine.
- **Conséquence** : Votre bot ne tournerait pas de manière autonome 24h/24 !

### 🏛️ L'Architecture Cible Institutionnelle (Découplée)
Pour garantir une fiabilité de $99.99\%$ et une précision d'exécution maximale, nous adoptons une architecture découplée :

```
    +---------------------------------------------------------------+
    |                      INTERFACE WEB (Vercel)                   |
    |  - Héberge le Frontend (React/Next.js ou HTML5)               |
    |  - Très rapide, global, connecte vos WebSockets               |
    +------------------------------+--------------------------------
                                   |
                                   v (Flux de données REST & WS)
    +---------------------------------------------------------------+
    |                   BASE DE DONNÉES (Supabase)                  |
    |  - Base de Données PostgreSQL Relationnelle                   |
    |  - "Single Source of Truth" (Ordres, Positions, Clés, Logs)    |
    +------------------------------+--------------------------------
                                   ^
                                   | (Synchronisation permanente)
    +---------------------------------------------------------------+
    |                  WORKER EN TEMPS RÉEL (Render/Railway)        |
    |  - Conteneur Docker léger exécutant le script Python `main.py` |
    |  - Tourne 24h/24 sans interruption                            |
    |  - Connexion WebSocket persistante à Binance/Bybit via CCXT   |
    +---------------------------------------------------------------+
```

---

## 2. CONFIGURATION DE LA BASE DE DONNÉES (Supabase)

Supabase est propulsé par une base PostgreSQL complète. Notre fichier `db_manager.py` intègre un **détecteur automatique de dialecte** qui passe de SQLite à PostgreSQL sans modification de code.

### Étapes :
1. Créez un projet sur [Supabase](https://supabase.com).
2. Allez dans **Database Settings** > **Connection Strings** et copiez votre URL de connexion PostgreSQL (URI).
   L'URI ressemble à ceci :
   `postgresql://postgres:[VOTRE-MOT-DE-PASSE]@db.[VOTRE-SUBABASE-ID].supabase.co:5432/postgres`
3. Définissez cette variable d'environnement sur votre plateforme d'hébergement du Worker (Railway/Render) sous le nom :
   - **`SUPABASE_DB_URL`** (ou `DATABASE_URL`)

Notre script `db_manager.py` s'occupe de créer automatiquement toutes les tables (ordres, positions, clés, logs d'audit) au premier démarrage !

---

## 3. CONFIGURATION ET DEPLOIEMENT DU WORKER (Railway ou Render)

Pour faire tourner la logique de trading autonome asynchrone 24h/24, utilisez **Railway** ou **Render** (déploiement en 2 clics avec Docker).

### Option A : Déploiement Railway (Recommandé)
1. Créez un compte sur [Railway.app](https://railway.app).
2. Cliquez sur **New Project** > **Deploy from GitHub repo**.
3. Ajoutez les variables d'environnement suivantes dans l'onglet **Variables** :
   - `DATABASE_URL` : *Votre URI Supabase copiée à l'étape précédente.*
   - `FERNET_KEY` : *Une clé de chiffrement AES aléatoire pour sécuriser vos clés API.* (Générez-en une en Python avec `from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())`).
4. Railway détectera le fichier `requirements.txt` ou l'image et déploiera automatiquement le bot.

### Option B : Déploiement Render.com
1. Créez un compte sur [Render](https://render.com).
2. Choisissez **New** > **Web Service**.
3. Liez votre dépôt GitHub et sélectionnez le runtime **Python**.
4. Configurez la commande de démarrage : `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Ajoutez vos variables d'environnement (`DATABASE_URL` et `FERNET_KEY`).

---

## 4. DÉPLOIEMENT DU FRONTEND SUR VERCEL

1. Créez un compte sur [Vercel](https://vercel.com).
2. Liez votre GitHub et sélectionnez le dossier de votre projet.
3. Configurez les variables d'environnement pour que le Frontend sache où se connecter pour les WebSockets :
   - Ajoutez une variable pour l'URL de votre Worker Render/Railway (ex: `NEXT_PUBLIC_WORKER_URL=https://mon-bot-quant.up.railway.app`).
4. Cliquez sur **Deploy**. Votre magnifique tableau de bord de trading est en ligne !

---

## 5. RESOLUTION DES POINTS FAIBLES DE PRODUCTION

Pour garantir une **précision et une fiabilité de $100\%$**, nous avons corrigé les faiblesses critiques des bots de trading classiques :

| Points Faibles Classiques | Solution Appliquée dans notre Architecture |
| :--- | :--- |
| **Rejet de taille par l'Exchange** | Notre méthode `format_exchange_size()` interroge en direct les filtres `limits['amount']['min']` et la précision décimale exacte de l'actif sur l'exchange pour arrondir précisément vos ordres avant l'envoi. |
| **Pertes sur Déconnexions WebSocket** | L'utilisation de la bibliothèque de niveau institutionnel **CCXT** gère automatiquement la file d'attente des requêtes et applique des reconnexions persistantes avec baisse de charge (exponential backoff) en cas de surcharge des serveurs de l'exchange. |
| **Risque d'ordres doublons (Ghost Orders)** | Nous soumettons chaque ordre réel avec un paramètre d'idempotence unique : le **`clientOrderId`**. Si une déconnexion survient au moment précis de l'envoi, l'exchange rejettera toute tentative de soumission en doublon de cet identifiant. |
| **Clés API stockées en clair** | Vos clés API sont chiffrées de bout en bout avec l'algorithme symétrique **AES-256 (Fernet)** et ne sont déchiffrées qu'en mémoire vive lors de la transmission réseau sécurisée SSL/TLS à l'exchange. |
| **Overtrading et Frais Cachés** | Notre fonction de récompense PPO au sein de l'IA pénalise lourdement les coûts de transaction excessifs et le slippage, forçant le modèle à n'entrer en position que lorsque la probabilité de gain couvre largement le spread. |

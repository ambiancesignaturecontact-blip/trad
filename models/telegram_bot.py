import os
import logging
import asyncio
import json
import httpx

logger = logging.getLogger("TelegramBot")

class TelegramBotManager:
    """
    Asynchronous Telegram Bot Manager for Mobile Push Notifications & Interactive Remote Control.
    Uses extremely friendly, beginner-accessible language (explaining quant concepts simply)
    and rich, beautiful emojis!
    """
    def __init__(self, state_dict=None, db_manager=None):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.state = state_dict if state_dict is not None else {}
        self.db = db_manager
        
        self.api_url = f"https://api.telegram.org/bot{self.token}" if self.token else None
        self.last_update_id = 0
        
        if self.token and self.chat_id:
            logger.info("Telegram Bot successfully configured and active.")
        else:
            logger.warning("Telegram token or chat_id is missing. Operating in alert-silent mode.")

    async def send_startup_message(self):
        """
        Sends the startup hello notification. 
        """
        if self.token and self.chat_id:
            await self.send_push_notification(
                "🚀 *VOTRE COPILOTE DE TRADING EST EN LIGNE !*\n\n"
                "Bonjour ! Je suis *Q-Bot*, votre assistant de trading autonome d'élite.\n"
                "Je viens de m'allumer avec succès sur Railway. Je surveille désormais de vrais marchés "
                "de crypto-monnaies, d'or et d'actions pour vous 24h/24 ! 🌤️"
            )

    async def send_push_notification(self, text: str) -> bool:
        """
        Sends an instantaneous markdown-formatted push alert to the user's mobile.
        """
        if not self.api_url or not self.chat_id:
            return False
            
        url = f"{self.api_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=5.0)
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram push alert: {str(e)}")
            return False

    async def send_interactive_macro_alert(self, event_name: str, time_to_event: float) -> bool:
        """
        Sends an instantaneous markdown alert with clickable inline keyboard buttons.
        """
        if not self.api_url or not self.chat_id:
            return False
            
        url = f"{self.api_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": (
                f"⏰ *ATTENTION : TEMPÊTE MACROÉCONOMIQUE APPROCHE !*\n"
                f"-----------------------------------------\n"
                f"L'événement très important `{event_name}` est prévu dans *{time_to_event:.1f} minutes* !\n\n"
                f"💡 *Explication simple* : Ces annonces provoquent souvent de fortes vagues sur les prix.\n"
                f"Par sécurité, je vous recommande vivement de réduire temporairement nos investissements de 60%."
            ),
            "parse_mode": "Markdown",
            "reply_markup": {
                "inline_keyboard": [
                    [
                        {"text": "🛑 SÉCURISER ET RÉDUIRE (60%)", "callback_data": f"macro_reduce_expo_{event_name}"},
                        {"text": "🟢 CONSERVER L'EXPOSITION", "callback_data": "macro_ignore_alert"}
                    ]
                ]
            }
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=5.0)
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send interactive Telegram macro alert: {str(e)}")
            return False

    async def poll_telegram_commands_loop(self):
        if not self.api_url:
            return
            
        logger.info("Starting Telegram Remote Control polling worker...")
        url = f"{self.api_url}/getUpdates"
        
        while True:
            try:
                params = {"offset": self.last_update_id + 1, "timeout": 20}
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, params=params, timeout=25.0)
                    
                    if resp.status_code == 200:
                        updates = resp.json().get("result", [])
                        for update in updates:
                            self.last_update_id = update.get("update_id", self.last_update_id)
                            
                            # Handle standard message commands
                            if "message" in update:
                                message = update.get("message", {})
                                chat_id = str(message.get("chat", {}).get("id", ""))
                                if chat_id != self.chat_id:
                                    continue
                                text = message.get("text", "").strip()
                                await self.process_command(text)
                                
                            # Handle tactile inline callback button clicks!
                            elif "callback_query" in update:
                                callback = update.get("callback_query", {})
                                callback_id = callback.get("id")
                                chat_id = str(callback.get("message", {}).get("chat", {}).get("id", ""))
                                if chat_id != self.chat_id:
                                    continue
                                    
                                data = callback.get("data", "")
                                await self.process_callback_query(callback_id, data)
            except Exception as e:
                logger.error(f"Error in Telegram polling loop: {str(e)}")
                
            await asyncio.sleep(3)

    async def process_command(self, command: str):
        cmd_lower = command.lower()
        
        if cmd_lower == "/start":
            welcome_msg = (
                "👋 *BIENVENUE SUR VOTRE CONSOLE DE TRADING TACTILE !*\n\n"
                "Je suis *Q-Bot*, votre copilote intelligent. Je gère vos investissements et protège votre capital de manière entièrement autonome.\n\n"
                "📌 *Commandes tactiles simples :*\n"
                "📊 `/status` - Voir ma santé, mon solde, et mes investissements en cours.\n"
                "⏸️ `/pause` - Mettre en pause le trading automatique (sécurité).\n"
                "🟢 `/resume` - Relancer le trading automatique.\n"
                "🚨 `/kill` - URGENCE : Vendre tous mes investissements et sécuriser tout mon argent."
            )
            await self.send_push_notification(welcome_msg)
            
        elif cmd_lower == "/status":
            mode = self.state.get("mode", "DEMO")
            equity = self.state.get("current_equity", 0.0)
            balance = self.state.get("balance_demo" if mode == "DEMO" else "balance_real", 0.0)
            regime = self.state.get("regime_name", "Unknown")
            is_active = "ACTIF 🟢" if self.state.get("is_running") else "EN PAUSE ⏸️"
            
            # Simple, beginner-friendly translations of regime
            hmm_translation = {
                "Bull Trend (Low Vol)": "Soleil Haussier ☀️ (Marché haussier calme)",
                "Bear Trend (High Vol)": "Tempête Baissière ⛈️ (Marché en baisse rapide)",
                "Mean-Reverting Range": "Temps Nuageux ⛅ (Marché stable qui oscille)",
                "Erratic High Volatility": "Volatilité Erratique 🌪️ (Marché agité et imprévisible)"
            }
            translated_regime = hmm_translation.get(regime, regime)
            
            pos_msg = ""
            if self.db:
                positions = self.db.get_positions()
                if positions:
                    pos_msg = "\n📂 *Vos investissements actifs :*"
                    for p in positions:
                        pos_msg += f"\n- *{p['symbol']}* : {p['qty']:.4f} (Prix d'achat: ${p['avg_price']:.2f})"
                else:
                    pos_msg = "\n📂 *Vos investissements actifs :* Aucun achat en cours. Votre argent dort en sécurité !"
            
            status_msg = (
                f"📊 *RAPPORT DE SANTÉ DE VOTRE TIRELIRE ({mode})*\n"
                f"-----------------------------------------\n"
                f"⚙️ État du Bot : *{is_active}*\n"
                f"🌦️ Météo du marché : *{translated_regime}*\n"
                f"💰 Votre argent disponible : *${balance:,.2f}*\n"
                f"📈 Valeur totale de votre tirelire : *${equity:,.2f}*\n"
                f"{pos_msg}"
            )
            await self.send_push_notification(status_msg)
            
        elif cmd_lower == "/pause":
            self.state["is_running"] = False
            await self.send_push_notification("⏸️ *TRADING MIS EN PAUSE !* J'ai arrêté toute prise de position automatique. Vos fonds actuels sont conservés au chaud.")
            
        elif cmd_lower == "/resume":
            self.state["is_running"] = True
            await self.send_push_notification("🟢 *TRADING AUTOMATIQUE RELANCÉ !* Je reprends la surveillance active des marchés réels pour investir selon les meilleures opportunités.")
            
        elif cmd_lower == "/kill":
            self.state["kill_switch_active"] = True
            self.state["is_running"] = False
            try:
                async with httpx.AsyncClient() as client:
                    await client.post("http://127.0.0.1:8000/api/kill-switch")
                await self.send_push_notification(
                    "🚨 *URGENCE : KILL SWITCH DÉCLENCHÉ !*\n"
                    "J'ai immédiatement VENDU tous nos investissements au prix du marché et gelé le robot. Tout votre capital est désormais à l'abri."
                )
            except Exception as e:
                await self.send_push_notification(f"⚠️ Échec du bouton d'urgence : {str(e)}")

    async def process_callback_query(self, callback_id: str, data: str):
        """
        Executes actions directly triggered by the user's tactile clicks!
        """
        if data.startswith("macro_reduce_expo_"):
            event_name = data.replace("macro_reduce_expo_", "")
            self.state["macro_scale_factor_tactile"] = 0.40
            
            await self.send_push_notification(
                f"🛑 *EXPOSITION RÉDUITE À 40% (RÉDUCTION DE 60%) !*\n"
                f"Félicitations pour votre prudence. J'ai immédiatement bridé nos tailles d'achats "
                f"pour amortir l'impact de l'annonce `{event_name}`. La sécurité avant tout !"
            )
        elif data == "macro_ignore_alert":
            self.state["macro_scale_factor_tactile"] = 1.0
            await self.send_push_notification(
                "🟢 *ALERTE IGNORÉE :* Vous avez choisi de maintenir notre stratégie standard. Je continue mes investissements habituels."
            )
 McNeil = True

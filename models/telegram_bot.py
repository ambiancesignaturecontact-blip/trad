import os
import logging
import asyncio
import httpx

logger = logging.getLogger("TelegramBot")

class TelegramBotManager:
    """
    Asynchronous Telegram Bot Manager for Mobile Push Notifications & Interactive Remote Control.
    Enables instant tactile commands (/start, /status, /pause, /resume, /kill) via Telegram API.
    """
    def __init__(self, state_dict=None, db_manager=None):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.state = state_dict or {}
        self.db = db_manager
        
        self.api_url = f"https://api.telegram.org/bot{self.token}" if self.token else None
        self.last_update_id = 0
        
        if self.token and self.chat_id:
            logger.info("Telegram Bot successfully configured and active.")
            # Send startup hello push notification!
            asyncio.create_task(self.send_push_notification(
                "🚀 *QUANT-PORTAL EXÉCUTEUR EN LIGNE*\n"
                "Le robot de trading IA vient de démarrer avec succès sur Railway !"
            ))
        else:
            logger.warning("Telegram token or chat_id is missing. Operating in alert-silent mode.")

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

    async def poll_telegram_commands_loop(self):
        """
        Continuous long-polling task to read interactive tactile commands from the user's phone.
        """
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
                            message = update.get("message", {})
                            chat_id = str(message.get("chat", {}).get("id", ""))
                            
                            # Strict sender authorization: ignore messages from other users!
                            if chat_id != self.chat_id:
                                continue
                                
                            text = message.get("text", "").strip()
                            await self.process_command(text)
            except Exception as e:
                logger.error(f"Error in Telegram polling loop: {str(e)}")
                
            await asyncio.sleep(3) # Polling tick rate

    async def process_command(self, command: str):
        """
        Processes tactile slash-commands and modifies the shared engine state instantly.
        """
        cmd_lower = command.lower()
        
        if cmd_lower == "/start":
            welcome_msg = (
                "👋 *BIENVENUE SUR VOTRE CONSOLE QUANT-PORTAL MOBILE !*\n"
                "Pilotez votre IA de trading directement depuis cette discussion.\n\n"
                "📌 *Commandes tactiles valides :*\n"
                "🔹 `/status` - État en temps réel (PnL, Balance, Positions)\n"
                "🔹 `/pause` - Mettre en pause le trading autonome\n"
                "🔹 `/resume` - Relancer le trading autonome\n"
                "🔹 `/kill` - URGENCE : Fermer toutes les positions et verrouiller le bot"
            )
            await self.send_push_notification(welcome_msg)
            
        elif cmd_lower == "/status":
            mode = self.state.get("mode", "DEMO")
            equity = self.state.get("current_equity", 0.0)
            balance = self.state.get("balance_demo" if mode == "DEMO" else "balance_real", 0.0)
            regime = self.state.get("regime_name", "Unknown")
            is_active = "ACTIF 🟢" if self.state.get("is_running") else "EN PAUSE ⏸️"
            
            # Fetch active positions
            pos_msg = ""
            if self.db:
                positions = self.db.get_positions()
                if positions:
                    pos_msg = "\n📂 *Expositions actives :*"
                    for p in positions:
                        pos_msg += f"\n- {p['symbol']} : {p['qty']:.4f} (Moy: ${p['avg_price']:.2f})"
                else:
                    pos_msg = "\n📂 *Expositions actives :* Aucune position ouverte."
            
            status_msg = (
                f"📊 *ÉTAT QUANT-PORTAL ({mode})*\n"
                f"-----------------------------------------\n"
                f"⚙️ Bot status : {is_active}\n"
                f"🌦️ Régime marché : `{regime}`\n"
                f"💰 Balance dispo : `${balance:,.2f}`\n"
                f"📈 Équité totale : `${equity:,.2f}`\n"
                f"{pos_msg}"
            )
            await self.send_push_notification(status_msg)
            
        elif cmd_lower == "/pause":
            self.state["is_running"] = False
            await self.send_push_notification("⏸️ *TRADING SUSPENDU :* Le bot est désormais en pause. Tous les calculs automatiques d'ordres sont bloqués.")
            
        elif cmd_lower == "/resume":
            self.state["is_running"] = True
            await self.send_push_notification("🟢 *TRADING ACTIF :* Le bot est relancé de manière autonome sur les marchés réels/démo !")
            
        elif cmd_lower == "/kill":
            self.state["kill_switch_active"] = True
            self.state["is_running"] = False
            
            # Request flat close via HTTP POST to our own local REST API
            # This makes the kill switch completely unified!
            try:
                async with httpx.AsyncClient() as client:
                    await client.post("http://127.0.0.1:8000/api/kill-switch")
                await self.send_push_notification(
                    "🚨 *EMERGENCY SHUTDOWN ENGAGED !*\n"
                    "Le robot vient de fermer TOUTES les positions au marché et a bloqué le moteur d'exécution."
                )
            except Exception as e:
                await self.send_push_notification(f"⚠️ Échec du kill-switch : {str(e)}")

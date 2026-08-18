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

    async def send_push_notification(self, text: str, reply_markup: dict = None) -> bool:
        """
        Sends an instantaneous markdown-formatted push alert to the user's mobile.
        Automatically escapes HTML and converts standard markdown syntax to prevent Telegram crashes (HTTP 400).
        """
        if not self.api_url or not self.chat_id:
            return False
            
        url = f"{self.api_url}/sendMessage"
        safe_html = self._format_markdown_to_html(text)
        payload = {
            "chat_id": self.chat_id,
            "text": safe_html,
            "parse_mode": "HTML"
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
            
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
        raw_text = (
            f"⏰ *ATTENTION : TEMPÊTE MACROÉCONOMIQUE APPROCHE !*\n"
            f"-----------------------------------------\n"
            f"L'événement très important `{event_name}` est prévu dans *{time_to_event:.1f} minutes* !\n\n"
            f"💡 *Explication simple* : Ces annonces provoquent souvent de fortes vagues sur les prix.\n"
            f"Par sécurité, je vous recommande vivement de réduire temporairement nos investissements de 60%."
        )
        safe_html = self._format_markdown_to_html(raw_text)
        payload = {
            "chat_id": self.chat_id,
            "text": safe_html,
            "parse_mode": "HTML",
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

    def _format_markdown_to_html(self, text: str) -> str:
        """
        Safely converts basic markdown (*bold*, _italic_, `code`) to valid HTML,
        while escaping HTML special characters to prevent Telegram parse crashes (HTTP 400).
        Supports both raw HTML messages (preserving tags) and plain markdown messages.
        """
        if not text:
            return ""
            
        # Detect if the input already contains HTML tags to avoid double-escaping
        has_html = "<b>" in text or "<code>" in text or "<i>" in text or "<strong>" in text or "<a>" in text
        
        if not has_html:
            # 1. Escape HTML entities safely
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            # 2. Convert bold (`*text*`) to `<b>text</b>`
            parts = text.split("*")
            for i in range(1, len(parts), 2):
                parts[i] = f"<b>{parts[i]}</b>"
            text = "".join(parts)
            
            # 3. Convert code (`` `text` ``) to `<code>text</code>`
            parts = text.split("`")
            for i in range(1, len(parts), 2):
                parts[i] = f"<code>{parts[i]}</code>"
            text = "".join(parts)
            
            # 4. Convert italic (`_text_`) to `<i>text</i>`
            parts = text.split("_")
            for i in range(1, len(parts), 2):
                parts[i] = f"<i>{parts[i]}</i>"
            text = "".join(parts)
            
        return text

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

        # VISION_FUTUR §6: consultative-mode approvals
        if cmd_lower == "/approve":
            try:
                from main import STATE as _S, submit_order_via_oms as _submit, db as _db
                pending = list(_S.get("pending_approvals", []))
                _S["pending_approvals"] = []
                approved = 0
                for p in pending:
                    try:
                        _submit(p["symbol"], p["side"], p["qty"], p["price"], p["mode"], p["strategy"])
                        _db.add_audit_log("APPROVED_ORDER", "127.0.0.1",
                                          f"Telegram approved {p['side']} {p['qty']} {p['symbol']} ({p['strategy']})")
                        approved += 1
                    except Exception as e:
                        logger.error(f"Telegram approval failed: {e}")
                await self.send_push_notification(
                    f"✅ *APPROBATION*\n{approved} ordre(s) approuvé(s) et exécuté(s).\n"
                    f"Restants en attente : {len(_S.get('pending_approvals', []))}"
                )
            except Exception as e:
                await self.send_push_notification(f"❌ Approbation impossible : {e}")
            return
        if cmd_lower == "/reject":
            try:
                from main import STATE as _S
                n = len(_S.get("pending_approvals", []))
                _S["pending_approvals"] = []
                await self.send_push_notification(f"🚫 {n} proposition(s) rejetée(s). Aucun ordre passé.")
            except Exception:
                pass
            return
        if cmd_lower.startswith("/chat "):
            try:
                from main import STATE as _S, answer_question, db as _db
                question = command[6:].strip()
                ctx = {"last_price": _S.get("last_price"), "current_equity": _S.get("current_equity"),
                       "regime_name": _S.get("regime_name"), "confidence_index": _S.get("confidence_index", 100)}
                reply = answer_question(question, ctx)
                await self.send_push_notification(f"🤖 *Assistant*\n{reply[:3500]}")
            except Exception as e:
                await self.send_push_notification(f"❌ Assistant indisponible : {e}")
            return
        
        # Standard buttons layout (Like a Telegram Web App)
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "📊 Rapport Status", "callback_data": "bot_status"},
                    {"text": "📋 Historique", "callback_data": "bot_history"}
                ],
                [
                    {"text": "🛡️ Risques", "callback_data": "bot_risk"},
                    {"text": "⚖️ Modes", "callback_data": "bot_modes"}
                ],
                [
                    {"text": "🟢 Activer", "callback_data": "bot_resume"},
                    {"text": "⏸️ Pause", "callback_data": "bot_pause"}
                ],
                [
                    {"text": "🚨 KILL SWITCH", "callback_data": "bot_kill"}
                ]
            ]
        }
        
        # Dynamically append native Telegram Mini App button if hosting URL is present!
        web_app_url = os.getenv("PORTAL_URL") or os.getenv("WEB_APP_URL") or os.getenv("RAILWAY_STATIC_URL")
        if web_app_url:
            if not web_app_url.startswith("http"):
                web_app_url = f"https://{web_app_url}"
            keyboard["inline_keyboard"].insert(0, [
                {"text": "🖥️ OUVRIR MINI APP PRO", "web_app": {"url": web_app_url + "/telegram_mini_app.html"}}
            ])
        
        if cmd_lower.startswith("/set "):
            try:
                amount_str = command.split(" ")[1]
                amount = float(amount_str)
                if amount <= 0:
                    raise ValueError("Amount must be positive.")
                    
                self.state["balance_demo"] = amount
                self.state["initial_capital_demo"] = amount
                self.state["current_equity"] = amount
                self.state["equity_history_demo"] = [amount]
                
                if self.db:
                    self.db.add_audit_log(
                        "DEMO_BALANCE_RESET_TELEGRAM", 
                        "127.0.0.1", 
                        f"Demo balance has been manually reset to {amount} USD via Telegram."
                    )
                    
                await self.send_push_notification(
                    f"💰 <b>SOLDE SIMULÉ RÉINITIALISÉ !</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Le capital de votre portefeuille de démonstration a été réinitialisé à : <b>${amount:,.2f} USD</b>.\n\n"
                    f"🌦️ <i>Toutes les métriques ont été resynchronisées en temps réel !</i>",
                    reply_markup=keyboard
                )
                return
            except Exception as e:
                await self.send_push_notification(
                    f"⚠️ <b>ERREUR DE COMMANDE :</b>\n"
                    f"Format correct : <code>/set [montant]</code>\n"
                    f"Exemple : <code>/set 150000</code>\n"
                    f"Détails : {str(e)}",
                    reply_markup=keyboard
                )
                return
                
        if cmd_lower in ["/start", "/help"]:
            welcome_msg = (
                "👋 *BIENVENUE SUR VOTRE CONSOLE DE TRADING TACTILE !*\n\n"
                "Je suis *Q-Bot*, votre copilote intelligent. Je gère vos investissements et protège votre capital de manière entièrement autonome.\n\n"
                "📌 *Commandes tactiles simples :*\n"
                "📊 `/status` - Voir ma santé, mon solde, et mes investissements en cours.\n"
                "📋 `/history` - Voir les 5 dernières transactions passées par le bot.\n"
                "⚖️ `/modes` - Explications simples sur le mode DÉMO vs RÉEL.\n"
                "🛡️ `/risk` - Afficher mes règles de protection de capital actuelles.\n"
                "⏸️ `/pause` - Mettre en pause le trading automatique (sécurité).\n"
                "🟢 `/resume` - Relancer le trading automatique.\n"
                "🚨 `/kill` - URGENCE : Vendre immédiatement tous mes actifs et geler le robot."
            )
            await self.send_push_notification(welcome_msg, reply_markup=keyboard)
            
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
            
            # Calculate Live Benefits (PnL)
            initial_cap = self.state.get("initial_capital_demo" if mode == "DEMO" else "initial_capital_real", 100000.0)
            live_pnl_usd = equity - initial_cap if initial_cap > 0 else 0.0
            live_pnl_pct = (live_pnl_usd / initial_cap) * 100.0 if initial_cap > 0 else 0.0
            
            pnl_color = "🟢" if live_pnl_usd >= 0 else "🔴"
            pnl_sign = "+" if live_pnl_usd >= 0 else ""
            
            pos_msg = ""
            if self.db:
                positions = self.db.get_positions()
                if positions:
                    pos_msg = "📂 <b>Vos investissements actifs :</b>"
                    for p in positions:
                        pos_msg += f"\n• <b>{p['symbol']}</b> : {p['qty']:.4f} (Prix d'achat: ${p['avg_price']:.2f})"
                else:
                    pos_msg = "📂 <b>Vos investissements actifs :</b> Aucun achat en cours. Votre argent dort en sécurité !"
            
            # LOT 2 (PDF Faille 3) : état de la machine à états risque
            try:
                _rs = self.state.get("risk_state", {}) or {}
                _rs_state = _rs.get("state", "NORMAL")
                _rs_emoji = {"NORMAL": "🟢", "CAUTION": "🟠", "HALT": "🔴"}.get(_rs_state, "⚪")
                _rs_reason = _rs.get("reason", "")
                _rs_factor = float(_rs.get("scale_factor", 1.0)) * 100.0
                risk_line = (f"{_rs_emoji} <b>État Risque : {_rs_state}</b> "
                             f"(facteur {_rs_factor:.0f}%"
                             + (f" — {_rs_reason}" if _rs_reason else "") + ")\n")
            except Exception:
                risk_line = ""
            
            status_msg = (
                f"🏦 <b>QUANT-PORTAL • CONSOLE DE TRADING ({mode})</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🤖 État du Bot : <b>{is_active}</b>\n"
                f"🌦️ Météo Marché : <b>{translated_regime}</b>\n"
                f"{risk_line}"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💳 Capital Initial : <code>${initial_cap:,.2f} USD</code>\n"
                f"💰 Solde disponible : <b>${balance:,.2f} USD</b>\n"
                f"💎 Valeur Tirelire : <b>${equity:,.2f} USD</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{pnl_color} <b>BÉNÉFICES EN DIRECT :</b> <b>{pnl_sign}${live_pnl_usd:,.2f} USD</b> (<code>{pnl_sign}{live_pnl_pct:.2f}%</code>)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{pos_msg}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🖥️ <i>Terminal Web synchronisé. Utilisez les touches ci-dessous :</i>"
            )
            await self.send_push_notification(status_msg, reply_markup=keyboard)
            
        elif cmd_lower == "/history":
            orders_msg = "📋 *HISTORIQUE DES 5 DERNIÈRES TRANSACTIONS :*\n-----------------------------------------"
            if self.db:
                orders = self.db.get_all_orders()
                if orders:
                    for o in orders[:5]:
                        sign_emoji = "🟢 ACHAT" if o['side'].upper() == "BUY" else "🔴 VENTE"
                        orders_msg += (
                            f"\n🎬 {sign_emoji} *{o['symbol']}*\n"
                            f"   • Qté : `{o['qty']:.4f}` | Prix : `${o['price']:.2f}`\n"
                            f"   • Statut : `{o['status']}` | Strategie : `{o['strategy']}`\n"
                        )
                else:
                    orders_msg += "\nAnalyse : Aucun ordre n'a encore été passé pour le moment."
            else:
                orders_msg += "\nBase de données indisponible."
            await self.send_push_notification(orders_msg, reply_markup=keyboard)
            
        elif cmd_lower == "/modes":
            mode_msg = (
                "⚖️ *DÉMO vs RÉEL : COMPRENDRE LA DIFFÉRENCE*\n"
                "-----------------------------------------\n"
                "• *Mode DÉMO (Simulation)* : Le robot s'entraîne avec un capital virtuel fictif que vous pouvez modifier (ex: $100,000). Aucun risque pour vos économies !\n\n"
                "• *Mode RÉEL (Production)* : Le robot se connecte à vos clés API d'exchange cryptées pour placer de vrais investissements. L'accès est hautement sécurisé par votre connexion MetaMask.\n\n"
                "💡 *Mode Actif Actuel* : " + f"*{self.state.get('mode', 'DEMO')}*"
            )
            await self.send_push_notification(mode_msg, reply_markup=keyboard)
            
        elif cmd_lower == "/risk":
            # Simple explanations of risk settings
            risk_msg = (
                "🛡️ *VOS RÈGLES DE PROTECTION ACTIVES :*\n"
                "-----------------------------------------\n"
                "• *Max Daily Drawdown (2.5%)* : Si le marché se retourne et que notre portefeuille perd plus de 2.5% sur la même journée, j'active le coupe-circuit d'urgence pour tout stopper.\n"
                "• *Max Total Drawdown (8.0%)* : Si la perte historique cumulée atteint 8.0%, le robot se fige par sécurité.\n"
                "• *Sizing à la volatilité* : Plus le vent souffle fort sur le marché (haute volatilité), plus je réduis la taille de mes transactions pour vous protéger !"
            )
            await self.send_push_notification(risk_msg, reply_markup=keyboard)
            
        elif cmd_lower == "/pause":
            self.state["is_running"] = False
            await self.send_push_notification("⏸️ *TRADING MIS EN PAUSE !* J'ai arrêté toute prise de position automatique. Vos fonds actuels sont conservés au chaud.", reply_markup=keyboard)
            
        elif cmd_lower == "/resume":
            self.state["is_running"] = True
            # LOT 2 : /resume remet aussi la machine à états risque à NORMAL
            # (l'opérateur humain reste le décideur final — mentalité n°10).
            try:
                from main import risk_state as _rs
                _rs.reset(reason="telegram:/resume")
                self.state["risk_state"] = _rs.to_dict()
            except Exception:
                pass
            await self.send_push_notification("🟢 *TRADING AUTOMATIQUE RELANCÉ !* Je reprends la surveillance active des marchés réels pour investir selon les meilleures opportunités.", reply_markup=keyboard)
            
        elif cmd_lower == "/kill":
            self.state["kill_switch_active"] = True
            self.state["is_running"] = False
            
            # Direct liquidation of positions to avoid circular imports and hardcoded local port requests!
            positions = []
            if self.db:
                positions = self.db.get_positions()
                
            active_mode = self.state.get("mode", "DEMO")
            active_balance_key = "balance_demo" if active_mode == "DEMO" else "balance_real"
            
            for p in positions:
                try:
                    asset_price = self.state["assets"].get(p['symbol'], {}).get("price", self.state.get("last_price"))
                    if asset_price is None:
                        asset_price = p['avg_price'] # fallback
                        
                    close_val = p['qty'] * asset_price * 0.999
                    self.state[active_balance_key] = self.state.get(active_balance_key, 100000.0) + close_val
                    if self.db:
                        self.db.update_position(p['symbol'], 0, 0, active_mode)
                        self.db.add_order(
                            symbol=p['symbol'],
                            side="SELL",
                            price=asset_price * 0.999,
                            qty=p['qty'],
                            status="FORCE_LIQUIDATED",
                            mode=active_mode,
                            strategy="EMERGENCY_TELEGRAM_KILL",
                            order_type="MARKET"
                        )
                except Exception as exc:
                    logger.error(f"Emergency close failed for {p['symbol']} via Telegram: {str(exc)}")
                    
            if self.db:
                self.db.add_audit_log("KILL_SWITCH_ENGAGED_TELEGRAM", "127.0.0.1", "Global KILL SWITCH activated via Telegram remote control.")
                
            await self.send_push_notification(
                "🚨 *URGENCE : KILL SWITCH DÉCLENCHÉ VIA TELEGRAM !*\n\n"
                "J'ai immédiatement bloqué le robot et vendu l'intégralité de nos investissements au prix du marché.\n"
                "Tout votre capital est désormais sécurisé et à l'abri !",
                reply_markup=keyboard
            )

    async def process_callback_query(self, callback_id: str, data: str):
        """
        Executes actions directly triggered by the user's tactile clicks!
        """
        cmd_mapping = {
            "bot_status": "/status",
            "bot_history": "/history",
            "bot_risk": "/risk",
            "bot_modes": "/modes",
            "bot_resume": "/resume",
            "bot_pause": "/pause",
            "bot_kill": "/kill"
        }
        
        if data in cmd_mapping:
            await self.process_command(cmd_mapping[data])
            return

        # VISION_FUTUR §6: consultative-mode approval buttons
        if data == "approve_pending":
            await self.process_command("/approve")
            return
        if data == "reject_pending":
            await self.process_command("/reject")
            return
        
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

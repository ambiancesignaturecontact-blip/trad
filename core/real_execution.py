"""
Exécution REAL améliorée - LOT 10
Gestion robuste des rejets, retry intelligent, notifications claires.
"""
import logging
import time
from typing import Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("RealExecution")

class RealExecutionEngine:
    """
    Moteur d'exécution REAL amélioré.
    - Gestion intelligente des rejets
    - Retry avec backoff
    - Notifications claires
    - Logging détaillé
    """
    
    def __init__(self, ccxt_client_getter, telegram_bot=None):
        self.get_ccxt = ccxt_client_getter
        self.telegram = telegram_bot
        self.rejected_orders = []
        self.successful_orders = []

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.8, min=0.6, max=8),
        retry=retry_if_exception_type((Exception,)),
        reraise=False
    )
    def place_real_order_with_retry(self, symbol: str, side: str, qty: float, price: float) -> Dict:
        """Place un ordre REAL avec retry intelligent"""
        client = self.get_ccxt()
        
        if not client:
            return {"success": False, "reason": "No CCXT client available"}
        
        try:
            formatted_symbol = symbol.replace("USDT", "/USDT")
            
            order = client.create_order(
                symbol=formatted_symbol,
                type='market',
                side=side.lower(),
                amount=qty,
                params={'clientOrderId': f"real_{int(time.time()*1000)}"}
            )
            
            self.successful_orders.append({
                "timestamp": time.time(),
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": order.get('price', price)
            })
            
            logger.info(f"REAL ORDER SUCCESS: {side} {qty} {symbol} @ {order.get('price', price)}")
            
            return {
                "success": True,
                "order_id": order.get("id"),
                "price": order.get("price", price),
                "filled_qty": order.get("filled", qty),
                "status": order.get("status", "FILLED")
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Classification des erreurs
            if "insufficient" in error_msg or "margin" in error_msg:
                reason = "INSUFFICIENT_MARGIN"
            elif "min" in error_msg or "lot" in error_msg:
                reason = "MIN_SIZE_VIOLATION"
            elif "price" in error_msg:
                reason = "PRICE_DEVIATION"
            else:
                reason = "UNKNOWN_REJECTION"
            
            self.rejected_orders.append({
                "timestamp": time.time(),
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "reason": reason,
                "error": str(e)
            })
            
            logger.error(f"REAL ORDER REJECTED [{reason}]: {side} {qty} {symbol} - {e}")
            
            # Notification Telegram si disponible
            if self.telegram:
                try:
                    self.telegram.send_push_notification(
                        f"⚠️ *ORDRE REAL REJETÉ*\n"
                        f"Symbole: `{symbol}`\n"
                        f"Action: *{side}*\n"
                        f"Raison: `{reason}`\n"
                        f"Quantité: `{qty}`"
                    )
                except:
                    pass
            
            return {
                "success": False,
                "reason": reason,
                "error": str(e)
            }

    def get_execution_stats(self) -> Dict:
        """Statistiques d'exécution"""
        total = len(self.successful_orders) + len(self.rejected_orders)
        success_rate = len(self.successful_orders) / total if total > 0 else 0
        
        return {
            "total_attempts": total,
            "successful": len(self.successful_orders),
            "rejected": len(self.rejected_orders),
            "success_rate": round(success_rate * 100, 1),
            "last_rejections": self.rejected_orders[-5:] if self.rejected_orders else []
        }
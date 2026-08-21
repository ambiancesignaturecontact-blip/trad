"""
Client CCXT authentifié (clés chiffrées en DB) extrait de main.py
(LOT C, F3). Corps inchangés.
"""

import logging

import ccxt
from main import (db)  # noqa: E402

logger = logging.getLogger("InstitutionalTradingBot")  # LOT C : même canal de logs que main


ccxt_client = None


def get_ccxt_client():
    """
    Dynamically loads and instantiates the CCXT Binance/Bybit client
    using securely encrypted keys from the database.
    """
    global ccxt_client
    if ccxt_client is not None:
        return ccxt_client

    api_key = db.get_setting("binance_api_key", decrypt=True)
    secret_key = db.get_setting("binance_secret_key", decrypt=True)

    if api_key and secret_key:
        try:
            ccxt_client = ccxt.binance({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'  # Default to perpetual futures
                }
            })
            ccxt_client.fetch_balance()
            logger.info("CCXT Exchange Client successfully instantiated and authenticated.")
            return ccxt_client
        except Exception as e:
            logger.error(f"Failed to authenticate with real exchange API: {str(e)}")
            ccxt_client = None
    return None


def format_exchange_size(symbol, quantity, price):
    """
    Formats the order size according to the exact lot size filters
    and precision limits of the exchange to avoid API execution rejections.
    """
    client = get_ccxt_client()
    if not client:
        return round(quantity, 5) # Safe fallback

    try:
        if symbol not in client.markets:
            client.load_markets()

        market = client.market(symbol)
        min_qty = market['limits']['amount']['min'] or 0.0001
        max_qty = market['limits']['amount']['max'] or 999999.0

        formatted_qty = client.amount_to_precision(symbol, quantity)
        formatted_qty = float(formatted_qty)
        formatted_qty = max(min_qty, min(formatted_qty, max_qty))
        return formatted_qty
    except Exception as e:
        logger.warning(f"Error formatting lot size precision: {str(e)}. Using safe rounding.")
        return round(quantity, 5)

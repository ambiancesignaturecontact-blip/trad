"""
Real copy-trading mirroring engine (VISION §5 / copy-trading execution).

- fetch_trader_positions(): Hyperliquid PUBLIC API - real positions of any trader
- build_mirror_orders(): scale the trader's book to our allocated capital and
  compute the DELTA vs our current portfolio (only BTC/ETH/SOL USDT pairs)
- execute via OMS when COPYTRADE_EXECUTION=auto + exchange keys are configured,
  otherwise honest SIGNAL_ONLY mode (no fake execution).
"""
import logging

import httpx

logger = logging.getLogger("CopyMirror")

HL_INFO = "https://api.hyperliquid.xyz/info"
COIN_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}


def fetch_trader_positions(trader_id: str, timeout: float = 12.0) -> list[dict]:
    """Returns real signed positions of a trader: [{coin, szi, entry_px, notional_usd}]."""
    try:
        resp = httpx.post(HL_INFO, json={"type": "clearinghouseState", "user": trader_id}, timeout=timeout)
        if resp.status_code != 200:
            return []
        out = []
        for ap in (resp.json().get("assetPositions") or []):
            po = ap.get("position") or {}
            coin = po.get("coin")
            szi = float(po.get("szi") or 0.0)
            entry = float(po.get("entryPx") or 0.0)
            if coin and szi != 0.0:
                out.append({
                    "coin": coin,
                    "symbol": COIN_MAP.get(coin, f"{coin}USDT"),
                    "szi": szi,          # signed: + long / - short
                    "entry_px": entry,
                    "notional_usd": abs(szi * entry),
                })
        return out
    except Exception as e:
        logger.warning(f"fetch_trader_positions failed for {trader_id}: {e}")
        return []


def build_mirror_orders(trader_positions: list[dict], my_positions: dict[str, float],
                        allocated_capital: float, trader_account_value: float,
                        min_notional: float = 10.0, max_asset_pct: float = 0.25) -> list[dict]:
    """
    Computes the DELTA orders to mirror the trader's book scaled to our allocation.
    my_positions: {symbol: qty}. Returns [{symbol, side, qty, reason}].
    """
    if trader_account_value <= 0:
        trader_account_value = sum(p["notional_usd"] for p in trader_positions) or 1.0
    scale = allocated_capital / trader_account_value
    scale = min(scale, 1.0)  # never leverage beyond the trader's own size

    orders = []
    for p in trader_positions:
        symbol = p["symbol"]
        if symbol not in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            continue  # only venues we can execute on
        target_qty = p["szi"] * scale
        cur_qty = float(my_positions.get(symbol, 0.0))
        delta = target_qty - cur_qty
        if abs(delta) * p["entry_px"] < min_notional:
            continue
        side = "BUY" if delta > 0 else "SELL"
        orders.append({
            "symbol": symbol, "side": side, "qty": abs(delta),
            "ref_qty": p["szi"], "scale": round(scale, 6), "reason": "MIRROR",
        })
    return orders


def mirror_status_text(following: dict[str, dict]) -> str:
    """Human-readable summary of the active mirroring (Telegram/dashboard)."""
    if not following:
        return "Aucun trader suivi (mode FOLLOW_ONLY / pas d'allocation)."
    lines = []
    for tid, alloc in following.items():
        mode = alloc.get("mode", "FOLLOW_ONLY")
        cap = alloc.get("allocated_capital", 0.0)
        pnl = alloc.get("pnl_estimate_usd", 0.0)
        lines.append(f"• `{tid[:10]}…` | mode **{mode}** | alloc ${cap:,.0f} | P&L est. {pnl:+,.2f}$")
    return "\n".join(lines)

"""
LOT 63: Centralized outbound API rate limiting (aiolimiter).

Prevents the hot trading loop from hammering exchange / data endpoints and
getting IP-banned (429) or geo-blocked. Every outbound call category shares a
limiter so bursts across symbols stay within each provider's tolerance.
"""
from aiolimiter import AsyncLimiter

# Requests per second, chosen conservatively under each provider's public limits.
bybit_limiter = AsyncLimiter(5, 1.0)      # Bybit public REST: 120 req/5s -> keep it safe
binance_limiter = AsyncLimiter(5, 1.0)    # Binance public REST: 1200 req/min weight-based
yahoo_limiter = AsyncLimiter(3, 1.0)      # Yahoo Finance chart API is rate-sensitive
news_limiter = AsyncLimiter(1, 2.0)       # CryptoCompare / AlphaVantage / Reddit news
rpc_limiter = AsyncLimiter(10, 1.0)       # Ethereum / Arbitrum public RPC nodes

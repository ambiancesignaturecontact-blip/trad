"""
Rate Limiter + Simple Cache for API calls
Utilisé pour protéger les APIs (Binance, Bybit, Yahoo, etc.)
"""
import time
import asyncio
from functools import wraps
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("RateLimiter")

class SimpleRateLimiter:
    """Rate limiter simple et efficace"""
    
    def __init__(self, max_calls: int = 10, period: float = 1.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    async def acquire(self):
        """Attend si nécessaire pour respecter la limite"""
        now = time.time()
        
        # Nettoie les anciens appels
        self.calls = [t for t in self.calls if now - t < self.period]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)
    
    def limit(self, func):
        """Décorateur async pour rate limiting"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await self.acquire()
            return await func(*args, **kwargs)
        return wrapper


class SimpleCache:
    """Cache simple avec TTL"""
    
    def __init__(self, default_ttl: int = 30):
        self.cache: Dict[str, Dict] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if time.time() - entry["timestamp"] > entry["ttl"]:
            del self.cache[key]
            return None
        
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self.cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl or self.default_ttl
        }
    
    def clear(self):
        self.cache.clear()


# Instances globales
api_limiter = SimpleRateLimiter(max_calls=8, period=1.0)  # 8 appels/sec max
price_cache = SimpleCache(default_ttl=8)                   # Cache prix 8 secondes
funding_cache = SimpleCache(default_ttl=300)               # Cache funding 5 minutes

def cached_price(ttl: int = 8):
    """Décorateur pour cacher les prix"""
    def decorator(func):
        @wraps(func)
        async def wrapper(symbol: str, *args, **kwargs):
            cache_key = f"price_{symbol}"
            cached = price_cache.get(cache_key)
            if cached is not None:
                return cached
            
            result = await func(symbol, *args, **kwargs)
            if result is not None:
                price_cache.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator
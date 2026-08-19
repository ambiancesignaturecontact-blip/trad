"""
Institutional HTTP middleware (audit B2-3, B3-1, B3-2, B5-1).

- request_id + request logging (who called what, when, how long)
- security headers (X-Frame-Options, HSTS, nosniff, CSP-lite, referrer)
- JSON exception handler (no raw 500 HTML/tracebacks to clients)
- configurable CORS from ALLOWED_ORIGINS env
- per-IP rate limiting for state-changing POST endpoints (audit B2-2)
"""
import logging
import os
import time
import uuid
from collections import defaultdict, deque

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("HTTP")

# ---- Security headers ----
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "X-XSS-Protection": "1; mode=block",
}
HSTS_HEADER = {"Strict-Transport-Security": "max-age=63072000; includeSubDomains"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        request.state.request_id = rid
        # LOT 7 (PDF Faille 6) : mémorise la VRAIE IP client pour les audit logs
        # (les middlewares s'exécutent dans l'ordre d'ajout ; celui-ci est
        # ajouté après les rate-limiters, donc ip est dispo).
        try:
            if request.client and request.client.host:
                from main import set_request_ip
                set_request_ip(request.client.host)
        except Exception:
            pass
        start = time.perf_counter()
        response = await call_next(request)
        dur_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            f"[{rid}] {request.method} {request.url.path} -> {response.status_code} "
            f"({dur_ms:.1f}ms) {request.client.host if request.client else '?'}"
        )
        response.headers["X-Request-ID"] = rid
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers. NOTE: Telegram Mini-App renders /telegram inside an
    iframe from web.telegram.org / native app, so X-Frame-Options must NOT be
    sent there (would block the mini-app with 'refuses to connect'). We instead
    allow framing via CSP frame-ancestors for Telegram origins on that path."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Telegram Mini-App must be embeddable - no X-Frame-Options on /telegram
        is_telegram = request.url.path.startswith("/telegram")
        for k, v in SECURITY_HEADERS.items():
            if k == "X-Frame-Options" and is_telegram:
                continue
            response.headers.setdefault(k, v)
        if is_telegram:
            # allow Telegram to frame the mini-app (CSP frame-ancestors is the modern way)
            response.headers["Content-Security-Policy"] = (
                "frame-ancestors https://web.telegram.org https://telegram.org;"
            )
        if os.getenv("ENABLE_HSTS", "true").lower() == "true":
            for k, v in HSTS_HEADER.items():
                response.headers.setdefault(k, v)
        return response


class IPRateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window per-IP rate limit on state-changing POST endpoints."""

    def __init__(self, app, max_requests: int = 30, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.hits = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and not request.url.path.startswith("/api/login"):
            ip = request.client.host if request.client else "?"
            now = time.time()
            q = self.hits[ip]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.max_requests:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Slow down."},
                )
            q.append(now)
        return await call_next(request)


class LoginRateLimitMiddleware(BaseHTTPMiddleware):
    """Aggressive per-IP limit on the login endpoint (audit B2-6)."""

    def __init__(self, app, max_attempts: int = 5, window_seconds: int = 60):
        super().__init__(app)
        self.max_attempts = max_attempts
        self.window = window_seconds
        self.attempts = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/api/login":
            ip = request.client.host if request.client else "?"
            now = time.time()
            q = self.attempts[ip]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.max_attempts:
                logger.warning(f"Login rate-limit hit for {ip}")
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many login attempts. Try again later."},
                )
            q.append(now)
        return await call_next(request)


def install_cors(app):
    """Configurable CORS from ALLOWED_ORIGINS env (comma separated)."""
    from fastapi.middleware.cors import CORSMiddleware

    raw = os.getenv("ALLOWED_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if origins:
        logger.info(f"CORS enabled for {origins}")

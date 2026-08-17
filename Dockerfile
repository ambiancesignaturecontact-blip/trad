# ==========================================
# INSTITUTIONAL TRADING BOT - MULTI-STAGE DOCKERFILE (audit B15-1)
# Builder: installs locked deps + builds the React dashboard.
# Runtime: slim image with only runtime artifacts (no build toolchain).
# ==========================================

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Python deps (locked - audit B15-2)
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Frontend build (audit B14-1: React UI served at /app)
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy locked deps + torch (CPU wheel)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Copy the application
COPY . .

# Copy the built React dashboard
COPY --from=builder /build/frontend/dist /app/frontend/dist

# Expose port (Railway overrides)
EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Health check (stdlib urllib only)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/status', timeout=5)" || exit 1

# Start command for Railway
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info

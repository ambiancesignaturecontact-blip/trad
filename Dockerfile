# ==========================================
# INSTITUTIONAL TRADING BOT - RAILWAY DOCKERFILE
# ==========================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port (Railway will override)
EXPOSE 8080

# Environment variables for Railway
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/api/status', timeout=5)" || exit 1

# Start command for Railway
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info
# Changelog

## 2026-08-17 — Institutional hardening + autonomous AI + audit remediation
- Boot crash fix (asyncio at module level), syntax repairs, deps (torch CPU enabled)
- Prometheus /metrics, rate limiting, loop hardening, config checklist
- Full roadmap: idempotence, REAL fill confirmation, DB backups+retention+restore,
  JWT auth+2FA, graceful shutdown, smoke/backtest tests, dead-code cleanup
- Real public copy-trading leaderboard (Hyperliquid) + autonomous AI cycle (LOT 66)
- **Audit remediation (this release):** SL/TP/trailing position manager, OMS/EMS
  routing wired into live execution + periodic reconciliation, input validation,
  login/JWT/prod secret blocks, security headers + CORS + request logging,
  DB pooling + WAL + indexes, WS heartbeat + client caps + dashboard auto-reconnect,
  config.yaml + .env loading, VPIN/Kyle/on-chain as signal features, hidden-layer
  PPO + deeper LSTM + honest model names, GAN/RLHF wired, max per-asset cap,
  daily Monte-Carlo stress, liquidation alerts, honest arbitrage labeling +
  DEX broadcast, copy-trading follow-only tracking, mini-app actions awaited,
  daily P&L report + health score + Telegram concierge, autopilot paper gate,
  TradingView webhooks, pagination + /api/v1, multi-stage Docker + locked deps +
  CI docker/lint, 13 new tests (71 total), SOR sell-side ranking bug fixed.

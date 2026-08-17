# 🔐 Security Policy

## Supported Versions
- `main` branch: actively maintained, security fixes applied on discovery.

## Reporting a Vulnerability
**Do NOT open a public issue.** Email the maintainers via a private channel
(GitHub private vulnerability report, or the Telegram bot contact if known).
Include: affected endpoint/module, steps to reproduce, and impact assessment.

We aim to acknowledge reports within 72h and release a fix as soon as possible.

## Security posture
- API keys encrypted at rest (Fernet). Set `FERNET_KEY` in production.
- JWT + optional TOTP 2FA; auth is FORCED in REAL mode.
- Hardcoded default secrets block startup when auth is enforced.
- Rate limits on state-changing endpoints and on login.
- Never commit `secret.key`, `trading_platform.db`, `.env` (see .gitignore).

#!/usr/bin/env python3
"""
Restore procedure (audit B4-6).

Usage:
    python scripts/restore_backup.py <path/to/trading_platform_YYYYMMDD_HHMMSS.db> [--dry-run]

Stops nothing (bot should be stopped first), copies the snapshot over the live
SQLite DB, then re-creates missing indexes/PRAGMAs via DBManager.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.db_manager import DB_PATH  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Restore a SQLite backup snapshot")
    ap.add_argument("backup", help="Path to the .db snapshot to restore")
    ap.add_argument("--dry-run", action="store_true", help="Validate without copying")
    args = ap.parse_args()

    if not os.path.exists(args.backup):
        print(f"❌ Backup not found: {args.backup}")
        sys.exit(1)
    if not args.backup.endswith(".db"):
        print("❌ Expected a .db snapshot file.")
        sys.exit(1)

    size = os.path.getsize(args.backup)
    print(f"ℹ️  Snapshot: {args.backup} ({size/1024:.0f} KB)")
    print(f"ℹ️  Target  : {DB_PATH}")
    if args.dry_run:
        print("✅ Dry-run: no changes made.")
        return

    # safety: require explicit confirmation
    confirm = input("⚠️  This OVERWRITES the live database. Type 'RESTORE' to continue: ")
    if confirm != "RESTORE":
        print("Aborted.")
        sys.exit(1)

    # copy with sqlite backup API for consistency
    import sqlite3
    src = sqlite3.connect(args.backup)
    dst = sqlite3.connect(DB_PATH)
    src.backup(dst)
    dst.close()
    src.close()
    print("✅ Database restored.")

    from database.db_manager import DBManager
    DBManager()  # re-applies WAL, busy_timeout, indexes
    print("✅ Indexes/PRAGMAs re-applied. Restart the bot.")


if __name__ == "__main__":
    main()

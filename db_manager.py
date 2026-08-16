import sqlite3
import os
import json
import logging
from cryptography.fernet import Fernet

logger = logging.getLogger("DBManager")

# Determine connection mode: SQLite local or Supabase PostgreSQL
DATABASE_URL = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
DB_PATH = "/home/user/trading_platform.db"
KEY_PATH = "/home/user/secret.key"

class DBManager:
    """
    Dual-dialect DB manager supporting SQLite for local zero-config runs,
    and PostgreSQL (Supabase) for cloud production environments.
    
    Translates schema declarations and upsert conflicts automatically.
    """
    def __init__(self):
        self.initialize_key()
        self.cipher = Fernet(self.load_key())
        self.is_postgres = DATABASE_URL is not None and (DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"))
        
        if self.is_postgres:
            logger.info("Database Mode: Production Supabase PostgreSQL")
            import psycopg2
            # Correct potentially old format of connection string (postgres:// -> postgresql://)
            self.pg_url = DATABASE_URL.replace("postgres://", "postgresql://")
        else:
            logger.info("Database Mode: Local SQLite Dev")
            
        self.init_db()

    def initialize_key(self):
        """Generates and persists an AES encryption key if not exists."""
        # Check if environment key exists first (Twelve-Factor App standard for Cloud deployments like Vercel)
        env_key = os.getenv("FERNET_KEY")
        if env_key:
            self.key = env_key.encode()
            return
            
        if not os.path.exists(KEY_PATH):
            key = Fernet.generate_key()
            with open(KEY_PATH, "wb") as key_file:
                key_file.write(key)
        
        self.key = self.load_key()

    def load_key(self):
        env_key = os.getenv("FERNET_KEY")
        if env_key:
            return env_key.encode()
        with open(KEY_PATH, "rb") as key_file:
            return key_file.read()

    def get_connection(self):
        if self.is_postgres:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            conn = psycopg2.connect(self.pg_url, cursor_factory=RealDictCursor)
            return conn
        else:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            return conn

    def init_db(self):
        """
        Creates all required tables using dual-dialect queries matching
        both SQLite and PostgreSQL standards.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.is_postgres:
                # 1. Orders
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        symbol VARCHAR(20),
                        side VARCHAR(10),
                        price DOUBLE PRECISION,
                        qty DOUBLE PRECISION,
                        status VARCHAR(20),
                        mode VARCHAR(10),
                        strategy VARCHAR(50),
                        order_type VARCHAR(20)
                    )
                """)
                
                # 2. Positions
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol VARCHAR(20) PRIMARY KEY,
                        qty DOUBLE PRECISION,
                        avg_price DOUBLE PRECISION,
                        mode VARCHAR(10)
                    )
                """)
                
                # 3. System Settings
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_settings (
                        key VARCHAR(100) PRIMARY KEY,
                        value TEXT
                    )
                """)
                
                # 4. Audit Logs
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        action VARCHAR(100),
                        user_ip VARCHAR(50),
                        details TEXT
                    )
                """)
                
                # 5. Copytrading
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS copy_allocations (
                        trader_id VARCHAR(50) PRIMARY KEY,
                        allocated_capital DOUBLE PRECISION,
                        active INTEGER DEFAULT 0
                    )
                """)
            else:
                # SQLite dialect
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT,
                        side TEXT,
                        price REAL,
                        qty REAL,
                        status TEXT,
                        mode TEXT,
                        strategy TEXT,
                        order_type TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol TEXT PRIMARY KEY,
                        qty REAL,
                        avg_price REAL,
                        mode TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                        action TEXT,
                        user_ip TEXT,
                        details TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS copy_allocations (
                        trader_id TEXT PRIMARY KEY,
                        allocated_capital REAL,
                        active INTEGER DEFAULT 0
                    )
                """)
                
            if self.is_postgres:
                conn.commit()
            else:
                conn.commit()

    # Encryption Helpers
    def encrypt_val(self, text: str) -> str:
        return self.cipher.encrypt(text.encode()).decode()

    def decrypt_val(self, encrypted_text: str) -> str:
        try:
            return self.cipher.decrypt(encrypted_text.encode()).decode()
        except Exception:
            return ""

    # Settings API
    def save_setting(self, key: str, value: str, encrypt=False):
        final_val = self.encrypt_val(value) if encrypt else value
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO system_settings (key, value)
                    VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """, (key, final_val))
            else:
                cursor.execute(
                    "INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)",
                    (key, final_val)
                )
            conn.commit()

    def get_setting(self, key: str, decrypt=False) -> str:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT value FROM system_settings WHERE key = %s", (key,))
            else:
                cursor.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                val = row['value']
                return self.decrypt_val(val) if decrypt else val
            return ""

    # Orders API
    def add_order(self, symbol, side, price, qty, status, mode, strategy, order_type):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO orders (symbol, side, price, qty, status, mode, strategy, order_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (symbol, side, price, qty, status, mode, strategy, order_type))
                order_id = cursor.fetchone()['id']
            else:
                cursor.execute("""
                    INSERT INTO orders (symbol, side, price, qty, status, mode, strategy, order_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, side, price, qty, status, mode, strategy, order_type))
                order_id = cursor.lastrowid
            conn.commit()
            return order_id

    def get_all_orders(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT * FROM orders ORDER BY timestamp DESC LIMIT 100")
            else:
                cursor.execute("SELECT * FROM orders ORDER BY timestamp DESC LIMIT 100")
            rows = cursor.fetchall()
            # Convert row objects properly to dictionaries
            return [dict(r) for r in rows]

    # Positions API
    def update_position(self, symbol, qty, avg_price, mode):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if qty <= 0:
                if self.is_postgres:
                    cursor.execute("DELETE FROM positions WHERE symbol = %s", (symbol,))
                else:
                    cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            else:
                if self.is_postgres:
                    cursor.execute("""
                        INSERT INTO positions (symbol, qty, avg_price, mode)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (symbol) DO UPDATE SET qty = EXCLUDED.qty, avg_price = EXCLUDED.avg_price, mode = EXCLUDED.mode
                    """, (symbol, qty, avg_price, mode))
                else:
                    cursor.execute("""
                        INSERT OR REPLACE INTO positions (symbol, qty, avg_price, mode)
                        VALUES (?, ?, ?, ?)
                    """, (symbol, qty, avg_price, mode))
            conn.commit()

    def get_positions(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # Audit Logs API
    def add_audit_log(self, action, user_ip, details):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO audit_logs (action, user_ip, details)
                    VALUES (%s, %s, %s)
                """, (action, user_ip, details))
            else:
                cursor.execute("""
                    INSERT INTO audit_logs (action, user_ip, details)
                    VALUES (?, ?, ?)
                """, (action, user_ip, details))
            conn.commit()

    def get_audit_logs(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 50")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # Copytrading API
    def save_copy_allocation(self, trader_id, capital, active):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            active_int = 1 if active else 0
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO copy_allocations (trader_id, allocated_capital, active)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (trader_id) DO UPDATE SET allocated_capital = EXCLUDED.allocated_capital, active = EXCLUDED.active
                """, (trader_id, capital, active_int))
            else:
                cursor.execute("""
                    INSERT OR REPLACE INTO copy_allocations (trader_id, allocated_capital, active)
                    VALUES (?, ?, ?)
                """, (trader_id, capital, active_int))
            conn.commit()

    def get_copy_allocations(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM copy_allocations")
            rows = cursor.fetchall()
            return {r['trader_id']: {"allocated_capital": r['allocated_capital'], "active": bool(r['active'])} for r in rows}

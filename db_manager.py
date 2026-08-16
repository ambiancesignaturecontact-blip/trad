import sqlite3
import os
import json
import logging
import hashlib
import base64
import pandas as pd

logger = logging.getLogger("DBManager")

DATABASE_URL = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
# Highly portable paths, dynamically resolved relative to the current working directory or environment variables
DB_PATH = os.getenv("SQLITE_DB_PATH", os.path.join(os.getcwd(), "trading_platform.db"))
KEY_PATH = os.getenv("SECRET_KEY_PATH", os.path.join(os.getcwd(), "secret.key"))

class DBManager:
    """
    Dual-dialect, Multi-User SaaS DB manager supporting SQLite for local runs,
    and PostgreSQL (Supabase) for production environments.
    
    Includes an automatic schema migration layer (adding user_id and hash columns to existing tables),
    a cryptographically chained double-audit ledger, and zero-downtime failover.
    """
    def __init__(self):
        self.initialize_key()
        
        from cryptography.fernet import Fernet
        self.cipher = Fernet(self.load_key())
        self.is_postgres = DATABASE_URL is not None and (DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"))
        
        if self.is_postgres:
            logger.info("Attempting to connect to Production Supabase PostgreSQL...")
            self.pg_url = DATABASE_URL.replace("postgres://", "postgresql://")
            try:
                import psycopg2
                conn = psycopg2.connect(self.pg_url, connect_timeout=5)
                conn.close()
                logger.info("Successfully connected and authenticated with Supabase PostgreSQL!")
            except Exception as e:
                logger.error(f"PostgreSQL Connection Failed: {str(e)}")
                logger.warning("FALLING BACK TO LOCAL SQLITE DATABASE TO ENSURE 100% BOT UPTIME...")
                self.is_postgres = False
        else:
            logger.info("Database Mode: Local SQLite Dev")
            
        self.init_db()

    def initialize_key(self):
        """Generates and persists an AES encryption key if not exists."""
        env_key = os.getenv("FERNET_KEY")
        if env_key:
            hashed = hashlib.sha256(env_key.encode()).digest()
            self.key = base64.urlsafe_b64encode(hashed)
            return
            
        if not os.path.exists(KEY_PATH):
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            with open(KEY_PATH, "wb") as key_file:
                key_file.write(key)
        
        self.key = self.load_key()

    def load_key(self):
        env_key = os.getenv("FERNET_KEY")
        if env_key:
            hashed = hashlib.sha256(env_key.encode()).digest()
            return base64.urlsafe_b64encode(hashed)
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
        Creates all required tables and runs automatic on-the-fly migrations
        to add columns like user_id and hash to pre-existing production databases.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.is_postgres:
                # SaaS Users Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE,
                        password_hash VARCHAR(128),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create a default admin user if not exists
                cursor.execute("""
                    INSERT INTO users (id, username, password_hash)
                    VALUES (1, 'admin_quant', 'hash_admin_secret')
                    ON CONFLICT (username) DO NOTHING
                """)
                
                # Orders
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
                
                # Positions
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol VARCHAR(20) PRIMARY KEY,
                        qty DOUBLE PRECISION,
                        avg_price DOUBLE PRECISION,
                        mode VARCHAR(10)
                    )
                """)
                
                # System Settings
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_settings (
                        key VARCHAR(100) PRIMARY KEY,
                        value TEXT
                    )
                """)
                
                # Audit Logs
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        action VARCHAR(100),
                        user_ip VARCHAR(50),
                        details TEXT
                    )
                """)
                
                # Copytrading
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS copy_allocations (
                        trader_id VARCHAR(50) PRIMARY KEY,
                        allocated_capital DOUBLE PRECISION,
                        active INTEGER DEFAULT 0
                    )
                """)
                
                # Market Candles Cache Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_candles (
                        symbol VARCHAR(20),
                        timestamp VARCHAR(50),
                        open DOUBLE PRECISION,
                        high DOUBLE PRECISION,
                        low DOUBLE PRECISION,
                        close DOUBLE PRECISION,
                        volume DOUBLE PRECISION,
                        PRIMARY KEY (symbol, timestamp)
                    )
                """)
            else:
                # SQLite dialect
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE,
                        password_hash TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    INSERT OR IGNORE INTO users (id, username, password_hash)
                    VALUES (1, 'admin_quant', 'hash_admin_secret')
                """)
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
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_candles (
                        symbol TEXT,
                        timestamp TEXT,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        PRIMARY KEY (symbol, timestamp)
                    )
                """)
                
            conn.commit()

            # RUN DYNAMIC AUTO-MIGRATIONS FOR PRODUCTION DATABASES (Supabase & local SQLite)
            # This safely injects 'user_id' and 'hash' columns to pre-existing tables on-the-fly!
            tables_to_migrate = ["orders", "positions", "system_settings", "audit_logs", "copy_allocations"]
            for tbl in tables_to_migrate:
                try:
                    cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN user_id INTEGER DEFAULT 1")
                    conn.commit()
                    logger.info(f"Database Migration: Added 'user_id' column to {tbl} table.")
                except Exception:
                    # Column already exists, rollback PG transaction safely
                    if self.is_postgres:
                        conn.rollback()
                        
            # Migrate 'hash' column for audit_logs
            try:
                if self.is_postgres:
                    cursor.execute("ALTER TABLE audit_logs ADD COLUMN hash VARCHAR(64)")
                else:
                    cursor.execute("ALTER TABLE audit_logs ADD COLUMN hash TEXT")
                conn.commit()
                logger.info("Database Migration: Added cryptographically chained 'hash' column to audit_logs.")
            except Exception:
                if self.is_postgres:
                    conn.rollback()

    # Encryption Helpers
    def encrypt_val(self, text: str) -> str:
        return self.cipher.encrypt(text.encode()).decode()

    def decrypt_val(self, encrypted_text: str) -> str:
        try:
            return self.cipher.decrypt(encrypted_text.encode()).decode()
        except Exception:
            return ""

    # Settings API
    def save_setting(self, key: str, value: str, user_id=1, encrypt=False):
        final_val = self.encrypt_val(value) if encrypt else value
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO system_settings (user_id, key, value)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, key) DO UPDATE SET value = EXCLUDED.value
                """, (user_id, key, final_val))
            else:
                cursor.execute("""
                    INSERT OR REPLACE INTO system_settings (user_id, key, value)
                    VALUES (?, ?, ?)
                """, (user_id, key, final_val))
            conn.commit()

    def get_setting(self, key: str, user_id=1, decrypt=False) -> str:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT value FROM system_settings WHERE user_id = %s AND key = %s", (user_id, key))
            else:
                cursor.execute("SELECT value FROM system_settings WHERE user_id = ? AND key = ?", (user_id, key))
            row = cursor.fetchone()
            if row:
                val = row['value']
                return self.decrypt_val(val) if decrypt else val
            return ""

    # Orders API
    def add_order(self, symbol, side, price, qty, status, mode, strategy, order_type, user_id=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO orders (user_id, symbol, side, price, qty, status, mode, strategy, order_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (user_id, symbol, side, price, qty, status, mode, strategy, order_type))
                order_id = cursor.fetchone()['id']
            else:
                cursor.execute("""
                    INSERT INTO orders (user_id, symbol, side, price, qty, status, mode, strategy, order_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, symbol, side, price, qty, status, mode, strategy, order_type))
                order_id = cursor.lastrowid
            conn.commit()
            return order_id

    def get_all_orders(self, user_id=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT * FROM orders WHERE user_id = %s ORDER BY timestamp DESC LIMIT 100", (user_id,))
            else:
                cursor.execute("SELECT * FROM orders WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100", (user_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # Positions API
    def update_position(self, symbol, qty, avg_price, mode, user_id=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if qty <= 0:
                if self.is_postgres:
                    cursor.execute("DELETE FROM positions WHERE user_id = %s AND symbol = %s", (user_id, symbol))
                else:
                    cursor.execute("DELETE FROM positions WHERE user_id = ? AND symbol = ?", (user_id, symbol))
            else:
                if self.is_postgres:
                    cursor.execute("""
                        INSERT INTO positions (user_id, symbol, qty, avg_price, mode)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, symbol) DO UPDATE SET qty = EXCLUDED.qty, avg_price = EXCLUDED.avg_price, mode = EXCLUDED.mode
                    """, (user_id, symbol, qty, avg_price, mode))
                else:
                    cursor.execute("""
                        INSERT OR REPLACE INTO positions (user_id, symbol, qty, avg_price, mode)
                        VALUES (?, ?, ?, ?, ?)
                    """, (user_id, symbol, qty, avg_price, mode))
            conn.commit()

    def get_positions(self, user_id=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT * FROM positions WHERE user_id = %s", (user_id,))
            else:
                cursor.execute("SELECT * FROM positions WHERE user_id = ?", (user_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # Cryptographically Chained Audit Logs API
    def add_audit_log(self, action, user_ip, details, user_id=1):
        """
        Saves a new audit log with a SHA-256 cryptographic chain hash,
        binding current log details to the previous log entry.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Fetch the previous log's hash
            prev_hash = "GENESIS_ROOT_HASH"
            if self.is_postgres:
                cursor.execute("SELECT hash FROM audit_logs WHERE user_id = %s ORDER BY id DESC LIMIT 1", (user_id,))
            else:
                cursor.execute("SELECT hash FROM audit_logs WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,))
            
            row = cursor.fetchone()
            if row and row['hash']:
                prev_hash = row['hash']
                
            # Compute current block hash (concatenating prev_hash + action + details + user_ip)
            content_str = f"{prev_hash}_{action}_{details}_{user_ip}"
            current_hash = hashlib.sha256(content_str.encode()).hexdigest()
            
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO audit_logs (user_id, action, user_ip, details, hash)
                    VALUES (%s, %s, %s, %s, %s)
                """, (user_id, action, user_ip, details, current_hash))
            else:
                cursor.execute("""
                    INSERT INTO audit_logs (user_id, action, user_ip, details, hash)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, action, user_ip, details, current_hash))
                
            conn.commit()
            logger.info(f"Cryptographically chained audit log created. Hash: {current_hash[:16]}...")

    def get_audit_logs(self, user_id=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT * FROM audit_logs WHERE user_id = %s ORDER BY timestamp DESC LIMIT 50", (user_id,))
            else:
                cursor.execute("SELECT * FROM audit_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50", (user_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # Copytrading API
    def save_copy_allocation(self, trader_id, capital, active, user_id=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            active_int = 1 if active else 0
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO copy_allocations (user_id, trader_id, allocated_capital, active)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, trader_id) DO UPDATE SET allocated_capital = EXCLUDED.allocated_capital, active = EXCLUDED.active
                """, (user_id, trader_id, capital, active_int))
            else:
                cursor.execute("""
                    INSERT OR REPLACE INTO copy_allocations (user_id, trader_id, allocated_capital, active)
                    VALUES (?, ?, ?, ?)
                """, (user_id, trader_id, capital, active_int))
            conn.commit()

    def get_copy_allocations(self, user_id=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT * FROM copy_allocations WHERE user_id = %s", (user_id,))
            else:
                cursor.execute("SELECT * FROM copy_allocations WHERE user_id = ?", (user_id,))
            rows = cursor.fetchall()
            return {r['trader_id']: {"allocated_capital": r['allocated_capital'], "active": bool(r['active'])} for r in rows}

    # Market Candles Cache API
    def save_candles(self, symbol: str, df_bars: pd.DataFrame):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for idx, row in df_bars.iterrows():
                ts_str = str(idx)
                if self.is_postgres:
                    cursor.execute("""
                        INSERT INTO market_candles (symbol, timestamp, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, timestamp) DO UPDATE 
                        SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, 
                            close = EXCLUDED.close, volume = EXCLUDED.volume
                    """, (symbol, ts_str, float(row['open']), float(row['high']), float(row['low']), float(row['close']), float(row['volume'])))
                else:
                    cursor.execute("""
                        INSERT OR REPLACE INTO market_candles (symbol, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (symbol, ts_str, float(row['open']), float(row['high']), float(row['low']), float(row['close']), float(row['volume'])))
            conn.commit()

    def load_candles(self, symbol: str, limit=200) -> pd.DataFrame:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("""
                    SELECT timestamp, open, high, low, close, volume 
                    FROM market_candles 
                    WHERE symbol = %s 
                    ORDER BY timestamp DESC LIMIT %s
                """, (symbol, limit))
            else:
                cursor.execute("""
                    SELECT timestamp, open, high, low, close, volume 
                    FROM market_candles 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC LIMIT ?
                """, (symbol, limit))
                
            rows = cursor.fetchall()
            if not rows:
                return pd.DataFrame()
                
            data = []
            for r in reversed(rows):
                data.append({
                    "timestamp": pd.to_datetime(r['timestamp']),
                    "open": float(r['open']),
                    "high": float(r['high']),
                    "low": float(r['low']),
                    "close": float(r['close']),
                    "volume": float(r['volume'])
                })
            df = pd.DataFrame(data).set_index("timestamp")
            return df

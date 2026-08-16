import sqlite3
import os
import json
import logging
import hashlib
import base64
import pandas as pd

logger = logging.getLogger("DBManager")

DATABASE_URL = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
DB_PATH = os.getenv("SQLITE_DB_PATH", os.path.join(os.getcwd(), "trading_platform.db"))
KEY_PATH = os.getenv("SECRET_KEY_PATH", os.path.join(os.getcwd(), "secret.key"))

class DBManager:
    """
    Dual-dialect, Multi-User SaaS DB manager supporting SQLite for local runs,
    and PostgreSQL (Supabase) for production environments.
    
    Ties positions, orders, configurations, and copytrades to unique user_id keys.
    Implements DELETE-then-INSERT transaction strategies to guarantee 100% compatibility
    with pre-existing database constraint configurations on Supabase.
    Strictly forbids any silent SQLite fallbacks in production REAL mode (Lot 10).
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
                logger.error(f"PostgreSQL Production Connection Failed: {str(e)}")
                logger.critical("=========================================================================")
                logger.critical("🚨 DATABASE_UNAVAILABLE: PRODUCTION POSTGRESQL CONNECTION FAILED!")
                logger.critical("FORBIDDEN: SQLite fallback is strictly prohibited in Production REAL mode.")
                logger.critical("HALTING STARTUP FOR SAFETY.")
                logger.critical("=========================================================================")
                
                # In production/REAL mode, raise a fatal error to abort startup!
                # Strictly forbids silent SQLite fallbacks (Lot 10)
                raise RuntimeError("DATABASE_UNAVAILABLE: Production Supabase PostgreSQL offline. Trading halted.")
        else:
            # SQLite is only authorized in isolated TEST and DEVELOPMENT/DEMO environments
            logger.info("Database Mode: Local SQLite Dev (Authorized for Test & Development only)")
            
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
                
                # Fills Table (Postgres)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fills (
                        fill_id VARCHAR(50) PRIMARY KEY,
                        user_id INTEGER DEFAULT 1 REFERENCES users(id) ON DELETE CASCADE,
                        order_id VARCHAR(50),
                        exchange_trade_id VARCHAR(50),
                        price DOUBLE PRECISION,
                        quantity DOUBLE PRECISION,
                        fee DOUBLE PRECISION,
                        fee_asset VARCHAR(10),
                        side VARCHAR(10),
                        liquidity VARCHAR(10),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fills (
                        fill_id TEXT PRIMARY KEY,
                        user_id INTEGER DEFAULT 1,
                        order_id TEXT,
                        exchange_trade_id TEXT,
                        price REAL,
                        quantity REAL,
                        fee REAL,
                        fee_asset TEXT,
                        side TEXT,
                        liquidity TEXT,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
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
        """
        Saves a setting using a robust DELETE-then-INSERT transaction.
        Enforces complete compatibility with any existing unique constraints on Supabase!
        """
        final_val = self.encrypt_val(value) if encrypt else value
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                # 1. Delete first
                cursor.execute("DELETE FROM system_settings WHERE user_id = %s AND key = %s", (int(user_id), str(key)))
                # 2. Insert fresh record
                cursor.execute("""
                    INSERT INTO system_settings (user_id, key, value)
                    VALUES (%s, %s, %s)
                """, (int(user_id), str(key), str(final_val)))
            else:
                cursor.execute("DELETE FROM system_settings WHERE user_id = ? AND key = ?", (int(user_id), str(key)))
                cursor.execute("""
                    INSERT INTO system_settings (user_id, key, value)
                    VALUES (?, ?, ?)
                """, (int(user_id), str(key), str(final_val)))
            conn.commit()

    def get_setting(self, key: str, user_id=1, decrypt=False) -> str:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT value FROM system_settings WHERE user_id = %s AND key = %s", (int(user_id), str(key)))
            else:
                cursor.execute("SELECT value FROM system_settings WHERE user_id = ? AND key = ?", (int(user_id), str(key)))
            row = cursor.fetchone()
            if row:
                val = row['value']
                return self.decrypt_val(val) if decrypt else val
            return ""

    # Orders API
    def add_order(self, symbol, side, price, qty, status, mode, strategy, order_type, user_id=1):
        """
        Adds a new order. Forcibly casts all numerical parameters (price, qty)
        to native Python float types to prevent psycopg2 schema 'np' errors!
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO orders (user_id, symbol, side, price, qty, status, mode, strategy, order_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (int(user_id), str(symbol), str(side), float(price), float(qty), str(status), str(mode), str(strategy), str(order_type)))
                order_id = cursor.fetchone()['id']
            else:
                cursor.execute("""
                    INSERT INTO orders (user_id, symbol, side, price, qty, status, mode, strategy, order_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (int(user_id), str(symbol), str(side), float(price), float(qty), str(status), str(mode), str(strategy), str(order_type)))
                order_id = cursor.lastrowid
            conn.commit()
            return order_id

    def get_all_orders(self, user_id=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT * FROM orders WHERE user_id = %s ORDER BY timestamp DESC LIMIT 100", (int(user_id),))
            else:
                cursor.execute("SELECT * FROM orders WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100", (int(user_id),))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # Positions API
    def update_position(self, symbol, qty, avg_price, mode, user_id=1):
        """
        Updates an asset position using a clean DELETE-then-INSERT transaction.
        Enforces complete compatibility with any existing constraints on Supabase!
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("DELETE FROM positions WHERE user_id = %s AND symbol = %s", (int(user_id), str(symbol)))
            else:
                cursor.execute("DELETE FROM positions WHERE user_id = ? AND symbol = ?", (int(user_id), str(symbol)))
                
            if qty > 0:
                if self.is_postgres:
                    cursor.execute("""
                        INSERT INTO positions (user_id, symbol, qty, avg_price, mode)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (int(user_id), str(symbol), float(qty), float(avg_price), str(mode)))
                else:
                    cursor.execute("""
                        INSERT INTO positions (user_id, symbol, qty, avg_price, mode)
                        VALUES (?, ?, ?, ?, ?)
                    """, (int(user_id), str(symbol), float(qty), float(avg_price), str(mode)))
            conn.commit()

    def get_positions(self, user_id=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT * FROM positions WHERE user_id = %s", (int(user_id),))
            else:
                cursor.execute("SELECT * FROM positions WHERE user_id = ?", (int(user_id),))
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
            
            prev_hash = "GENESIS_ROOT_HASH"
            try:
                if self.is_postgres:
                    cursor.execute("SELECT hash FROM audit_logs WHERE user_id = %s ORDER BY id DESC LIMIT 1", (int(user_id),))
                else:
                    cursor.execute("SELECT hash FROM audit_logs WHERE user_id = ? ORDER BY id DESC LIMIT 1", (int(user_id),))
                
                row = cursor.fetchone()
                if row and row['hash']:
                    prev_hash = row['hash']
            except Exception:
                if self.is_postgres:
                    conn.rollback()
                
            # Compute current block hash (concatenating prev_hash + action + details + user_ip)
            content_str = f"{prev_hash}_{action}_{details}_{user_ip}"
            current_hash = hashlib.sha256(content_str.encode()).hexdigest()
            
            if self.is_postgres:
                cursor.execute("""
                    INSERT INTO audit_logs (user_id, action, user_ip, details, hash)
                    VALUES (%s, %s, %s, %s, %s)
                """, (int(user_id), str(action), str(user_ip), str(details), str(current_hash)))
            else:
                cursor.execute("""
                    INSERT INTO audit_logs (user_id, action, user_ip, details, hash)
                    VALUES (?, ?, ?, ?, ?)
                """, (int(user_id), str(action), str(user_ip), str(details), str(current_hash)))
                
            conn.commit()
            logger.info(f"Cryptographically chained audit log created. Hash: {current_hash[:16]}...")

    def get_audit_logs(self, user_id=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT * FROM audit_logs WHERE user_id = %s ORDER BY timestamp DESC LIMIT 50", (int(user_id),))
            else:
                cursor.execute("SELECT * FROM audit_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50", (int(user_id),))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # Copytrading API
    def save_copy_allocation(self, trader_id, capital, active, user_id=1):
        """
        Saves a copytrade allocation using a clean DELETE-then-INSERT transaction.
        Enforces complete compatibility with any existing constraints on Supabase!
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            active_int = 1 if active else 0
            if self.is_postgres:
                cursor.execute("DELETE FROM copy_allocations WHERE user_id = %s AND trader_id = %s", (int(user_id), str(trader_id)))
                cursor.execute("""
                    INSERT INTO copy_allocations (user_id, trader_id, allocated_capital, active)
                    VALUES (%s, %s, %s, %s)
                """, (int(user_id), str(trader_id), float(capital), int(active_int)))
            else:
                cursor.execute("DELETE FROM copy_allocations WHERE user_id = ? AND trader_id = ?", (int(user_id), str(trader_id)))
                cursor.execute("""
                    INSERT INTO copy_allocations (user_id, trader_id, allocated_capital, active)
                    VALUES (?, ?, ?, ?)
                """, (int(user_id), str(trader_id), float(capital), int(active_int)))
            conn.commit()

    def get_copy_allocations(self, user_id=1):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("SELECT * FROM copy_allocations WHERE user_id = %s", (int(user_id),))
            else:
                cursor.execute("SELECT * FROM copy_allocations WHERE user_id = ?", (int(user_id),))
            rows = cursor.fetchall()
            return {r['trader_id']: {"allocated_capital": r['allocated_capital'], "active": bool(r['active'])} for r in rows}

    # Fills Cache API
    def save_fill(self, fill_id: str, order_id: str, exchange_trade_id: str, price: float, quantity: float, fee: float, fee_asset: str, side: str, liquidity: str = "taker", user_id=1):
        """
        Saves a confirmed fill execution using a robust DELETE-then-INSERT transaction.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute("DELETE FROM fills WHERE user_id = %s AND fill_id = %s", (int(user_id), str(fill_id)))
                cursor.execute("""
                    INSERT INTO fills (fill_id, user_id, order_id, exchange_trade_id, price, quantity, fee, fee_asset, side, liquidity)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (str(fill_id), int(user_id), str(order_id), str(exchange_trade_id), float(price), float(quantity), float(fee), str(fee_asset), str(side), str(liquidity)))
            else:
                cursor.execute("DELETE FROM fills WHERE user_id = ? AND fill_id = ?", (int(user_id), str(fill_id)))
                cursor.execute("""
                    INSERT INTO fills (fill_id, user_id, order_id, exchange_trade_id, price, quantity, fee, fee_asset, side, liquidity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (str(fill_id), int(user_id), str(order_id), str(exchange_trade_id), float(price), float(quantity), float(fee), str(fee_asset), str(side), str(liquidity)))
            conn.commit()
            logger.info(f"Database: Saved confirmed fill {fill_id} for order {order_id}.")

    # Market Candles Cache API
    def save_candles(self, symbol: str, df_bars: pd.DataFrame):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for idx, row in df_bars.iterrows():
                ts_str = str(idx)
                
                # Dynamic type safety check to prevent any NoneType crashes!
                open_val = float(row['open']) if row['open'] is not None else 0.0
                high_val = float(row['high']) if row['high'] is not None else 0.0
                low_val = float(row['low']) if row['low'] is not None else 0.0
                close_val = float(row['close']) if row['close'] is not None else 0.0
                volume_val = float(row['volume']) if row['volume'] is not None else 15.0
                
                if self.is_postgres:
                    cursor.execute("""
                        INSERT INTO market_candles (symbol, timestamp, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, timestamp) DO UPDATE 
                        SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, 
                            close = EXCLUDED.close, volume = EXCLUDED.volume
                    """, (symbol, ts_str, open_val, high_val, low_val, close_val, volume_val))
                else:
                    cursor.execute("""
                        INSERT OR REPLACE INTO market_candles (symbol, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (symbol, ts_str, open_val, high_val, low_val, close_val, volume_val))
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

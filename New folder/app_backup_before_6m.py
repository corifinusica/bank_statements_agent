# database.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "app.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")  # 30s
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        tx_id TEXT PRIMARY KEY,
        date TEXT,
        operation_type TEXT,
        counterparty TEXT,
        details TEXT,
        amount REAL,
        currency TEXT,
        record_type TEXT,
        confidence TEXT,
        reason TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS training_examples (
        example_id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id TEXT,
        entry_type TEXT,
        input_text TEXT,      
        input_json TEXT,      
        output_json TEXT,     
        created_at TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_te_entry ON training_examples(entry_id)")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS entries (
        entry_id TEXT PRIMARY KEY,
        entry_type TEXT,
        date TEXT,
        status TEXT,
        confidence REAL,
        reason TEXT,
        updated_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS journal_lines (
        line_id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id TEXT,
        entry_type TEXT,
        date TEXT,
        line_no INTEGER,
        dc TEXT,
        account TEXT,
        amount_eur REAL,
        source_currency TEXT,
        source_amount REAL,
        rate_source TEXT,
        rate_used REAL,
        memo TEXT,
        status TEXT,
        updated_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_type TEXT,
        entity_id TEXT,
        field TEXT,
        old_value TEXT,
        new_value TEXT,
        ts TEXT,
        user TEXT
    )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_jl_entry ON journal_lines(entry_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_entries_status ON entries(status)")

    conn.commit()
    conn.close()

def reset_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS audit_log")
    cur.execute("DROP TABLE IF EXISTS journal_lines")
    cur.execute("DROP TABLE IF EXISTS entries")
    cur.execute("DROP TABLE IF EXISTS training_examples")
    cur.execute("DROP TABLE IF EXISTS transactions")
    conn.commit()
    conn.close()
    init_db()
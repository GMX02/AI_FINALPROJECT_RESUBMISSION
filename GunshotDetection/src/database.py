# database.py (dummy sqlite interactions)

import sqlite3
import os

DB_PATH = 'queries.db'

def init_db():
    # initialize db and table if not exists
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY,
            file_name TEXT,
            detection_result TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def query_past_files():
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    return rows
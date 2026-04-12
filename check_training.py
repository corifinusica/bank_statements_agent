import sqlite3
from database import DB_PATH

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("PRAGMA table_info(training_examples)")
print(cur.fetchall())
conn.close()
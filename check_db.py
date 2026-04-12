import sqlite3

conn = sqlite3.connect("app.db")
cur = conn.cursor()

cur.execute("PRAGMA table_info(journal_lines)")
cols = cur.fetchall()

print("journal_lines columns:")
for c in cols:
    print(c)

conn.close()
import sqlite3
import pandas as pd

ENTRY_ID = "TX_050a497b_0001"

conn = sqlite3.connect("app.db")

e = pd.read_sql_query(
    "SELECT entry_id, status, updated_at FROM entries WHERE entry_id=?",
    conn, params=(ENTRY_ID,)
)
jl = pd.read_sql_query(
    "SELECT entry_id, status, COUNT(*) AS n_lines FROM journal_lines WHERE entry_id=? GROUP BY entry_id, status",
    conn, params=(ENTRY_ID,)
)

print("ENTRY:")
print(e)
print("\nJOURNAL_LINES statuses:")
print(jl)

conn.close()
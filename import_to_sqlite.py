# import_to_sqlite.py
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from database import init_db, get_conn, reset_db

def now():
    return datetime.utcnow().isoformat()

def import_csvs(data_dir: Path):
    init_db()
    conn = get_conn()
    cur = conn.cursor()


    cur.execute("DELETE FROM audit_log")
    cur.execute("DELETE FROM journal_lines")
    cur.execute("DELETE FROM entries")
    cur.execute("DELETE FROM transactions")
    conn.commit()

    # 1) transactions
    tx_path = data_dir / "transactions_clean.csv"
    if tx_path.exists():
        df_tx = pd.read_csv(tx_path)
        df_tx.columns = [c.strip() for c in df_tx.columns]
        df_tx.to_sql("transactions", conn, if_exists="append", index=False)
        print(f"Imported transactions: {len(df_tx)} rows")
    else:
        print("No transactions_clean.csv found (ok).")

    # 2) entries
    entries_path = data_dir / "entries_draft.csv"
    df_e = pd.read_csv(entries_path)
    df_e["updated_at"] = now()
    df_e.to_sql("entries", conn, if_exists="append", index=False)
    print(f"Imported entries: {len(df_e)} rows")

    # 3) journal lines
    jl_path = data_dir / "journal_lines_draft.csv"
    df_jl = pd.read_csv(jl_path)
    df_jl["updated_at"] = now()


    if "line_id" in df_jl.columns:
        df_jl = df_jl.drop(columns=["line_id"])

    df_jl.to_sql("journal_lines", conn, if_exists="append", index=False)
    print(f"Imported journal_lines: {len(df_jl)} rows")

    conn.close()
    print("Done. DB = app.db")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data", help="Folder with CSV files")
    p.add_argument("--reset", action="store_true", help="Drop & recreate DB tables")
    args = p.parse_args()

    if args.reset:
        reset_db()

    import_csvs(Path(args.data_dir))
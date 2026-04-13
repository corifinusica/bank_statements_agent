import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime, date, timedelta

import pandas as pd
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
from database import init_db, get_conn
from import_to_sqlite import import_csvs


init_db()


def has_data():
    conn = get_conn()
    try:
        return bool(conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
    finally:
        conn.close()

if not has_data():
    print("No entries found. Importing from data folder...")
    import_csvs(Path("data"))
    print("Import done.")

USER = "accountant"


# -------------------- helpers --------------------
def now() -> str:
    return datetime.utcnow().isoformat()


def get_openai_api_key():
    """
    API key lookup order:
    1) Local development: .streamlit/secrets.toml
    2) Render / Docker: environment variable OPENAI_API_KEY
    """
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


@st.cache_resource
def get_openai_client():
    api_key = get_openai_api_key()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def ask_llm(user_message: str, entry_id: str) -> str:
    """
    Minimal LLM response with context for the selected entry:
    - transaction (if TX_)
    - journal_lines
    - entry metadata
    - balance
    - top-3 similar approved examples (RAG-lite)
    """
    client = get_openai_client()
    if client is None:
        return "LLM is disabled: OPENAI_API_KEY not found (neither in secrets.toml nor in env)."

    conn = get_conn()
    try:
        e = pd.read_sql_query("SELECT * FROM entries WHERE entry_id=?", conn, params=(entry_id,))
        jl = pd.read_sql_query(
            "SELECT line_no, dc, account, amount_eur, memo, source_currency, source_amount "
            "FROM journal_lines WHERE entry_id=? ORDER BY line_no",
            conn,
            params=(entry_id,),
        )
        tx = pd.DataFrame()
        if entry_id.startswith("TX_"):
            tx_id = entry_id.replace("TX_", "")
            tx = pd.read_sql_query("SELECT * FROM transactions WHERE tx_id=?", conn, params=(tx_id,))
    finally:
        conn.close()

    # Balance
    if jl.empty:
        bal = {"debit": 0.0, "credit": 0.0, "diff": 0.0}
    else:
        debit = float(jl.loc[jl["dc"] == "D", "amount_eur"].sum())
        credit = float(jl.loc[jl["dc"] == "C", "amount_eur"].sum())
        bal = {"debit": round(debit, 2), "credit": round(credit, 2), "diff": round(debit - credit, 2)}

    # Similar cases (RAG-lite)
    sim = find_similar_examples(entry_id, k=3)
    sim_short = sim[["entry_id", "score"]].to_dict(orient="records") if not sim.empty else []

    context = {
        "entry_id": entry_id,
        "entry_meta": e.iloc[0].to_dict() if not e.empty else {},
        "transaction": tx.iloc[0].to_dict() if not tx.empty else {},
        "journal_lines": jl.to_dict(orient="records"),
        "balance": bal,
        "similar_examples": sim_short,
    }

    system = (
        "You are an accounting assistant. Reply in English.\n"
        "You MUST NOT modify the database. Only explain and suggest.\n"
        "Do not invent amounts. If unsure, say 'needs review'.\n"
        "You may propose changes to accounts and D/C, but DO NOT change amount_eur.\n"
        "If you propose journal lines, return JSON only: [{line_no, dc, account, memo}].\n"
        "At the end, add a 'Next actions' section with 2–4 bullets."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"CONTEXT:\n{json.dumps(context, ensure_ascii=False)}\n\nQUESTION:\n{user_message}",
            },
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def audit_cur(cur, entity_type, entity_id, field, old, new):
    """Write to audit_log in the SAME transaction (prevents DB locked errors)."""
    cur.execute(
        """INSERT INTO audit_log(entity_type, entity_id, field, old_value, new_value, ts, user)
           VALUES (?,?,?,?,?,?,?)""",
        (entity_type, str(entity_id), field, str(old), str(new), now(), USER),
    )


def load_entries(status_filter="all") -> pd.DataFrame:
    conn = get_conn()
    try:
        q = "SELECT * FROM entries"
        params = ()
        if status_filter != "all":
            q += " WHERE status = ?"
            params = (status_filter,)
        q += " ORDER BY date, entry_id"
        return pd.read_sql_query(q, conn, params=params)
    finally:
        conn.close()


def load_entry(entry_id: str) -> dict:
    conn = get_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM entries WHERE entry_id=?", conn, params=(entry_id,))
        return {} if df.empty else df.iloc[0].to_dict()
    finally:
        conn.close()


def load_journal_lines(entry_id: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        return pd.read_sql_query(
            "SELECT * FROM journal_lines WHERE entry_id=? ORDER BY line_no",
            conn,
            params=(entry_id,),
        )
    finally:
        conn.close()


def entry_balance_info(jl_df: pd.DataFrame):
    if jl_df.empty:
        return 0.0, 0.0, 0.0
    debit = float(jl_df.loc[jl_df["dc"] == "D", "amount_eur"].sum())
    credit = float(jl_df.loc[jl_df["dc"] == "C", "amount_eur"].sum())
    return round(debit, 2), round(credit, 2), round(debit - credit, 2)


def build_input_text(conn, entry_id: str) -> str:
    """
    Text for similarity search:
    - For TX_<tx_id>: use transaction fields
    - Otherwise: fallback to reason + memos
    """
    if entry_id.startswith("TX_"):
        tx_id = entry_id.replace("TX_", "")
        tx = pd.read_sql_query("SELECT * FROM transactions WHERE tx_id=?", conn, params=(tx_id,))
        if not tx.empty:
            r = tx.iloc[0].to_dict()
            return (
                f"{r.get('operation_type','')} | {r.get('counterparty','')} | {r.get('details','')} | "
                f"{r.get('currency','')} | {r.get('amount','')}"
            )

    e = pd.read_sql_query("SELECT reason, entry_type FROM entries WHERE entry_id=?", conn, params=(entry_id,))
    jl = pd.read_sql_query(
        "SELECT memo, account, dc FROM journal_lines WHERE entry_id=? ORDER BY line_no",
        conn,
        params=(entry_id,),
    )
    reason = e["reason"].iloc[0] if not e.empty else ""
    memos = " ".join(jl["memo"].astype(str).tolist()) if not jl.empty else ""
    return f"{reason} | {memos}"


def build_input_json(conn, entry_id: str) -> dict:
    """
    JSON context for training examples:
    - For TX_<tx_id>: return the transaction row
    - Otherwise: return entry metadata
    """
    if entry_id.startswith("TX_"):
        tx_id = entry_id.replace("TX_", "")
        tx = pd.read_sql_query("SELECT * FROM transactions WHERE tx_id=?", conn, params=(tx_id,))
        if not tx.empty:
            return tx.iloc[0].to_dict()

    e = pd.read_sql_query("SELECT * FROM entries WHERE entry_id=?", conn, params=(entry_id,))
    return {} if e.empty else e.iloc[0].to_dict()


def save_training_example_cur(cur, conn, entry_id: str):
    """
    Save an approved entry as a training example in the SAME transaction.
    Schema:
    training_examples(entry_id, entry_type, input_text, input_json, output_json, created_at)
    """
    e = pd.read_sql_query("SELECT entry_type FROM entries WHERE entry_id=?", conn, params=(entry_id,))
    jl = pd.read_sql_query(
        "SELECT line_no, dc, account, amount_eur, memo FROM journal_lines WHERE entry_id=? ORDER BY line_no",
        conn,
        params=(entry_id,),
    )

    entry_type = e["entry_type"].iloc[0] if not e.empty else ""
    input_text = build_input_text(conn, entry_id)
    input_json = json.dumps(build_input_json(conn, entry_id), ensure_ascii=False)
    output_json = json.dumps(jl.to_dict(orient="records"), ensure_ascii=False)

    cur.execute(
        """INSERT INTO training_examples(entry_id, entry_type, input_text, input_json, output_json, created_at)
           VALUES (?,?,?,?,?,?)""",
        (entry_id, entry_type, input_text, input_json, output_json, now()),
    )


def find_similar_examples(entry_id: str, k=3) -> pd.DataFrame:
    conn = get_conn()
    try:
        query_text = build_input_text(conn, entry_id)
        df_ex = pd.read_sql_query(
            "SELECT example_id, entry_id, entry_type, input_text, created_at FROM training_examples",
            conn,
        )
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

    if df_ex.empty:
        return pd.DataFrame()

    vect = TfidfVectorizer(min_df=1)
    X = vect.fit_transform(df_ex["input_text"].astype(str))
    q = vect.transform([query_text])
    sims = cosine_similarity(q, X).flatten()

    out = df_ex.copy()
    out["score"] = sims
    return out.sort_values("score", ascending=False).head(k)


def save_journal_lines(edited_df: pd.DataFrame, original_df: pd.DataFrame) -> int:
    """
    Save editable fields back to DB for each line_id.
    Editable: dc, account, amount_eur, memo
    """
    conn = get_conn()
    cur = conn.cursor()
    updated = 0

    try:
        for _, row in edited_df.iterrows():
            line_id = int(row["line_id"])
            orig = original_df[original_df["line_id"] == line_id].iloc[0]

            new_dc = str(row["dc"]).strip().upper()
            new_acc = str(row["account"]).strip()
            new_amt = float(row["amount_eur"])
            new_memo = "" if pd.isna(row["memo"]) else str(row["memo"])

            old_dc = str(orig["dc"])
            old_acc = str(orig["account"])
            old_amt = float(orig["amount_eur"])
            old_memo = "" if pd.isna(orig["memo"]) else str(orig["memo"])

            if new_dc != old_dc:
                cur.execute("UPDATE journal_lines SET dc=?, updated_at=? WHERE line_id=?", (new_dc, now(), line_id))
                audit_cur(cur, "journal_lines", line_id, "dc", old_dc, new_dc)
                updated += 1

            if new_acc != old_acc:
                cur.execute(
                    "UPDATE journal_lines SET account=?, updated_at=? WHERE line_id=?",
                    (new_acc, now(), line_id),
                )
                audit_cur(cur, "journal_lines", line_id, "account", old_acc, new_acc)
                updated += 1

            if round(new_amt, 2) != round(old_amt, 2):
                cur.execute(
                    "UPDATE journal_lines SET amount_eur=?, updated_at=? WHERE line_id=?",
                    (new_amt, now(), line_id),
                )
                audit_cur(cur, "journal_lines", line_id, "amount_eur", old_amt, new_amt)
                updated += 1

            if new_memo != old_memo:
                cur.execute("UPDATE journal_lines SET memo=?, updated_at=? WHERE line_id=?", (new_memo, now(), line_id))
                audit_cur(cur, "journal_lines", line_id, "memo", old_memo, new_memo)
                updated += 1

        conn.commit()
        return updated
    finally:
        conn.close()


def approve_entry(entry_id: str):
    """
    Approve an entry only if it is balanced.
    Updates statuses and writes a training example.
    """
    conn = get_conn()
    cur = conn.cursor()

    try:
        jl = pd.read_sql_query(
            "SELECT dc, amount_eur FROM journal_lines WHERE entry_id=?",
            conn,
            params=(entry_id,),
        )
        debit = round(float(jl.loc[jl["dc"] == "D", "amount_eur"].sum()), 2)
        credit = round(float(jl.loc[jl["dc"] == "C", "amount_eur"].sum()), 2)
        diff = round(debit - credit, 2)
        if diff != 0:
            return False, f"Entry not balanced: Debit={debit} Credit={credit} Diff={diff}"

        old = cur.execute("SELECT status FROM entries WHERE entry_id=?", (entry_id,)).fetchone()
        old_status = old["status"] if old else None

        cur.execute("UPDATE entries SET status=?, updated_at=? WHERE entry_id=?", ("approved", now(), entry_id))
        cur.execute("UPDATE journal_lines SET status=?, updated_at=? WHERE entry_id=?", ("approved", now(), entry_id))

        audit_cur(cur, "entries", entry_id, "status", old_status, "approved")

        # Store training example
        save_training_example_cur(cur, conn, entry_id)

        conn.commit()
        return True, "Approved"
    finally:
        conn.close()


def agent_reply(msg: str) -> str:
    """Fallback response (when LLM is disabled)."""
    s = msg.lower().strip()
    if "636" in s:
        return "636 = FX loss (negative impact). 536 = FX gain."
    if "536" in s:
        return "536 = FX gain (positive impact). 636 = FX loss."
    if "kurs" in s or "rate" in s:
        return "FX rates source: Bank of Lithuania SOAP getFxRates tp='LT' (official accounting rates)."
    if "approve" in s:
        return "To approve: select an entry on the left and click Approve (entry must be balanced)."
    return "I can explain journal entries, accounts, FX logic, and show similar approved cases."


# -------------------- chat tools --------------------
LB_URL = "https://www.lb.lt/webservices/FxRates/FxRates.asmx/getFxRates"


def export_approved_xlsx_bytes() -> bytes:
    conn = get_conn()
    try:
        approved = pd.read_sql_query(
            "SELECT entry_id, date, line_no, dc, account, amount_eur, memo "
            "FROM journal_lines WHERE status='approved' "
            "ORDER BY date, entry_id, line_no",
            conn,
        )
    finally:
        conn.close()

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        approved.to_excel(w, index=False, sheet_name="approved")
    return bio.getvalue()


def apply_top1_suggestion(entry_id: str, copy_dc: bool = False):
    sim = find_similar_examples(entry_id, k=5)
    if sim.empty:
        return False, "No training examples yet. Approve a few entries first."

    sim = sim[sim["entry_id"] != entry_id].copy()
    if sim.empty:
        return False, "Only the current entry is similar. Nothing to apply."

    top_entry_id = str(sim.iloc[0]["entry_id"])
    top_score = float(sim.iloc[0]["score"])

    conn = get_conn()
    cur = conn.cursor()
    try:
        ex = pd.read_sql_query(
            "SELECT output_json FROM training_examples WHERE entry_id=? ORDER BY created_at DESC LIMIT 1",
            conn,
            params=(top_entry_id,),
        )
        if ex.empty:
            return False, f"Could not find output_json for example {top_entry_id}"

        example_lines = json.loads(ex.iloc[0]["output_json"])

        jl = pd.read_sql_query(
            "SELECT line_id, line_no, dc, account FROM journal_lines WHERE entry_id=? ORDER BY line_no",
            conn,
            params=(entry_id,),
        )
        if jl.empty:
            return False, "The current entry has no journal_lines."

        ex_map = {}
        for ln in example_lines:
            try:
                ex_map[int(ln["line_no"])] = {
                    "account": str(ln["account"]).strip(),
                    "dc": str(ln.get("dc", "")).strip().upper(),
                }
            except Exception:
                continue

        updated = 0
        for _, r in jl.iterrows():
            line_no = int(r["line_no"])
            if line_no not in ex_map:
                continue

            new_acc = ex_map[line_no]["account"]
            old_acc = str(r["account"])
            if new_acc and new_acc != old_acc:
                cur.execute(
                    "UPDATE journal_lines SET account=?, updated_at=? WHERE line_id=?",
                    (new_acc, now(), int(r["line_id"])),
                )
                audit_cur(cur, "journal_lines", int(r["line_id"]), "account", old_acc, new_acc)
                updated += 1

            if copy_dc:
                new_dc = ex_map[line_no]["dc"]
                old_dc = str(r["dc"])
                if new_dc in ("D", "C") and new_dc != old_dc:
                    cur.execute(
                        "UPDATE journal_lines SET dc=?, updated_at=? WHERE line_id=?",
                        (new_dc, now(), int(r["line_id"])),
                    )
                    audit_cur(cur, "journal_lines", int(r["line_id"]), "dc", old_dc, new_dc)
                    updated += 1

        conn.commit()
        return True, f"Applied from {top_entry_id} (score={top_score:.3f}). Updates={updated}."
    finally:
        conn.close()


def fetch_lb_eur_to_gbp_tp_lt(d: date):
    # Backfill up to 10 days (weekends/holidays)
    for back in range(0, 10):
        dd = d - timedelta(days=back)
        url = f"{LB_URL}?tp=LT&dt={dd.strftime('%Y-%m-%d')}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml_text = resp.read()

        root = ET.fromstring(xml_text)

        def strip(tag):
            return tag.split("}")[-1]

        for fxrate in root.iter():
            if strip(fxrate.tag) != "FxRate":
                continue

            ccyamts = [x for x in fxrate if strip(x.tag) == "CcyAmt"]
            if len(ccyamts) < 2:
                continue

            def parse_ccyamt(node):
                ccy = None
                amt = None
                for ch in node:
                    if strip(ch.tag) == "Ccy":
                        ccy = ch.text
                    elif strip(ch.tag) == "Amt":
                        amt = float(ch.text)
                return ccy, amt

            c1, a1 = parse_ccyamt(ccyamts[0])  # EUR
            c2, a2 = parse_ccyamt(ccyamts[1])  # GBP / ...
            if c1 == "EUR" and c2 == "GBP":
                return a2, dd  # 1 EUR = a2 GBP

    raise ValueError("Cannot fetch EUR/GBP tp=LT near date")


def recalc_fx_entry(entry_id: str):
    conn = get_conn()
    cur = conn.cursor()
    try:
        e = pd.read_sql_query("SELECT date, entry_type FROM entries WHERE entry_id=?", conn, params=(entry_id,))
        if e.empty:
            return False, "Entry not found."
        if str(e.iloc[0]["entry_type"]) != "conversion":
            return False, "This is not a conversion entry."

        fx_date = pd.to_datetime(str(e.iloc[0]["date"]), errors="coerce").date()
        rate, used_date = fetch_lb_eur_to_gbp_tp_lt(fx_date)  # EUR/GBP

        jl = pd.read_sql_query(
            "SELECT line_id, line_no, dc, account, amount_eur, source_currency, source_amount "
            "FROM journal_lines WHERE entry_id=? ORDER BY line_no",
            conn,
            params=(entry_id,),
        )

        eur_line = jl[(jl["dc"] == "D") & (jl["account"].astype(str).str.startswith("2710"))]
        gbp_line = jl[
            (jl["dc"] == "C")
            & (jl["account"].astype(str).str.startswith("2715"))
            & (jl["source_currency"].astype(str).str.upper() == "GBP")
        ]

        if eur_line.empty or gbp_line.empty:
            return False, "Could not find lines 2710 (D) and 2715 (C, GBP) to recalculate."

        eur_in = float(eur_line.iloc[0]["amount_eur"])
        gbp_amount = float(gbp_line.iloc[0]["source_amount"])

        eur_equiv = abs(gbp_amount) / rate
        fx_diff = eur_equiv - eur_in  # >0 loss, <0 gain

        # Update 2715 amount_eur
        gbp_line_id = int(gbp_line.iloc[0]["line_id"])
        old_2715 = float(gbp_line.iloc[0]["amount_eur"])
        new_2715 = round(eur_equiv, 2)
        if round(old_2715, 2) != new_2715:
            cur.execute(
                "UPDATE journal_lines SET amount_eur=?, rate_source=?, rate_used=?, updated_at=? WHERE line_id=?",
                (new_2715, "LB_tp_LT", float(rate), now(), gbp_line_id),
            )
            audit_cur(cur, "journal_lines", gbp_line_id, "amount_eur", old_2715, new_2715)

        # Update/insert FX gain/loss line (636 loss, 536 gain)
        loss = jl[jl["account"].astype(str).str.startswith("636")]
        gain = jl[jl["account"].astype(str).str.startswith("536")]

        if fx_diff > 0:
            amt = round(fx_diff, 2)
            if not loss.empty:
                lid = int(loss.iloc[0]["line_id"])
                old = float(loss.iloc[0]["amount_eur"])
                cur.execute(
                    "UPDATE journal_lines SET dc='D', account='636', amount_eur=?, rate_source=?, rate_used=?, updated_at=? WHERE line_id=?",
                    (amt, "LB_tp_LT", float(rate), now(), lid),
                )
                audit_cur(cur, "journal_lines", lid, "amount_eur", old, amt)
            else:
                max_no = int(jl["line_no"].max())
                cur.execute(
                    """INSERT INTO journal_lines(entry_id, entry_type, date, line_no, dc, account,
                               amount_eur, source_currency, source_amount, rate_source, rate_used, memo, status, updated_at)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        entry_id,
                        "conversion",
                        fx_date.isoformat(),
                        max_no + 1,
                        "D",
                        "636",
                        amt,
                        "EUR",
                        amt,
                        "LB_tp_LT",
                        float(rate),
                        "FX loss (recalc)",
                        "draft",
                        now(),
                    ),
                )
        elif fx_diff < 0:
            amt = round(-fx_diff, 2)
            if not gain.empty:
                lid = int(gain.iloc[0]["line_id"])
                old = float(gain.iloc[0]["amount_eur"])
                cur.execute(
                    "UPDATE journal_lines SET dc='C', account='536', amount_eur=?, rate_source=?, rate_used=?, updated_at=? WHERE line_id=?",
                    (amt, "LB_tp_LT", float(rate), now(), lid),
                )
                audit_cur(cur, "journal_lines", lid, "amount_eur", old, amt)
            else:
                max_no = int(jl["line_no"].max())
                cur.execute(
                    """INSERT INTO journal_lines(entry_id, entry_type, date, line_no, dc, account,
                               amount_eur, source_currency, source_amount, rate_source, rate_used, memo, status, updated_at)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        entry_id,
                        "conversion",
                        fx_date.isoformat(),
                        max_no + 1,
                        "C",
                        "536",
                        amt,
                        "EUR",
                        amt,
                        "LB_tp_LT",
                        float(rate),
                        "FX gain (recalc)",
                        "draft",
                        now(),
                    ),
                )

        conn.commit()
        return True, f"FX recalculated. EUR/GBP={rate} (used_date={used_date})"
    finally:
        conn.close()


def handle_chat_command(cmd: str, entry_id: str):
    parts = cmd.strip().split()
    name = parts[0].lower()

    if name in ("/help", "/?"):
        return (
            "Commands:\n"
            "- /help\n"
            "- /show_similar\n"
            "- /apply_top1 [--with-dc]\n"
            "- /recalc_fx\n"
            "- /export_approved\n"
        )

    if name == "/show_similar":
        sim = find_similar_examples(entry_id, k=3)
        if sim.empty:
            return "No approved examples yet. Approve a few entries first."
        return "Top similar:\n" + "\n".join([f"{r.entry_id} score={r.score:.3f}" for r in sim.itertuples()])

    if name == "/apply_top1":
        copy_dc = any(p.lower() in ("--with-dc", "--dc") for p in parts[1:])
        ok, msg = apply_top1_suggestion(entry_id, copy_dc=copy_dc)
        return msg

    if name == "/recalc_fx":
        ok, msg = recalc_fx_entry(entry_id)
        return msg

    if name == "/export_approved":
        st.session_state["approved_xlsx_bytes"] = export_approved_xlsx_bytes()
        return "Done. The file is ready — the Download button will appear below."

    return "Unknown command. Type /help."


# -------------------- UI --------------------


st.title("Accounting Agent — Review + Chat")

with st.sidebar:
    status_filter = st.selectbox("Filter entries", ["all", "draft", "approved", "rejected"], index=0)

entries_df = load_entries(status_filter)
if entries_df.empty:
    st.warning("No entries in DB. Run: python import_to_sqlite.py --reset")
    st.stop()

left, mid, right = st.columns([1.25, 2.0, 1.0])

with left:
    st.subheader("Entries")
    show_cols = ["entry_id", "date", "entry_type", "status", "confidence"]
    for c in show_cols:
        if c not in entries_df.columns:
            entries_df[c] = None

    st.dataframe(entries_df[show_cols], width="stretch", height=520)
    selected_entry = st.selectbox("Select entry_id", entries_df["entry_id"].tolist())

with mid:
    st.subheader("Entry review")
    meta = load_entry(selected_entry)
    st.caption(f"entry_type={meta.get('entry_type')} | status={meta.get('status')}")

    jl = load_journal_lines(selected_entry)
    if jl.empty:
        st.info("No journal lines for this entry.")
    else:
        debit, credit, diff = entry_balance_info(jl)
        st.markdown(f"**Balance:** Debit={debit} | Credit={credit} | Diff={diff}")

        original = jl[["line_id", "line_no", "dc", "account", "amount_eur", "memo"]].copy()

        edited = st.data_editor(
            original,
            width="stretch",
            num_rows="fixed",
            column_config={
                "dc": st.column_config.SelectboxColumn(options=["D", "C"]),
                "account": st.column_config.TextColumn(),
                "amount_eur": st.column_config.NumberColumn(format="%.2f"),
                "memo": st.column_config.TextColumn(width="large"),
            },
            disabled=["line_id", "line_no"],
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save changes"):
                updated = save_journal_lines(edited, original)
                st.success(f"Saved. Fields updated: {updated}")
                st.rerun()

        with c2:
            if st.button("Approve"):
                ok, msg = approve_entry(selected_entry)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()

    st.divider()
    st.subheader("Similar approved examples (RAG-lite)")
    sim = find_similar_examples(selected_entry, k=3)
    if sim.empty:
        st.info("No training examples yet. Approve a few entries to start learning.")
    else:
        st.dataframe(sim[["score", "entry_id", "entry_type", "created_at"]], width="stretch", height=160)

    st.divider()
    st.subheader("Export approved journal lines")
    if st.button("Export approved to Excel"):
        st.session_state["approved_xlsx_bytes"] = export_approved_xlsx_bytes()

    if "approved_xlsx_bytes" in st.session_state:
        st.download_button(
            "Download",
            data=st.session_state["approved_xlsx_bytes"],
            file_name="journal_lines_approved.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

with right:
    st.subheader("Chat")
    st.caption("Tip: /help for commands. Plain text goes to the LLM.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, txt in st.session_state.chat:
        with st.chat_message(role):
            st.write(txt)

    msg = st.chat_input("Ask the agent…")
    if msg:
        st.session_state.chat.append(("user", msg))

        if msg.strip().startswith("/"):
            ans = handle_chat_command(msg, selected_entry)
        else:
            client = get_openai_client()
            ans = ask_llm(msg, selected_entry) if client is not None else agent_reply(msg)

        st.session_state.chat.append(("assistant", ans))
        st.rerun()
        st.session_state.chat.append(("assistant", ans))
        st.rerun()

    # Appears after /export_approved or the export button in the mid column
    if "approved_xlsx_bytes" in st.session_state:
        st.download_button(
            "Download approved Excel",
            data=st.session_state["approved_xlsx_bytes"],
            file_name="journal_lines_approved.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.divider()
    st.subheader("Audit log (last 30)")
    conn = get_conn()
    try:
        audit_df = pd.read_sql_query("SELECT * FROM audit_log ORDER BY ts DESC LIMIT 30", conn)
    finally:
        conn.close()

    st.dataframe(audit_df, width="stretch", height=260)
"""SEB Excel statement converter.

Converts one SEB Excel statement (possibly with multiple IBAN blocks)
into CSV files compatible with this project:
`transactions_clean.csv`, `entries_draft.csv`, `journal_lines_draft.csv`.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

COMPANY_BANK_ACCOUNT_EUR = "2710"
COMPANY_BANK_ACCOUNT_GBP = "2715"
DEFAULT_FX_DEBIT_LOSS = "636"


@dataclass(frozen=True)
class ParsedBlock:
    iban: Optional[str]
    header_row: int
    data_start: int
    data_end: int


def fold(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.upper().replace("\n", " ").strip()


def parse_amount(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace("\xa0", "").replace(" ", "")
    if not text:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_date(value: object) -> Optional[str]:
    """Parses Excel serial numbers and common date strings to ISO date."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    if re.fullmatch(r"\d{4,6}", text):
        try:
            serial = int(text)
            dt = datetime(1899, 12, 30) + timedelta(days=serial)
            return dt.date().isoformat()
        except ValueError:
            pass

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y.%m.%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue

    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


def is_marker_row(row_values: List[object]) -> bool:
    joined = fold(" ".join(str(x) for x in row_values))
    return "SASKAITOS" in joined and "ISRASAS" in joined


def is_header_row(row_values: List[object]) -> bool:
    joined = fold(" ".join(str(x) for x in row_values))
    return "DOK NR" in joined and "DATA" in joined and "SUMA" in joined


def extract_iban(row_values: List[object]) -> Optional[str]:
    joined = " ".join(str(x) for x in row_values)
    match = re.search(r"LT\d{18,20}", joined)
    return match.group(0) if match else None


def map_header(row_values: List[object]) -> Dict[str, int]:
    norm = [fold(x) for x in row_values]
    mapping: Dict[str, int] = {}
    for idx, name in enumerate(norm):
        if "DOK NR" in name:
            mapping["doc_no"] = idx
        elif name == "DATA":
            mapping["date"] = idx
        elif name == "VALIUTA":
            mapping["currency"] = idx
        elif name == "SUMA":
            mapping["amount"] = idx
        elif "PAVADINIMAS" in name and "KREDITO" not in name:
            mapping["counterparty"] = idx
        elif "IDENTIFIKACINIS" in name:
            mapping["counterparty_id"] = idx
        elif name == "SASKAITA":
            mapping["account_iban"] = idx
        elif "SWIFT" in name:
            mapping["swift"] = idx
        elif "PASKIRTIS" in name:
            mapping["details"] = idx
        elif "TRANSAKCIJOS KODAS" in name:
            mapping["tx_code"] = idx
        elif "DOKUMENTO DATA" in name:
            mapping["doc_date"] = idx
        elif "TRANSAKCIJOS TIPAS" in name:
            mapping["tx_type"] = idx
        elif "NUORODA" in name:
            mapping["reference"] = idx
        elif "DEBETAS" in name:
            mapping["dc"] = idx
        elif "SUMA SASKAITOS" in name:
            mapping["amount_acc"] = idx
        elif "SASKAITOS VALIUTA" in name:
            mapping["acc_currency"] = idx
    return mapping


def detect_blocks(raw: pd.DataFrame) -> List[ParsedBlock]:
    markers = [i for i in range(len(raw)) if is_marker_row(raw.iloc[i].tolist())]
    blocks: List[ParsedBlock] = []
    for idx, marker in enumerate(markers):
        iban = extract_iban(raw.iloc[marker].tolist())
        end_idx = markers[idx + 1] if idx + 1 < len(markers) else len(raw)
        header_idx = None
        for j in range(marker + 1, end_idx):
            if is_header_row(raw.iloc[j].tolist()):
                header_idx = j
                break
        if header_idx is None:
            continue
        blocks.append(
            ParsedBlock(
                iban=iban,
                header_row=header_idx,
                data_start=header_idx + 1,
                data_end=end_idx,
            )
        )
    return blocks


def classify_entry_type(tx_type: str, details: str) -> str:
    marker = f"{tx_type} {details}".upper()
    if "FORX" in marker or "KONVERT" in marker:
        return "conversion"
    return "single"


def infer_reason(tx_type: str, details: str, dc: str) -> str:
    text = f"{tx_type} {details}".upper()
    if "FEES" in text or "MOKESTIS" in text:
        return "Based on: Bank service fee EUR"
    if "PLAIS" in text or "REGISTRY" in text or "REGISTR" in text:
        return "Based on: Penalties"
    if "SODRA" in text or "SOCIALIN" in text:
        return "Based on: Social insurance"
    if "FORX" in text or "KONVERT" in text:
        return "Based on: Currency conversion"
    if dc == "C":
        return "Based on: Customer payment"
    return "Based on: Outgoing payment"


def default_accounts(currency: str, tx_type: str, details: str, dc: str) -> tuple[str, str, str]:
    marker = f"{tx_type} {details}".upper()
    bank_acc = COMPANY_BANK_ACCOUNT_EUR if currency == "EUR" else COMPANY_BANK_ACCOUNT_GBP

    if "FEES" in marker or "MOKESTIS" in marker:
        return ("61120", bank_acc, "Bank fee (draft)")
    if "PLAIS" in marker or "REGISTRY" in marker or "REGISTR" in marker:
        return ("635", bank_acc, "Registry/PLAIS fee (draft)")
    if "SODRA" in marker or "SOCIALIN" in marker:
        return ("4462", bank_acc, "Sodra (draft)")
    if "FORX" in marker or "KONVERT" in marker:
        return ("2710", "2715", "FX conversion (draft)")
    if dc == "C":
        return (bank_acc, "2410", "Customer payment (draft)")
    return ("4430", bank_acc, "Outgoing payment (draft)")


def _stable_prefix(input_xlsx_path: Path) -> str:
    digest = hashlib.sha1(str(input_xlsx_path.resolve()).encode("utf-8")).hexdigest()
    return digest[:8]


def _row_get(row: List[object], mapping: Dict[str, int], key: str) -> str:
    idx = mapping.get(key)
    if idx is None or idx >= len(row):
        return ""
    return str(row[idx])


def parse_raw_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    all_rows: List[dict] = []
    for block in detect_blocks(raw):
        mapping = map_header(raw.iloc[block.header_row].tolist())
        for i in range(block.data_start, block.data_end):
            row = raw.iloc[i].tolist()
            if not any(str(c).strip() for c in row):
                continue

            joined = fold(" ".join(str(c) for c in row))
            if "DOK NR" in joined:
                continue
            if "SASKAITOS" in joined and "ISRASAS" in joined:
                continue
            if "VISO" in joined or "PRADINIS LIKUTIS" in joined or "GALUTINIS LIKUTIS" in joined:
                continue

            amount = parse_amount(_row_get(row, mapping, "amount"))
            if amount is None:
                amount = parse_amount(_row_get(row, mapping, "amount_acc"))
            if amount is None:
                continue

            currency = (
                _row_get(row, mapping, "currency").strip()
                or _row_get(row, mapping, "acc_currency").strip()
                or "EUR"
            ).upper()
            dc_raw = _row_get(row, mapping, "dc").strip().upper()
            dc = "D" if dc_raw.startswith("D") else ("C" if dc_raw.startswith("C") else "")

            signed_amount = amount
            if dc == "D":
                signed_amount = -abs(amount)
            elif dc == "C":
                signed_amount = abs(amount)

            all_rows.append(
                {
                    "iban": block.iban or "",
                    "doc_no": _row_get(row, mapping, "doc_no").strip(),
                    "date": parse_date(_row_get(row, mapping, "date")),
                    "currency": currency,
                    "amount": signed_amount,
                    "counterparty": _row_get(row, mapping, "counterparty").strip(),
                    "counterparty_id": _row_get(row, mapping, "counterparty_id").strip(),
                    "account_iban": _row_get(row, mapping, "account_iban").strip(),
                    "swift": _row_get(row, mapping, "swift").strip(),
                    "details": " ".join(_row_get(row, mapping, "details").split()),
                    "tx_code": _row_get(row, mapping, "tx_code").strip(),
                    "doc_date": parse_date(_row_get(row, mapping, "doc_date")),
                    "tx_type": " ".join(_row_get(row, mapping, "tx_type").split()),
                    "reference": _row_get(row, mapping, "reference").strip(),
                    "dc": dc,
                }
            )

    return pd.DataFrame(all_rows)


def build_output_frames(parsed_df: pd.DataFrame, source_xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx_rows: List[dict] = []
    entry_rows: List[dict] = []
    jl_rows: List[dict] = []

    tx_prefix = _stable_prefix(source_xlsx_path)
    for idx, row in parsed_df.reset_index(drop=True).iterrows():
        tx_id = f"{tx_prefix}_{idx:04d}"
        date_iso = row.get("date") or datetime.utcnow().date().isoformat()
        operation_type = row.get("tx_type") or "SEB operation"
        details = row.get("details") or ""
        dc = row.get("dc") or ""
        amount = float(row.get("amount", 0.0))
        amount_abs = round(abs(amount), 2)
        currency = str(row.get("currency") or "EUR").upper()
        entry_type = classify_entry_type(str(operation_type), str(details))
        reason = infer_reason(str(operation_type), str(details), dc)
        debit_acc, credit_acc, memo = default_accounts(currency, str(operation_type), str(details), dc)

        tx_rows.append(
            {
                "tx_id": tx_id,
                "date": date_iso,
                "operation_type": operation_type,
                "counterparty": row.get("counterparty", ""),
                "details": details,
                "amount": amount,
                "currency": currency,
                "record_type": "transaction",
                "confidence": "high",
                "reason": reason,
            }
        )

        entry_id = f"TX_{tx_id}"
        entry_rows.append(
            {
                "entry_id": entry_id,
                "entry_type": entry_type,
                "date": date_iso,
                "status": "draft",
                "confidence": 0.7,
                "reason": reason,
            }
        )

        jl_rows.append(
            {
                "entry_id": entry_id,
                "entry_type": entry_type,
                "date": date_iso,
                "line_no": 1,
                "dc": "D",
                "account": debit_acc,
                "amount_eur": amount_abs,
                "source_currency": currency,
                "source_amount": amount,
                "rate_source": "",
                "rate_used": "",
                "memo": details or memo,
                "status": "draft",
            }
        )
        jl_rows.append(
            {
                "entry_id": entry_id,
                "entry_type": entry_type,
                "date": date_iso,
                "line_no": 2,
                "dc": "C",
                "account": credit_acc,
                "amount_eur": amount_abs,
                "source_currency": currency,
                "source_amount": amount,
                "rate_source": "",
                "rate_used": "",
                "memo": details or memo,
                "status": "draft",
            }
        )

        if entry_type == "conversion":
            fx_delta = round(abs(amount) * 0.02, 2)
            if fx_delta > 0:
                jl_rows.append(
                    {
                        "entry_id": entry_id,
                        "entry_type": entry_type,
                        "date": date_iso,
                        "line_no": 3,
                        "dc": "D",
                        "account": DEFAULT_FX_DEBIT_LOSS,
                        "amount_eur": fx_delta,
                        "source_currency": "EUR",
                        "source_amount": fx_delta,
                        "rate_source": "estimation",
                        "rate_used": "",
                        "memo": "FX loss (draft estimate)",
                        "status": "draft",
                    }
                )

    tx_df = pd.DataFrame(tx_rows)
    entries_df = pd.DataFrame(entry_rows)
    jl_df = pd.DataFrame(jl_rows)
    return tx_df, entries_df, jl_df


def convert_seb_excel(input_xlsx_path: str, output_dir: str = "data") -> Dict[str, object]:
    input_path = Path(input_xlsx_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(input_path, header=None, dtype=str).fillna("")
    parsed_df = parse_raw_transactions(raw)
    if parsed_df.empty:
        raise ValueError("No transactions were parsed from SEB Excel file.")

    tx_df, entries_df, jl_df = build_output_frames(parsed_df, input_path)

    tx_path = out_dir / "transactions_clean.csv"
    entries_path = out_dir / "entries_draft.csv"
    jl_path = out_dir / "journal_lines_draft.csv"

    tx_df.to_csv(tx_path, index=False, encoding="utf-8")
    entries_df.to_csv(entries_path, index=False, encoding="utf-8")
    jl_df.to_csv(jl_path, index=False, encoding="utf-8")

    return {
        "parsed_rows": len(parsed_df),
        "transactions": len(tx_df),
        "entries": len(entries_df),
        "journal_lines": len(jl_df),
        "output_dir": str(out_dir),
        "files": [str(tx_path), str(entries_path), str(jl_path)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SEB Excel statement to project CSV files.")
    parser.add_argument("--input", required=True, help="Path to SEB Excel statement (*.xlsx)")
    parser.add_argument("--output-dir", default="data", help="Target directory for generated CSV files")
    args = parser.parse_args()

    result = convert_seb_excel(args.input, output_dir=args.output_dir)
    print(f"Parsed rows: {result['parsed_rows']}")
    print(f"transactions_clean.csv: {result['transactions']} rows")
    print(f"entries_draft.csv: {result['entries']} rows")
    print(f"journal_lines_draft.csv: {result['journal_lines']} rows")
    print(f"Output directory: {result['output_dir']}")


if __name__ == "__main__":
    main()
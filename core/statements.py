"""
Bank-statement CSV import for real users.

    parsed = parse_statement(uploaded_bytes)            # raises InvalidInput with a readable message if it cannot be understood
    result = import_statement(db, user, parsed)         # writes the user's own transactions; needs the statement consent

What it handles
  * Preamble lines before the header, either one Amount + Dr/Cr column or separate Debit / Credit columns, Indian date formats
    (dd/mm/yy, dd-Mon-yyyy ...), "1,23,456.78" and "Rs", "Dr"/"Cr" suffixes, newest-first files, footer / summary rows.
  * Privacy minimisation: only the columns FinSight needs are kept, the file itself is never stored, and digit runs of 9+
    (account, card, phone, UPI reference numbers) are masked except the last 4 digits.
  * Duplicates: re-uploading the same statement (or an overlapping one) does not double-count anything.
  * First-pass categories from keywords. Debits go to `predicted_category` (confidence 0.8 for a keyword hit, 0.4 for
    a bare transfer) so the ML categoriser (`python -m ml.categorizer.predict`) can improve on them later. Credits must carry
    `category` for the income views, so they are labelled Salary / Pension / Other Income by keyword, or Salary when large relative to
    the declared monthly income.

Not handled: PDF or Excel statements (export CSV from net-banking), multiple accounts in one file, manual column mapping.
"""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Optional

from core import privacy
from core.auth import ConsentRequired, InvalidInput, User

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 20_000

# ------------------------------------------------------------------------------------------------ columns
_ALIASES = {
    "date": ["date", "txn date", "transaction date", "tran date", "trans date", "posting date", "value date", "value dt"],
    "desc": ["description", "narration", "transaction remarks", "particulars", "remarks", "transaction details",
             "transaction description", "details"],
    "debit": ["debit", "withdrawal", "withdrawal amt", "withdrawal amount", "withdrawal amount inr", "debit amount",
              "withdrawals", "amount debited", "dr amount", "paid out", "debit dr"],
    "credit": ["credit", "deposit", "deposit amt", "deposit amount", "deposit amount inr", "credit amount", "deposits",
               "amount credited", "cr amount", "paid in", "credit cr"],
    "amount": ["amount", "transaction amount", "amount inr", "txn amount"],
    "drcr": ["dr cr", "cr dr", "type", "transaction type", "debit credit"],
    "balance": ["balance", "closing balance", "balance inr", "running balance", "available balance", "bal"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def detect_columns(header: list[str]) -> dict[str, int]:
    names = [_norm(h) for h in header]
    found: dict[str, int] = {}
    for role, aliases in _ALIASES.items():
        for alias in aliases:                               # alias order = priority (transaction date before value date)
            if alias in names:
                found[role] = names.index(alias)
                break
    return found


def _usable(cols: dict[str, int]) -> bool:
    return "date" in cols and "desc" in cols and ("debit" in cols or "credit" in cols or "amount" in cols)


# ------------------------------------------------------------------------------------------------ values
_DATE_FORMATS = ("%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y", "%d-%b-%Y", "%d-%b-%y", "%d %b %Y", "%d %b %y",
                 "%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S",
                 "%Y-%m-%d %H:%M:%S", "%d %B %Y", "%d-%B-%Y")


def parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_amount(s: str) -> Optional[tuple[float, Optional[str]]]:
    """-> (signed value, 'dr'|'cr'|None) or None when the cell is empty / not a number."""
    s = (s or "").strip()
    if not s or s in {"-", "--", "–"}:
        return None
    s = re.sub(r"(?i)\brs\.?|\binr\b|₹", "", s).strip()
    hint = None
    m = re.search(r"(?i)\s*(dr|cr)\.?\s*$", s)
    if m:
        hint, s = m.group(1).lower(), s[:m.start()]
    neg = s.strip().startswith("-") or (s.strip().startswith("(") and s.strip().endswith(")"))
    s = s.strip().strip("()-+ ").replace(",", "").replace(" ", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return (-abs(v) if neg else v), hint


def mask_digits(text: str) -> str:
    """Keep the last 4 digits of any run of 9+ digits (account / card / phone / reference numbers)."""
    text = re.sub(r"\s+", " ", (text or "")).strip()[:200]
    return re.sub(r"\d{9,}", lambda m: "X" * (len(m.group()) - 4) + m.group()[-4:], text)


def infer_channel(desc: str) -> Optional[str]:
    d = desc.upper()
    for channel, pat in (("UPI", r"\bUPI\b"), ("ATM", r"\b(ATM|NWD|ATW)\b"), ("NEFT", r"\b(NEFT|RTGS|IMPS)\b"),
                         ("AutoDebit", r"\b(ACH|NACH|ECS|AUTO ?DEBIT|MANDATE)\b"), ("Card", r"\b(POS|CARD|ECOM)\b"),
                         ("NetBanking", r"\b(NETBANKING|INB|BILLPAY|BILLDESK)\b")):
        if re.search(pat, d):
            return channel
    return None


# ------------------------------------------------------------------------------------------------ categories
# Earlier rules win (so SWIGGY INSTAMART is Groceries before SWIGGY is Food Delivery). Names match dim_category.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Cash Withdrawal", ("ATM", "NWD", "ATW", "CASH WDL", "CASH WITHDRAWAL")),
    ("Investments", ("GROWW", "ZERODHA", "KUVERA", "MUTUAL FUND", "NIPPON", "PPFAS", "SMALLCASE", "ET MONEY", "SIP", "MF", "NPS", "ICCL")),
    ("EMI", ("EMI", "LOAN", "BAJAJ FINSERV", "TATA CAPITAL")),
    ("Insurance", ("LIC", "INSURANCE", "HDFC LIFE", "ICICI PRUDENTIAL", "STAR HEALTH", "NIVA BUPA", "BAJAJ ALLIANZ", "ACKO",
                   "POLICYBAZAAR", "PREMIUM")),
    ("Rent", ("RENT", "NOBROKER", "CRED RENT")),
    ("Groceries", ("SWIGGY INSTAMART", "INSTAMART", "BIGBASKET", "BLINKIT", "ZEPTO", "DMART", "AVENUE SUPERMARTS",
                   "RELIANCE FRESH", "RELIANCE SMART", "MORE SUPERMARKET", "SUPERMARKET", "KIRANA", "JIOMART",
                   "AMAZON FRESH", "SPENCERS", "GROCERY", "GROCERIES")),
    ("Subscriptions", ("NETFLIX", "SPOTIFY", "AMAZON PRIME", "PRIME VIDEO", "HOTSTAR", "DISNEY", "YOUTUBE", "SONYLIV", "ZEE5",
                       "APPLE SERVICES", "APPLE COM BILL", "GOOGLE ONE", "LINKEDIN", "SWIGGY ONE", "ZOMATO GOLD")),
    ("Food Delivery", ("SWIGGY", "ZOMATO", "EATSURE", "DOMINOS", "FAASOS", "MAGICPIN", "BEHROUZ", "MCDONALDS")),
    ("Dining Out", ("STARBUCKS", "CAFE COFFEE DAY", "CCD", "KFC", "PIZZA HUT", "HALDIRAMS", "BARBEQUE NATION", "TRUFFLES",
                    "CHAI POINT", "RESTAURANT", "CAFE", "BURGER KING", "DHABA", "BAKERY")),
    ("Transport", ("UBER", "OLA", "OLACABS", "RAPIDO", "METRO", "IRCTC", "REDBUS", "FASTAG", "BMTC", "KSRTC", "MSRTC")),
    ("Fuel", ("INDIAN OIL", "IOCL", "HPCL", "HP PETROL", "BPCL", "BHARAT PETROLEUM", "SHELL", "PETROL", "FUEL")),
    ("Utilities", ("BESCOM", "MSEDCL", "TATA POWER", "ELECTRICITY", "ADANI ELECTRICITY", "BWSSB", "MAHANAGAR GAS", "INDANE",
                   "MSEB", "TORRENT POWER", "GAS", "WATER")),
    ("Mobile & Internet", ("JIO", "AIRTEL", "VODAFONE", "BSNL", "ACT FIBERNET", "RECHARGE", "BROADBAND", "JIOFIBER")),
    ("Health", ("PHARMACY", "PHARMEASY", "APOLLO", "1MG", "MEDPLUS", "PRACTO", "CLINIC", "HOSPITAL", "PATH LABS", "PATHLABS",
                "DR LAL", "DIAGNOSTIC", "MEDICAL")),
    ("Education", ("SCHOOL", "COLLEGE", "UNIVERSITY", "TUITION", "COACHING", "UNACADEMY", "BYJUS", "BYJU S", "COURSERA",
                   "UDEMY", "UPGRAD")),
    ("Travel", ("MAKEMYTRIP", "GOIBIBO", "INDIGO", "AIR INDIA", "OYO", "AIRBNB", "YATRA", "CLEARTRIP", "VISTARA", "TAJ HOTELS",
                "HOTEL", "AKASA", "SPICEJET")),
    ("Entertainment", ("BOOKMYSHOW", "PVR", "INOX", "DREAM11", "STEAM", "PLAYSTORE", "INSIDER")),
    ("Personal Care", ("SALON", "BARBER", "URBAN COMPANY", "URBANCLAP", "CULT FIT", "CULTFIT", "NATURALS", "JAWED HABIB", "SPA", "GYM")),
    ("Gifts & Donations", ("DONATION", "TEMPLE", "CHARITY", "CRY INDIA", "GOONJ", "PM CARES", "FERNS N PETALS", "TANISHQ", "CHURCH")),
    ("Shopping", ("AMAZON", "FLIPKART", "MYNTRA", "AJIO", "NYKAA", "DECATHLON", "CROMA", "MEESHO", "WESTSIDE", "IKEA",
                  "LIFESTYLE", "RELIANCE DIGITAL", "PAYTM MALL")),
]
_COMPILED = [(cat, re.compile(r"\b(?:" + "|".join(re.escape(k).replace(r"\ ", r"[\s]+") for k in kws) + r")\b"))
             for cat, kws in _RULES]
_TRANSFER = re.compile(r"\b(UPI|IMPS|NEFT|RTGS|TRANSFER|TRF)\b")
_SALARY = re.compile(r"\b(SALARY|SAL|PAYROLL)\b")
_PENSION = re.compile(r"\bPENSION\b")


def categorize_debit(desc: str) -> tuple[Optional[str], Optional[float]]:
    d = re.sub(r"[^A-Z0-9&]+", " ", desc.upper())
    for cat, pat in _COMPILED:
        if pat.search(d):
            return cat, 0.8
    if _TRANSFER.search(d):
        return "Transfers", 0.4
    return None, None


def categorize_credit(desc: str, amount: float, monthly_income: float) -> str:
    d = re.sub(r"[^A-Z0-9&]+", " ", desc.upper())
    if _PENSION.search(d):
        return "Pension"
    if _SALARY.search(d):
        return "Salary"
    if re.search(r"\b(REFUND|REVERSAL|INTEREST|CASHBACK)\b", d):
        return "Other Income"
    if monthly_income > 0 and amount >= 0.6 * monthly_income:
        return "Salary"                                     # large credit comparable to the declared income
    return "Other Income"


# ------------------------------------------------------------------------------------------------ parsing
@dataclass
class StmtRow:
    date: date
    description: str
    type: str                       # 'debit' | 'credit'
    amount: float                   # always positive
    balance: Optional[float] = None


@dataclass
class ParsedStatement:
    rows: list[StmtRow]
    columns: dict[str, str]         # role -> header text found in the file
    has_balance: bool
    warnings: list[str] = field(default_factory=list)


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def parse_statement(data: bytes) -> ParsedStatement:
    if not data:
        raise InvalidInput("The file is empty.")
    if len(data) > MAX_BYTES:
        raise InvalidInput(f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB. Export a shorter period.")
    text = _decode(data)
    if "\x00" in text or text.lstrip().startswith("%PDF"):
        raise InvalidInput("This does not look like a CSV file. PDF and Excel statements are not supported: export CSV from net-banking.")
    head = "\n".join(text.splitlines()[:80])
    header_idx, cols, delimiter = None, {}, ","
    for delim in (",", ";", "\t", "|"):                        # first delimiter that yields a usable header row wins
        for i, row in enumerate(list(csv.reader(io.StringIO(head), delimiter=delim))[:60]):
            c = detect_columns(row)
            if _usable(c):
                header_idx, cols, delimiter = i, c, delim
                break
        if header_idx is not None:
            break
    table = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if header_idx is None:
        seen = next((r for r in table if sum(1 for x in r if x.strip()) >= 3), [])
        raise InvalidInput("Could not find the transaction table. We look for a date column, a description / narration column "
                           "and debit + credit (or amount) columns. First columns seen: " + ", ".join(x.strip() for x in seen[:8] if x.strip()))
    header = table[header_idx]
    has_split = "debit" in cols or "credit" in cols
    warnings: list[str] = []
    rows: list[StmtRow] = []
    skipped = 0
    sign_inferred = False
    cell = lambda r, role: r[cols[role]] if role in cols and cols[role] < len(r) else ""

    for r in table[header_idx + 1:]:
        if not any(x.strip() for x in r):
            continue
        d = parse_date(cell(r, "date"))
        if d is None:
            skipped += 1                                      # footer, totals, wrapped text lines
            continue
        desc = cell(r, "desc")
        typ, amt = None, None
        if has_split:
            dr, cr = parse_amount(cell(r, "debit")), parse_amount(cell(r, "credit"))
            if dr and abs(dr[0]) > 0:
                typ, amt = "debit", abs(dr[0])
            elif cr and abs(cr[0]) > 0:
                typ, amt = "credit", abs(cr[0])
        else:
            a = parse_amount(cell(r, "amount"))
            if a and a[0] != 0:
                flag = (cell(r, "drcr").strip().upper()[:1] if "drcr" in cols else "") or (a[1] or "").upper()[:1]
                if flag in ("D", "C"):
                    typ = "debit" if flag == "D" else "credit"
                else:
                    typ = "debit" if a[0] < 0 else "credit"
                    sign_inferred = True
                amt = abs(a[0])
        if typ is None:
            skipped += 1
            continue
        bal = None
        if "balance" in cols:
            b = parse_amount(cell(r, "balance"))
            if b:
                bal = -abs(b[0]) if b[1] == "dr" else b[0]
        if not (date(2000, 1, 1) <= d <= date.today() + timedelta(days=1)):
            skipped += 1
            continue
        rows.append(StmtRow(d, mask_digits(desc), typ, round(amt, 2), None if bal is None else round(bal, 2)))
        if len(rows) > MAX_ROWS:
            raise InvalidInput(f"More than {MAX_ROWS:,} transactions in one file. Upload a shorter period.")

    if not rows:
        raise InvalidInput("No transactions could be read from the file (check the date format and the amount columns).")
    if skipped > max(5, 0.2 * (len(rows) + skipped)):
        warnings.append(f"{skipped} rows could not be read and were skipped. Check the totals against your bank statement.")
    elif skipped:
        warnings.append(f"{skipped} non-transaction rows (headers, totals or footers) were skipped.")
    if sign_inferred:
        warnings.append("Only one amount column was found; we treated negative amounts as spending and positive as income.")

    if rows[0].date > rows[-1].date:                            # newest-first export
        rows.reverse()
    rows.sort(key=lambda r: r.date)                             # stable: keeps the file order within a day
    has_balance = "balance" in cols and all(r.balance is not None for r in rows)
    if "balance" in cols and not has_balance:
        warnings.append("Some rows have no balance, so balances will be recomputed from the opening balance.")
    return ParsedStatement(rows, {k: header[v].strip() for k, v in cols.items()}, has_balance, warnings)


# ------------------------------------------------------------------------------------------------ import
@dataclass
class ImportResult:
    inserted: int
    duplicates: int
    first_date: Optional[date]
    last_date: Optional[date]
    uncategorized: int
    by_category: dict[str, int]
    warnings: list[str]


_INSERT = ("INSERT INTO transactions (txn_id, customer_id, txn_date, txn_ts, amount, type, raw_description, merchant, category, "
           "true_category, predicted_category, confidence, channel, balance_after) VALUES (:txn_id, :customer_id, :txn_date, "
           ":txn_ts, :amount, :type, :raw_description, NULL, :category, NULL, :predicted_category, :confidence, :channel, :balance_after)")


def preview_categories(parsed: ParsedStatement, monthly_income: float = 0.0) -> dict[str, int]:
    """Category counts the import would assign (for the confirmation screen)."""
    out: Counter = Counter()
    for r in parsed.rows:
        cat = categorize_credit(r.description, r.amount, monthly_income) if r.type == "credit" else (categorize_debit(r.description)[0] or "Uncategorized")
        out[cat] += 1
    return dict(out.most_common())


def import_statement(db, user: User, parsed: ParsedStatement, *, opening_balance: Optional[float] = None,
                     monthly_income: float = 0.0) -> ImportResult:
    if user.is_demo:
        raise InvalidInput("The shared demo account is read-only.")
    if not privacy.owns_customer(user.customer_id):
        raise InvalidInput("Statements can only be imported into accounts created through sign-up.")
    if not privacy.has_consent(db, user.user_id, privacy.PURPOSE_STATEMENTS):
        raise ConsentRequired("Please give your consent for processing your statement before importing it.")
    cid = user.customer_id
    first, last = parsed.rows[0].date, parsed.rows[-1].date
    warnings = list(parsed.warnings)

    with db.transaction(customer_id=cid) as q:
        if getattr(db, "dialect", "") == "postgresql":
            q("LOCK TABLE transactions IN EXCLUSIVE MODE")           # txn_id has no sequence: serialise id allocation
        next_id = int(q("SELECT COALESCE(MAX(txn_id), 0) AS m FROM transactions")[0]["m"]) + 1
        prior = int(q("SELECT COUNT(*) AS n FROM transactions WHERE customer_id = :c", {"c": cid})[0]["n"])
        existing = Counter()
        for e in q("SELECT txn_date, amount, type, raw_description FROM transactions "
                   "WHERE customer_id = :c AND txn_date >= :a AND txn_date <= :b", {"c": cid, "a": first, "b": last}):
            existing[(str(e["txn_date"])[:10], round(float(e["amount"]), 2), e["type"], e["raw_description"])] += 1

        if parsed.has_balance:
            running = None
        elif prior:
            last_bal = q("SELECT balance_after FROM transactions WHERE customer_id = :c ORDER BY txn_ts DESC, txn_id DESC LIMIT 1", {"c": cid})
            running = float(last_bal[0]["balance_after"] or 0) if last_bal else 0.0
            warnings.append("No balance column: balances continue from your previous statement.")
        else:
            running = float(opening_balance or 0.0)
            warnings.append("No balance column: balances were computed from the opening balance you entered (0 if left blank).")

        records, dup, per_day, by_cat, uncategorized = [], 0, Counter(), Counter(), 0
        for r in parsed.rows:
            key = (r.date.isoformat(), r.amount, r.type, r.description)
            if existing[key] > 0:
                existing[key] -= 1
                dup += 1
                continue
            if r.type == "credit":
                category, predicted, conf = categorize_credit(r.description, r.amount, monthly_income), None, None
                label = category
            else:
                category = None
                predicted, conf = categorize_debit(r.description)
                label = predicted or "Uncategorized"
            if label == "Uncategorized":
                uncategorized += 1
            by_cat[label] += 1
            if running is not None:
                running = round(running + (r.amount if r.type == "credit" else -r.amount), 2)
            bal = r.balance if parsed.has_balance else running
            k = per_day[r.date]
            per_day[r.date] += 1
            records.append({"txn_id": next_id, "customer_id": cid, "txn_date": r.date,
                            "txn_ts": datetime.combine(r.date, time(0, 0)) + timedelta(seconds=k),
                            "amount": r.amount, "type": r.type, "raw_description": r.description, "category": category,
                            "predicted_category": predicted, "confidence": conf, "channel": infer_channel(r.description),
                            "balance_after": bal})
            next_id += 1
        if records:
            q(_INSERT, records)                                      # executemany
            if prior == 0:                                           # first import: set the account's opening balance
                f = records[0]
                signed = f["amount"] if f["type"] == "credit" else -f["amount"]
                ob = (f["balance_after"] - signed) if parsed.has_balance else float(opening_balance or 0.0)
                q("UPDATE accounts SET opening_balance = :ob WHERE customer_id = :c", {"ob": round(ob, 2), "c": cid})
    if dup:
        warnings.append(f"{dup} transactions were already in your account and were skipped.")
    return ImportResult(len(records), dup, first, last, uncategorized, dict(by_cat.most_common()), warnings)

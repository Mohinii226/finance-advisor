"""Statement CSV parsing and import (no PostgreSQL needed: SQLite stand-in, see tests/auth_helpers.py)."""
from datetime import date

import pytest

from core import privacy
from core.auth import ConsentRequired, InvalidInput
from core.statements import (MAX_BYTES, categorize_credit, categorize_debit, import_statement, infer_channel, mask_digits,
                             parse_amount, parse_date, parse_statement, preview_categories)
from tests.auth_helpers import make_auth, reg

KNOWN_CATEGORIES = {
    "Rent", "Groceries", "Food Delivery", "Dining Out", "Transport", "Fuel", "Utilities", "Mobile & Internet", "Shopping",
    "Entertainment", "Health", "Education", "Travel", "Personal Care", "Subscriptions", "Insurance", "Gifts & Donations",
    "Transfers", "Cash Withdrawal", "EMI", "Investments", "Self Transfer", "Salary", "Freelance Income", "Pension",
    "Allowance", "Other Income"}

HDFC_LIKE = b"""Statement of account,,,,,,
Account No: 50100123456789,,,,,,
,,,,,,
Date,Narration,Chq./Ref.No.,Value Dt,Withdrawal Amt.,Deposit Amt.,Closing Balance
01/10/25,UPI-SWIGGY-9876543210123-PAYMENT,0000123,01/10/25,"1,250.00",,"48,750.00"
01/10/25,UPI-ZOMATO-5555555555555,0000124,01/10/25,450.00,,"48,300.00"
02/10/25,SALARY CREDIT INFOSYS LTD,,02/10/25,,"85,000.00","1,33,300.00"
03/10/25,ATM WDL 123456789012 PUNE,,03/10/25,"5,000.00",,"1,28,300.00"
05/10/25,NEFT-FRIEND RAHUL,,05/10/25,"2,000.00",,"1,26,300.00"
STATEMENT SUMMARY,,,,,,
Opening Balance,Debits,Credits,Closing Balance,,,
"""

STATEMENT_ROWS = 5


def _ready(income=0.0, consent=True, **kw):
    svc, db = make_auth(**kw)
    u = reg(svc, "asha@example.com", income=income)
    if consent:
        privacy.record_consent(db, u.user_id, privacy.PURPOSE_STATEMENTS)
    return svc, db, u


# ---------------------------------------------------------------- values
@pytest.mark.parametrize("text,expected", [
    ("1,23,456.78", (123456.78, None)), ("(500.00)", (-500.0, None)), ("Rs. 1,000 Dr", (1000.0, "dr")),
    ("2,500.50Cr", (2500.5, "cr")), ("₹2,500.50", (2500.5, None)), ("-1,250", (-1250.0, None)), ("0.00", (0.0, None)),
    ("-", None), ("", None), ("abc", None), (None, None)])
def test_parse_amount(text, expected):
    assert parse_amount(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("01/10/25", date(2025, 10, 1)), ("01/10/2025", date(2025, 10, 1)), ("1-Oct-2025", date(2025, 10, 1)),
    ("01 Oct 25", date(2025, 10, 1)), ("2025-10-01", date(2025, 10, 1)), ("31.12.2025", date(2025, 12, 31)),
    ("01/10/2025 14:30:00", date(2025, 10, 1)), ("13/13/2025", None), ("Total", None), ("", None)])
def test_parse_date_is_day_first(text, expected):
    assert parse_date(text) == expected


def test_mask_digits_hides_long_numbers_but_keeps_last_four():
    assert mask_digits("UPI-9876543210123-PAY") == "UPI-XXXXXXXXX0123-PAY"
    assert mask_digits("ATM 123456789012") == "ATM XXXXXXXX9012"
    assert mask_digits("RENT 12345678 MARCH") == "RENT 12345678 MARCH"        # 8 digits: left alone
    assert mask_digits("  lots   of    space ") == "lots of space"
    assert len(mask_digits("x" * 500)) == 200


def test_channel_inference():
    assert infer_channel("UPI-SWIGGY") == "UPI" and infer_channel("ATM WDL PUNE") == "ATM"
    assert infer_channel("NEFT-RAHUL") == "NEFT" and infer_channel("ACH D- LIC") == "AutoDebit"
    assert infer_channel("POS 4321 CROMA") == "Card" and infer_channel("just words") is None


@pytest.mark.parametrize("desc,expected", [
    ("UPI-SWIGGY-12345@ybl", "Food Delivery"), ("SWIGGY INSTAMART ORDER", "Groceries"), ("UPI-ZOMATO", "Food Delivery"),
    ("AMAZON PRIME MEMBERSHIP", "Subscriptions"), ("AMAZON PAY INDIA", "Shopping"), ("NETFLIX.COM", "Subscriptions"),
    ("ATM WDL PUNE", "Cash Withdrawal"), ("NWD-123-HDFC", "Cash Withdrawal"), ("BESCOM ELECTRICITY BILL", "Utilities"),
    ("AIRTEL PREPAID RECHARGE", "Mobile & Internet"), ("INDIAN OIL PETROL PUMP", "Fuel"), ("UBER INDIA", "Transport"),
    ("APOLLO PHARMACY", "Health"), ("LIC PREMIUM", "Insurance"), ("HOUSE RENT MARCH", "Rent"),
    ("GROWW SIP", "Investments"), ("HDFC BANK LOAN EMI", "EMI"), ("MAKEMYTRIP", "Travel"), ("BOOKMYSHOW", "Entertainment"),
    ("TEMPLE DONATION", "Gifts & Donations"), ("UPI-RAHUL KUMAR-9999@ybl", "Transfers"), ("NEFT-FRIEND", "Transfers"),
    ("PAYATM WALLET TOPUP", None), ("MYSTERY SHOP", None)])
def test_debit_rules(desc, expected):
    cat, conf = categorize_debit(desc)
    assert cat == expected
    assert (conf is None) == (expected is None)
    if expected:
        assert cat in KNOWN_CATEGORIES


def test_word_boundaries_prevent_false_hits():
    assert categorize_debit("CURRENT ACCOUNT FEE")[0] is None             # "RENT" inside CURRENT
    assert categorize_debit("PAYATM")[0] is None                          # "ATM" inside PAYATM
    assert categorize_debit("OLAFOOD STORE")[0] is None                   # "OLA" is not the whole word


def test_credit_rules():
    assert categorize_credit("SALARY OCT", 10, 0) == "Salary"
    assert categorize_credit("NEFT-INFOSYS-SAL", 10, 0) == "Salary"
    assert categorize_credit("PENSION CREDIT", 10, 0) == "Pension"
    assert categorize_credit("INTEREST PAID", 99999, 50000) == "Other Income"
    assert categorize_credit("UPI-RAHUL", 40000, 50000) == "Salary"       # large vs declared income
    assert categorize_credit("UPI-RAHUL", 400, 50000) == "Other Income"
    assert categorize_credit("UPI-RAHUL", 40000, 0) == "Other Income"     # no declared income: do not guess


# ---------------------------------------------------------------- parsing
def test_parse_hdfc_like_statement():
    p = parse_statement(HDFC_LIKE)
    assert len(p.rows) == STATEMENT_ROWS and p.has_balance
    assert p.columns["date"] == "Date" and p.columns["desc"] == "Narration" and p.columns["debit"] == "Withdrawal Amt."
    first = p.rows[0]
    assert (first.date, first.type, first.amount, first.balance) == (date(2025, 10, 1), "debit", 1250.0, 48750.0)
    assert "XXXXXXXXX0123" in first.description and "9876543210123" not in first.description
    assert p.rows[2].type == "credit" and p.rows[2].amount == 85000.0 and p.rows[2].balance == 133300.0
    assert any("non-transaction rows" in w for w in p.warnings)           # summary lines were skipped, not parsed


def test_single_amount_column_with_dr_cr_marker():
    data = b"Txn Date,Description,Amount,Dr/Cr,Balance\n01-Oct-2025,UPI SWIGGY,450.00,DR,9550.00\n02-Oct-2025,SALARY,50000.00,CR,59550.00\n"
    p = parse_statement(data)
    assert [(r.type, r.amount) for r in p.rows] == [("debit", 450.0), ("credit", 50000.0)]


def test_signed_amounts_without_a_marker():
    data = b"Date,Description,Amount\n2025-10-01,UPI SWIGGY,-450.00\n2025-10-02,REFUND,120.00\n"
    p = parse_statement(data)
    assert [(r.type, r.amount) for r in p.rows] == [("debit", 450.0), ("credit", 120.0)]
    assert not p.has_balance and any("negative amounts" in w for w in p.warnings)


def test_semicolon_and_tab_delimited_files():
    semi = b"Transaction Date;Transaction Remarks;Withdrawal Amount (INR);Deposit Amount (INR);Balance (INR)\n01/10/2025;UPI ZOMATO;300.00;0.00;700.00\n"
    assert parse_statement(semi).rows[0].amount == 300.0
    tab = b"Date\tNarration\tDebit\tCredit\n01/10/25\tUPI ZOMATO\t300\t\n"
    assert parse_statement(tab).rows[0].type == "debit"


def test_newest_first_files_are_put_in_date_order():
    data = b"Date,Narration,Debit,Credit,Balance\n03/10/25,C,10,,70\n02/10/25,B,10,,80\n01/10/25,A,10,,90\n"
    assert [r.description for r in parse_statement(data).rows] == ["A", "B", "C"]


def test_cp1252_bom_and_rupee_symbol_files():
    data = "\ufeffDate,Narration,Debit,Credit\n01/10/25,Caf\u00e9 \u20b9 shop,\u20b9 300.00,\n".encode("utf-8")
    assert parse_statement(data).rows[0].amount == 300.0
    assert parse_statement("Date,Narration,Debit,Credit\n01/10/25,Caf\xe9,300.00,\n".encode("cp1252")).rows[0].amount == 300.0


@pytest.mark.parametrize("data,fragment", [
    (b"", "empty"), (b"%PDF-1.4 binary", "does not look like a CSV"), (b"hello\nworld\n", "Could not find the transaction table"),
    (b"Date,Narration,Debit,Credit\nTotal,xx,,\n", "No transactions could be read"),
    (b"Date,Narration,Debit,Credit\n01/10/25,A,0,0\n", "No transactions could be read"),
    (b"Date,Narration,Debit,Credit\n01/10/1980,A,10,\n", "No transactions could be read"),
    (b"Date,Narration,Debit,Credit\n01/10/2999,A,10,\n", "No transactions could be read")])
def test_unreadable_files_give_a_clear_error(data, fragment):
    with pytest.raises(InvalidInput) as e:
        parse_statement(data)
    assert fragment in str(e.value)


def test_unrecognised_header_lists_the_columns_it_saw():
    with pytest.raises(InvalidInput) as e:
        parse_statement(b"When,What,How much\n01/10/25,A,10\n")
    assert "When" in str(e.value)


def test_size_limits():
    with pytest.raises(InvalidInput):
        parse_statement(b"x" * (MAX_BYTES + 1))


def test_many_unreadable_rows_produce_a_warning():
    rows = b"".join(b"01/10/25,A,10,\n" for _ in range(3)) + b"".join(b"bad,B,10,\n" for _ in range(10))
    p = parse_statement(b"Date,Narration,Debit,Credit\n" + rows)
    assert any("could not be read" in w for w in p.warnings)


# ---------------------------------------------------------------- import
def test_import_requires_consent_and_a_signup_account():
    svc, db, u = _ready(consent=False)
    with pytest.raises(ConsentRequired):
        import_statement(db, u, parse_statement(HDFC_LIKE))
    demo = make_auth(demo_customer_id="C0001")[0].demo_user()
    with pytest.raises(InvalidInput):
        import_statement(db, demo, parse_statement(HDFC_LIKE))
    linked = svc.admin_create_user("x@example.com", "another long passphrase", "X", customer_id="C0002")
    with pytest.raises(InvalidInput):
        import_statement(db, linked, parse_statement(HDFC_LIKE))
    assert db.scalar("SELECT COUNT(*) FROM transactions") == 0


def test_import_writes_what_the_views_expect():
    svc, db, u = _ready(income=85000)
    db.conn.execute("INSERT INTO transactions (txn_id, customer_id, txn_date, txn_ts, amount, type, raw_description) "
                    "VALUES (100, 'C0001', '2025-01-01', '2025-01-01 00:00:00', 5, 'debit', 'old synthetic row')")
    res = import_statement(db, u, parse_statement(HDFC_LIKE), monthly_income=85000)
    assert (res.inserted, res.duplicates) == (STATEMENT_ROWS, 0) and res.first_date == date(2025, 10, 1)
    rows = db.conn.execute("SELECT * FROM transactions WHERE customer_id = ? ORDER BY txn_id", (u.customer_id,)).fetchall()
    assert [r["txn_id"] for r in rows] == [101, 102, 103, 104, 105]                      # continues after the existing maximum
    swiggy, zomato, salary, atm, neft = rows
    assert (swiggy["type"], swiggy["category"], swiggy["predicted_category"], swiggy["confidence"]) == ("debit", None, "Food Delivery", 0.8)
    assert swiggy["channel"] == "UPI" and swiggy["merchant"] is None and swiggy["true_category"] is None
    assert (salary["type"], salary["category"], salary["predicted_category"]) == ("credit", "Salary", None)   # income views read `category`
    assert atm["predicted_category"] == "Cash Withdrawal" and neft["predicted_category"] == "Transfers"
    assert [r["balance_after"] for r in rows] == [48750, 48300, 133300, 128300, 126300]
    assert swiggy["txn_ts"] < zomato["txn_ts"]                                          # same day: file order preserved
    assert db.scalar("SELECT opening_balance FROM accounts WHERE customer_id = ?", (u.customer_id,)) == 50000
    assert res.by_category == {"Food Delivery": 2, "Salary": 1, "Cash Withdrawal": 1, "Transfers": 1} and res.uncategorized == 0
    assert {r["category"] or r["predicted_category"] for r in rows} <= KNOWN_CATEGORIES
    assert "9876543210123" not in str([tuple(r) for r in rows])                          # masked before it reached the database


def test_reimporting_or_overlapping_statements_never_double_count():
    svc, db, u = _ready()
    first = import_statement(db, u, parse_statement(HDFC_LIKE))
    again = import_statement(db, u, parse_statement(HDFC_LIKE))
    assert (again.inserted, again.duplicates) == (0, first.inserted)
    assert any("already in your account" in w for w in again.warnings)
    extra = HDFC_LIKE.replace(b"STATEMENT SUMMARY", b"07/10/25,UPI-UBER-1,,07/10/25,200.00,,\"1,26,100.00\"\nSTATEMENT SUMMARY")
    overlap = import_statement(db, u, parse_statement(extra))
    assert (overlap.inserted, overlap.duplicates) == (1, STATEMENT_ROWS)
    assert db.scalar("SELECT COUNT(*) FROM transactions WHERE customer_id = ?", (u.customer_id,)) == STATEMENT_ROWS + 1


def test_two_identical_rows_on_one_day_are_both_kept_once():
    twin = b"Date,Narration,Debit,Credit,Balance\n01/10/25,UPI CHAI POINT,50,,950\n01/10/25,UPI CHAI POINT,50,,900\n"
    svc, db, u = _ready()
    assert import_statement(db, u, parse_statement(twin)).inserted == 2
    again = import_statement(db, u, parse_statement(twin))
    assert (again.inserted, again.duplicates) == (0, 2)


def test_without_a_balance_column_balances_are_computed_and_continue_across_uploads():
    first = b"Date,Narration,Debit,Credit\n01/10/25,UPI ZOMATO,300,\n02/10/25,SALARY,,50000\n"
    second = b"Date,Narration,Debit,Credit\n03/10/25,UPI UBER,200,\n"
    svc, db, u = _ready()
    r = import_statement(db, u, parse_statement(first), opening_balance=1000)
    assert any("opening balance" in w for w in r.warnings)
    assert import_statement(db, u, parse_statement(second), opening_balance=999999).inserted == 1     # opening ignored: continues
    balances = [r["balance_after"] for r in db.conn.execute("SELECT balance_after FROM transactions ORDER BY txn_id")]
    assert balances == [700, 50700, 50500]
    assert db.scalar("SELECT opening_balance FROM accounts WHERE customer_id = ?", (u.customer_id,)) == 1000


def test_uncategorised_rows_are_counted_and_stay_null():
    data = b"Date,Narration,Debit,Credit,Balance\n01/10/25,MYSTERY SHOP,10,,90\n"
    svc, db, u = _ready()
    res = import_statement(db, u, parse_statement(data))
    assert res.uncategorized == 1 and res.by_category == {"Uncategorized": 1}
    row = db.conn.execute("SELECT category, predicted_category, confidence FROM transactions").fetchone()
    assert tuple(row) == (None, None, None)


def test_preview_matches_what_import_assigns():
    p = parse_statement(HDFC_LIKE)
    assert preview_categories(p, 85000) == {"Food Delivery": 2, "Salary": 1, "Cash Withdrawal": 1, "Transfers": 1}


def test_users_cannot_touch_each_others_transactions_on_import():
    svc, db, a = _ready()
    b = reg(svc, "bob@example.com", "Bob")
    privacy.record_consent(db, b.user_id, privacy.PURPOSE_STATEMENTS)
    import_statement(db, a, parse_statement(HDFC_LIKE))
    res = import_statement(db, b, parse_statement(HDFC_LIKE))          # same file, other user: nothing is a "duplicate"
    assert res.inserted == STATEMENT_ROWS and res.duplicates == 0
    for u in (a, b):
        assert db.scalar("SELECT COUNT(*) FROM transactions WHERE customer_id = ?", (u.customer_id,)) == STATEMENT_ROWS

"""
Static configuration for the synthetic data generator.

Everything that is "domain knowledge" lives here (categories, merchants,
festivals, persona archetypes) so the generator logic stays clean and you can
tweak the simulated world without touching code.
"""
from __future__ import annotations

from datetime import date

# --------------------------------------------------------------------------
# Time window
# --------------------------------------------------------------------------
START_DATE = date(2023, 1, 1)
N_MONTHS = 36                       # Jan-2023 .. Dec-2025
ANNUAL_INFLATION = 0.06             # applied monthly to all spending
MIN_HISTORY_MONTHS = 6              # injected events never start before this

# --------------------------------------------------------------------------
# Categories
#   weight   : baseline share of "discretionary+living" budget (normalised per customer)
#   freq     : average transactions per month
#   sigma    : lognormal spread of ticket size
#   kind     : variable | fixed | quarterly
#   essential: used later by the savings planner (true = hard to cut)
#   floor    : minimum ticket (Rs)
# --------------------------------------------------------------------------
CATEGORIES = {
    "Rent":              dict(weight=0.20,  freq=1.0,  sigma=0.00, kind="fixed",     essential=True,  floor=3000),
    "Groceries":         dict(weight=0.10,  freq=8.0,  sigma=0.55, kind="variable",  essential=True,  floor=40),
    "Food Delivery":     dict(weight=0.04,  freq=6.0,  sigma=0.40, kind="variable",  essential=False, floor=120),
    "Dining Out":        dict(weight=0.04,  freq=3.0,  sigma=0.60, kind="variable",  essential=False, floor=150),
    "Transport":         dict(weight=0.03,  freq=12.0, sigma=0.70, kind="variable",  essential=True,  floor=20),
    "Fuel":              dict(weight=0.03,  freq=3.0,  sigma=0.35, kind="variable",  essential=True,  floor=200),
    "Utilities":         dict(weight=0.04,  freq=3.0,  sigma=0.30, kind="variable",  essential=True,  floor=100),
    "Mobile & Internet": dict(weight=0.02,  freq=2.0,  sigma=0.15, kind="fixed",     essential=True,  floor=99),
    "Shopping":          dict(weight=0.06,  freq=4.0,  sigma=0.80, kind="variable",  essential=False, floor=150),
    "Entertainment":     dict(weight=0.015, freq=2.0,  sigma=0.50, kind="variable",  essential=False, floor=100),
    "Health":            dict(weight=0.02,  freq=1.5,  sigma=0.90, kind="variable",  essential=True,  floor=80),
    "Education":         dict(weight=0.02,  freq=1.0,  sigma=0.20, kind="fixed",     essential=True,  floor=500),
    "Travel":            dict(weight=0.03,  freq=0.8,  sigma=1.00, kind="variable",  essential=False, floor=400),
    "Personal Care":     dict(weight=0.012, freq=1.5,  sigma=0.50, kind="variable",  essential=False, floor=100),
    "Subscriptions":     dict(weight=0.012, freq=2.5,  sigma=0.00, kind="fixed",     essential=False, floor=59),
    "Insurance":         dict(weight=0.03,  freq=1.0,  sigma=0.10, kind="quarterly", essential=True,  floor=1000),
    "Gifts & Donations": dict(weight=0.01,  freq=1.0,  sigma=0.70, kind="variable",  essential=False, floor=100),
    "Transfers":         dict(weight=0.03,  freq=4.0,  sigma=0.90, kind="variable",  essential=False, floor=50),
    "Cash Withdrawal":   dict(weight=0.02,  freq=2.0,  sigma=0.40, kind="variable",  essential=False, floor=500),
}
# Categories generated outside the weight system
SPECIAL_CATEGORIES = ["EMI", "Investments", "Salary", "Freelance Income", "Pension", "Allowance",
                      "Other Income", "Self Transfer"]
INCOME_CATEGORY = {"salaried": "Salary", "govt": "Salary", "pension": "Pension",
                   "freelance": "Freelance Income", "student": "Allowance"}
ALL_CATEGORIES = list(CATEGORIES) + SPECIAL_CATEGORIES
DISCRETIONARY = [c for c, v in CATEGORIES.items() if not v["essential"]]

# --------------------------------------------------------------------------
# Merchants: (canonical name, relative popularity)
#   Deliberately ambiguous cases to make categorisation non-trivial:
#   Swiggy vs Swiggy Instamart, Amazon vs Amazon Prime, Uber vs Uber Eats-like, etc.
# --------------------------------------------------------------------------
MERCHANTS = {
    "Groceries": [("BIGBASKET", 3), ("BLINKIT", 4), ("ZEPTO", 3), ("SWIGGY INSTAMART", 3), ("DMART", 4),
                  ("RELIANCE FRESH", 2), ("MORE SUPERMARKET", 1), ("JIOMART", 1), ("LOCAL KIRANA", 3),
                  ("AMAZON FRESH", 1), ("SPENCERS", 1)],
    "Food Delivery": [("SWIGGY", 6), ("ZOMATO", 6), ("EATSURE", 1), ("DOMINOS PIZZA", 2), ("MCDONALDS", 1),
                      ("BEHROUZ BIRYANI", 1), ("FAASOS", 1), ("MAGICPIN", 1)],
    "Dining Out": [("BARBEQUE NATION", 1), ("PIZZA HUT", 1), ("STARBUCKS", 2), ("CAFE COFFEE DAY", 2),
                   ("KFC", 1), ("HALDIRAMS", 1), ("TRUFFLES", 1), ("LOCAL RESTAURANT", 4), ("CHAI POINT", 2),
                   ("BURGER KING", 1)],
    "Transport": [("UBER", 5), ("OLA CABS", 4), ("RAPIDO", 4), ("NAMMA METRO", 1), ("DELHI METRO", 1),
                  ("IRCTC", 2), ("REDBUS", 1), ("MUMBAI METRO", 1), ("FASTAG", 2)],
    "Fuel": [("INDIAN OIL", 4), ("HP PETROL PUMP", 3), ("BHARAT PETROLEUM", 3), ("SHELL", 1), ("RELIANCE PETROL", 1)],
    "Utilities": [("BESCOM", 1), ("MSEDCL", 1), ("TATA POWER", 1), ("ADANI ELECTRICITY", 1), ("BWSSB", 1),
                  ("MAHANAGAR GAS", 1), ("INDANE GAS", 1), ("BILLDESK ELECTRICITY", 2)],
    "Mobile & Internet": [("JIO PREPAID", 3), ("AIRTEL", 3), ("VI RECHARGE", 1), ("ACT FIBERNET", 1),
                          ("JIOFIBER", 1), ("AIRTEL XSTREAM", 1), ("BSNL", 1)],
    "Shopping": [("AMAZON", 6), ("FLIPKART", 5), ("MYNTRA", 3), ("AJIO", 2), ("NYKAA", 1), ("DECATHLON", 1),
                 ("CROMA", 1), ("RELIANCE DIGITAL", 1), ("LIFESTYLE STORES", 1), ("MEESHO", 2),
                 ("WESTSIDE", 1), ("IKEA", 1)],
    "Entertainment": [("BOOKMYSHOW", 4), ("PVR CINEMAS", 3), ("INOX", 2), ("DREAM11", 1), ("STEAM GAMES", 1),
                      ("PLAYSTORE", 2), ("PAYTM INSIDER", 1)],
    "Health": [("APOLLO PHARMACY", 4), ("PHARMEASY", 3), ("1MG", 2), ("MEDPLUS", 2), ("PRACTO", 1),
               ("LOCAL CLINIC", 2), ("DR LAL PATHLABS", 1), ("MANIPAL HOSPITAL", 1), ("FORTIS HOSPITAL", 1)],
    "Education": [("UNACADEMY", 1), ("BYJUS", 1), ("COURSERA", 1), ("UDEMY", 2), ("SCHOOL FEES", 3),
                  ("COACHING CLASSES", 2), ("TUITION FEE", 2), ("UPGRAD", 1)],
    "Travel": [("MAKEMYTRIP", 4), ("GOIBIBO", 2), ("INDIGO AIRLINES", 3), ("AIR INDIA", 2), ("OYO ROOMS", 2),
               ("AIRBNB", 1), ("YATRA", 1), ("CLEARTRIP", 1), ("VISTARA", 1), ("TAJ HOTELS", 1)],
    "Personal Care": [("LAKME SALON", 1), ("NATURALS SALON", 2), ("URBAN COMPANY", 3), ("GREEN TRENDS", 1),
                      ("JAWED HABIB", 1), ("LOCAL BARBER", 2), ("CULT FIT", 2)],
    "Subscriptions": [("NETFLIX", 4), ("SPOTIFY", 3), ("AMAZON PRIME", 4), ("HOTSTAR", 3), ("YOUTUBE PREMIUM", 2),
                      ("SONYLIV", 1), ("APPLE SERVICES", 2), ("GOOGLE ONE", 2), ("LINKEDIN PREMIUM", 1),
                      ("ZEE5", 1), ("SWIGGY ONE", 1), ("ZOMATO GOLD", 1)],
    "Insurance": [("LIC PREMIUM", 4), ("HDFC LIFE", 2), ("ICICI PRUDENTIAL", 2), ("STAR HEALTH", 2),
                  ("NIVA BUPA", 1), ("BAJAJ ALLIANZ", 1), ("ACKO", 1), ("POLICYBAZAAR", 1)],
    "Gifts & Donations": [("TEMPLE DONATION", 2), ("CRY INDIA", 1), ("GOONJ", 1), ("FERNS N PETALS", 2),
                          ("IGP GIFTS", 1), ("CHURCH OFFERING", 1), ("TANISHQ", 1), ("PM CARES", 1)],
    "Rent": [("HOUSE RENT", 3), ("NOBROKER RENT", 2), ("PG RENT", 1), ("CRED RENT PAY", 2)],
}

INVESTMENT_MERCHANTS = ["GROWW SIP", "ZERODHA COIN", "KUVERA MF", "HDFC MUTUAL FUND", "SBI MUTUAL FUND",
                        "NIPPON INDIA MF", "PPFAS MF", "ET MONEY SIP", "NPS TIER1"]
LENDERS = ["HDFC BANK LOAN", "BAJAJ FINSERV EMI", "SBI HOME LOAN", "ICICI BANK CAR LOAN",
           "KOTAK PERSONAL LOAN", "AXIS BANK LOAN", "TATA CAPITAL EMI"]
COMPANIES = ["INFOSYS LTD", "TCS LTD", "WIPRO LTD", "HCL TECH", "ACCENTURE SOLUTIONS", "CAPGEMINI INDIA",
             "COGNIZANT TECH", "RELIANCE INDUSTRIES", "HDFC BANK LTD", "DELOITTE USHI", "ZOHO CORP",
             "TECH MAHINDRA", "PERSISTENT SYSTEMS", "BAJAJ AUTO LTD", "AMAZON DEV CENTRE"]
BANKS = ["HDFC", "SBI", "ICICI", "AXIS", "KOTAK"]
UPI_HANDLES = ["okhdfcbank", "oksbi", "okicici", "okaxis", "ybl", "paytm", "ibl", "axl"]

# Channel mix per category: UPI, Card, NetBanking, AutoDebit, ATM
CHANNEL_MIX = {
    "default":           dict(UPI=0.65, Card=0.25, NetBanking=0.10),
    "Rent":              dict(UPI=0.60, NetBanking=0.40),
    "Groceries":         dict(UPI=0.80, Card=0.20),
    "Food Delivery":     dict(UPI=0.60, Card=0.40),
    "Transport":         dict(UPI=0.85, Card=0.15),
    "Fuel":              dict(UPI=0.50, Card=0.50),
    "Utilities":         dict(UPI=0.50, NetBanking=0.50),
    "Mobile & Internet": dict(UPI=0.50, AutoDebit=0.30, Card=0.20),
    "Shopping":          dict(UPI=0.30, Card=0.60, NetBanking=0.10),
    "Travel":            dict(Card=0.60, UPI=0.20, NetBanking=0.20),
    "Subscriptions":     dict(Card=0.60, AutoDebit=0.40),
    "Insurance":         dict(AutoDebit=0.60, NetBanking=0.40),
    "Education":         dict(NetBanking=0.50, UPI=0.50),
    "Cash Withdrawal":   dict(ATM=1.0),
    "Transfers":         dict(UPI=1.0),
}

# --------------------------------------------------------------------------
# Weekday profiles (Mon..Sun). Default is flat.
# --------------------------------------------------------------------------
WEEKDAY_WEIGHTS = {
    "default":         [1, 1, 1, 1, 1, 1, 1],
    "Food Delivery":   [0.9, 0.8, 0.9, 1.0, 1.4, 1.5, 1.4],
    "Dining Out":      [0.5, 0.5, 0.6, 0.8, 1.6, 2.2, 2.0],
    "Entertainment":   [0.4, 0.4, 0.5, 0.7, 1.5, 2.3, 2.0],
    "Shopping":        [0.7, 0.7, 0.8, 0.9, 1.2, 1.8, 1.7],
    "Groceries":       [0.7, 0.7, 0.8, 0.9, 1.0, 1.8, 1.8],
    "Transport":       [1.4, 1.4, 1.4, 1.4, 1.4, 0.7, 0.6],
    "Fuel":            [1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 1.2],
    "Travel":          [0.6, 0.6, 0.7, 0.9, 1.8, 1.6, 1.0],
    "Personal Care":   [0.5, 0.6, 0.7, 0.8, 1.2, 2.0, 1.8],
    "Health":          [1.2, 1.2, 1.1, 1.1, 1.1, 0.8, 0.5],
}
# Categories where transactions cluster in the evening
EVENING_CATEGORIES = {"Food Delivery", "Dining Out", "Entertainment", "Shopping"}

# --------------------------------------------------------------------------
# Festivals: date, name, per-category budget boost for that month (+0.8 = +80%)
# Day-level weights are also raised in a +/-7 day window around the date.
# --------------------------------------------------------------------------
FESTIVALS = [
    # 2023
    (date(2023, 3, 8),  "Holi",      {"Shopping": 0.25, "Groceries": 0.20, "Dining Out": 0.25, "Gifts & Donations": 0.4}),
    (date(2023, 4, 22), "Eid",       {"Shopping": 0.50, "Groceries": 0.30, "Dining Out": 0.30, "Gifts & Donations": 0.8}),
    (date(2023, 11, 12), "Diwali",   {"Shopping": 0.90, "Groceries": 0.35, "Dining Out": 0.40, "Gifts & Donations": 1.2, "Travel": 0.5}),
    (date(2023, 12, 25), "Christmas", {"Shopping": 0.40, "Dining Out": 0.50, "Gifts & Donations": 0.6, "Travel": 0.6}),
    # 2024
    (date(2024, 3, 25), "Holi",      {"Shopping": 0.25, "Groceries": 0.20, "Dining Out": 0.25, "Gifts & Donations": 0.4}),
    (date(2024, 4, 10), "Eid",       {"Shopping": 0.50, "Groceries": 0.30, "Dining Out": 0.30, "Gifts & Donations": 0.8}),
    (date(2024, 11, 1), "Diwali",    {"Shopping": 0.90, "Groceries": 0.35, "Dining Out": 0.40, "Gifts & Donations": 1.2, "Travel": 0.5}),
    (date(2024, 12, 25), "Christmas", {"Shopping": 0.40, "Dining Out": 0.50, "Gifts & Donations": 0.6, "Travel": 0.6}),
    # 2025
    (date(2025, 3, 14), "Holi",      {"Shopping": 0.25, "Groceries": 0.20, "Dining Out": 0.25, "Gifts & Donations": 0.4}),
    (date(2025, 3, 31), "Eid",       {"Shopping": 0.50, "Groceries": 0.30, "Dining Out": 0.30, "Gifts & Donations": 0.8}),
    (date(2025, 10, 20), "Diwali",   {"Shopping": 0.90, "Groceries": 0.35, "Dining Out": 0.40, "Gifts & Donations": 1.2, "Travel": 0.5}),
    (date(2025, 12, 25), "Christmas", {"Shopping": 0.40, "Dining Out": 0.50, "Gifts & Donations": 0.6, "Travel": 0.6}),
]

# Month-of-year multipliers (seasonality independent of festivals)
SEASONAL = {
    "Travel":    {5: 1.5, 6: 1.4, 12: 1.3, 1: 1.1},
    "Utilities": {4: 1.3, 5: 1.5, 6: 1.4, 7: 1.1, 12: 0.85, 1: 0.85, 2: 0.9},
    "Health":    {7: 1.15, 8: 1.20, 9: 1.15},
    "Shopping":  {1: 1.10, 6: 1.10, 11: 1.10},          # sale seasons
    "Gifts & Donations": {11: 1.2, 12: 1.2, 1: 1.1, 2: 1.1},   # wedding season
    "Food Delivery": {7: 1.15, 8: 1.15},                # monsoon
}

# --------------------------------------------------------------------------
# Geography
# --------------------------------------------------------------------------
CITIES = {
    1: ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad"],
    2: ["Jaipur", "Lucknow", "Indore", "Nagpur", "Coimbatore", "Kochi", "Chandigarh", "Nashik", "Surat"],
    3: ["Mysuru", "Raipur", "Guwahati", "Dehradun", "Belagavi", "Kolhapur", "Udaipur"],
}
RENT_TIER_MULT = {1: 1.30, 2: 1.00, 3: 0.75}

# --------------------------------------------------------------------------
# Persona archetypes
#   income         : (low, high) monthly take-home, sampled log-uniformly
#   savings_rate   : (low, high) intended savings as share of income
#   sip            : (low, high) share of income invested monthly
#   loan_prob      : probability of an EMI
#   mult           : category weight multipliers vs the baseline
#   volatility     : month-to-month noise (std of log multiplier)
# --------------------------------------------------------------------------
ARCHETYPES = {
    "young_techie": dict(
        prob=0.22, age=(23, 32), occupation=["Software Engineer", "Data Analyst", "QA Engineer", "Product Designer"],
        income_type="salaried", income=(45000, 180000), dependents=(0, 0), savings_rate=(0.10, 0.30),
        sip=(0.05, 0.18), loan_prob=0.25, owns_home=0.0, vehicle=0.45, volatility=(0.08, 0.18),
        mult={"Food Delivery": 2.2, "Dining Out": 1.6, "Subscriptions": 1.6, "Shopping": 1.4, "Entertainment": 1.5,
              "Groceries": 0.6, "Health": 0.6, "Insurance": 0.5, "Education": 0.3, "Gifts & Donations": 0.7}),
    "family_single_earner": dict(
        prob=0.18, age=(33, 48), occupation=["Manager", "Accountant", "Sales Executive", "Teacher", "Engineer"],
        income_type="salaried", income=(40000, 130000), dependents=(2, 4), savings_rate=(0.03, 0.15),
        sip=(0.02, 0.08), loan_prob=0.70, owns_home=0.35, vehicle=0.75, volatility=(0.07, 0.14),
        mult={"Groceries": 1.8, "Education": 3.0, "Health": 1.6, "Insurance": 1.8, "Dining Out": 0.6,
              "Food Delivery": 0.5, "Utilities": 1.4, "Entertainment": 0.7, "Gifts & Donations": 1.4}),
    "frugal_saver": dict(
        prob=0.14, age=(26, 55), occupation=["Clerk", "Engineer", "Teacher", "Accountant", "Technician"],
        income_type="salaried", income=(30000, 90000), dependents=(0, 2), savings_rate=(0.30, 0.50),
        sip=(0.10, 0.25), loan_prob=0.20, owns_home=0.35, vehicle=0.40, volatility=(0.04, 0.09),
        mult={"Food Delivery": 0.3, "Dining Out": 0.4, "Shopping": 0.5, "Entertainment": 0.4, "Travel": 0.5,
              "Subscriptions": 0.6, "Groceries": 1.1, "Personal Care": 0.6}),
    "student_parttime": dict(
        prob=0.09, age=(18, 25), occupation=["Student", "Intern", "Part-time Tutor"],
        income_type="student", income=(8000, 25000), dependents=(0, 0), savings_rate=(0.00, 0.10),
        sip=(0.00, 0.03), loan_prob=0.02, owns_home=0.0, vehicle=0.15, volatility=(0.15, 0.30),
        mult={"Rent": 0.5, "Food Delivery": 1.8, "Dining Out": 1.4, "Entertainment": 1.5, "Shopping": 1.0,
              "Education": 1.5, "Insurance": 0.0, "Utilities": 0.4, "Health": 0.5, "Transfers": 1.5}),
    "freelancer": dict(
        prob=0.12, age=(24, 45), occupation=["Freelance Developer", "Designer", "Consultant", "Content Creator", "Photographer"],
        income_type="freelance", income=(30000, 150000), dependents=(0, 2), savings_rate=(0.05, 0.25),
        sip=(0.02, 0.10), loan_prob=0.20, owns_home=0.10, vehicle=0.35, volatility=(0.15, 0.30),
        mult={"Subscriptions": 2.0, "Mobile & Internet": 1.5, "Dining Out": 1.2, "Travel": 1.4}),
    "high_earner_spender": dict(
        prob=0.10, age=(28, 50), occupation=["Senior Manager", "Doctor", "Architect", "Director", "Consultant"],
        income_type="salaried", income=(150000, 400000), dependents=(0, 3), savings_rate=(0.10, 0.30),
        sip=(0.08, 0.20), loan_prob=0.60, owns_home=0.40, vehicle=0.95, volatility=(0.10, 0.22),
        mult={"Travel": 3.0, "Shopping": 2.5, "Dining Out": 2.8, "Entertainment": 2.0, "Personal Care": 2.5,
              "Subscriptions": 1.8, "Rent": 0.8, "Groceries": 0.8, "Gifts & Donations": 1.8, "Insurance": 1.5}),
    "govt_stable": dict(
        prob=0.09, age=(30, 58), occupation=["Government Clerk", "Bank Officer", "Railway Employee", "School Teacher", "Police Constable"],
        income_type="govt", income=(35000, 110000), dependents=(1, 4), savings_rate=(0.15, 0.35),
        sip=(0.03, 0.12), loan_prob=0.55, owns_home=0.50, vehicle=0.55, volatility=(0.03, 0.08),
        mult={"Food Delivery": 0.4, "Insurance": 2.0, "Gifts & Donations": 1.5, "Health": 1.2, "Education": 1.5}),
    "retiree": dict(
        prob=0.06, age=(58, 75), occupation=["Retired", "Retired Govt Servant", "Retired Bank Manager"],
        income_type="pension", income=(20000, 70000), dependents=(0, 1), savings_rate=(0.10, 0.30),
        sip=(0.00, 0.05), loan_prob=0.03, owns_home=0.85, vehicle=0.25, volatility=(0.04, 0.10),
        mult={"Health": 4.0, "Insurance": 2.0, "Food Delivery": 0.2, "Dining Out": 0.3, "Shopping": 0.4,
              "Gifts & Donations": 2.0, "Travel": 0.8, "Entertainment": 0.4, "Fuel": 0.5, "Transport": 0.4}),
}

# --------------------------------------------------------------------------
# Injected ground-truth events (used later to evaluate detectors)
# --------------------------------------------------------------------------
EVENT_PROBS = {
    "category_creep":     0.30,   # persistent overspend that ramps up
    "category_spike":     0.30,   # short 1-2 month burst
    "job_loss":           0.06,
    "income_raise":       0.15,
    "medical_emergency":  0.10,
}
CREEP_CATEGORIES = ["Food Delivery", "Dining Out", "Shopping", "Entertainment", "Travel"]
SPIKE_CATEGORIES = ["Shopping", "Travel", "Gifts & Donations", "Entertainment", "Dining Out", "Personal Care"]

# --------------------------------------------------------------------------
# Label noise (simulates the "uncategorised transactions" problem)
# --------------------------------------------------------------------------
BLANK_LABEL_RATE = 0.08       # category left empty (the ones your model must classify)
WRONG_LABEL_RATE = 0.03       # user / bank mislabelled the transaction

# Typical monthly prices (Rs) for subscriptions
SUBSCRIPTION_PRICES = {"NETFLIX": 649, "SPOTIFY": 119, "AMAZON PRIME": 299, "HOTSTAR": 299, "YOUTUBE PREMIUM": 149,
                       "SONYLIV": 299, "APPLE SERVICES": 199, "GOOGLE ONE": 130, "LINKEDIN PREMIUM": 1699,
                       "ZEE5": 99, "SWIGGY ONE": 149, "ZOMATO GOLD": 149}

# Label confusions that real users / banks make (used for WRONG_LABEL_RATE)
CONFUSABLE = {
    "Food Delivery": ["Dining Out", "Groceries"], "Dining Out": ["Food Delivery", "Entertainment"],
    "Groceries": ["Shopping", "Food Delivery"], "Shopping": ["Groceries", "Personal Care"],
    "Entertainment": ["Subscriptions", "Dining Out"], "Subscriptions": ["Entertainment", "Shopping"],
    "Transport": ["Travel", "Fuel"], "Travel": ["Transport", "Entertainment"], "Fuel": ["Transport"],
    "Health": ["Personal Care", "Insurance"], "Personal Care": ["Health", "Shopping"],
    "Utilities": ["Mobile & Internet"], "Mobile & Internet": ["Utilities", "Subscriptions"],
    "Transfers": ["Gifts & Donations", "Cash Withdrawal"], "Gifts & Donations": ["Transfers", "Shopping"],
}

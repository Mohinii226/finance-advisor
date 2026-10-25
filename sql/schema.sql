-- =====================================================================
-- Intelligent Personal Finance Advisor : PostgreSQL schema
-- Runs automatically on first `docker compose up` (see docker-compose.yml)
-- =====================================================================

-- Reference table (populated by data_gen/load_to_postgres.py from config.py,
-- so categories are defined in exactly one place)
CREATE TABLE IF NOT EXISTS dim_category (
    category      TEXT PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('expense','income','investment','debt','transfer')),
    is_essential  BOOLEAN NOT NULL DEFAULT FALSE,
    is_expense    BOOLEAN NOT NULL DEFAULT FALSE,   -- counts toward "spending" in analytics
    is_primary_income BOOLEAN NOT NULL DEFAULT FALSE -- salary/freelance/pension/allowance (excludes refunds, family top-ups)
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id     VARCHAR(10) PRIMARY KEY,
    name            TEXT NOT NULL,
    age             SMALLINT,
    city            TEXT,
    city_tier       SMALLINT CHECK (city_tier BETWEEN 1 AND 3),
    occupation      TEXT,
    income_type     TEXT CHECK (income_type IN ('salaried','govt','freelance','student','pension')),
    monthly_income  NUMERIC(12,2),
    dependents      SMALLINT,
    has_loan        BOOLEAN,
    owns_home       BOOLEAN,
    has_vehicle     BOOLEAN,
    persona         TEXT                    -- generator ground truth; do NOT use as an ML feature
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id       VARCHAR(10) PRIMARY KEY,
    customer_id      VARCHAR(10) NOT NULL REFERENCES customers(customer_id),
    bank             TEXT,
    account_last4    CHAR(4),
    opening_balance  NUMERIC(14,2),         -- balance in the main account on day 1
    opening_reserve  NUMERIC(14,2),         -- FD/RD/savings outside main account on day 1
    emi_amount       NUMERIC(12,2),
    sip_amount       NUMERIC(12,2)
);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id              BIGINT PRIMARY KEY,
    customer_id         VARCHAR(10) NOT NULL REFERENCES customers(customer_id),
    txn_date            DATE NOT NULL,
    txn_ts              TIMESTAMP NOT NULL,
    amount              NUMERIC(12,2) NOT NULL CHECK (amount > 0),
    type                TEXT NOT NULL CHECK (type IN ('debit','credit')),
    raw_description     TEXT NOT NULL,      -- the ONLY text the categoriser may see
    merchant            TEXT,
    category            TEXT,               -- what the bank/user labelled (8% NULL, 3% wrong)
    true_category       TEXT,               -- generator ground truth, for evaluation only
    predicted_category  TEXT,               -- filled by ML categoriser
    confidence          REAL,
    channel             TEXT,               -- UPI | Card | NetBanking | AutoDebit | ATM | NEFT
    balance_after       NUMERIC(14,2)
);
CREATE INDEX IF NOT EXISTS idx_txn_cust_date ON transactions (customer_id, txn_date);
CREATE INDEX IF NOT EXISTS idx_txn_date      ON transactions (txn_date);
CREATE INDEX IF NOT EXISTS idx_txn_category  ON transactions (category);

CREATE TABLE IF NOT EXISTS sms_raw (
    sms_id          BIGINT PRIMARY KEY,
    customer_id     VARCHAR(10) REFERENCES customers(customer_id),
    sender          TEXT,
    body            TEXT NOT NULL,
    received_at     TIMESTAMP NOT NULL,
    is_transaction  BOOLEAN,                -- ground truth: is this a transaction alert?
    true_txn_id     BIGINT,                 -- ground truth link (for parser evaluation)
    parsed_txn_id   BIGINT                  -- filled by your SMS parser
);

-- Ground truth for evaluating the overspending detector
CREATE TABLE IF NOT EXISTS injected_events (
    event_id     INT PRIMARY KEY,
    customer_id  VARCHAR(10) NOT NULL REFERENCES customers(customer_id),
    event_type   TEXT NOT NULL,             -- category_creep | category_spike | job_loss | income_raise | medical_emergency
    category     TEXT,
    start_month  DATE NOT NULL,
    end_month    DATE NOT NULL,
    magnitude    NUMERIC(6,2),
    description  TEXT
);

-- ---------------- model output tables (written by later modules) ----------------
CREATE TABLE IF NOT EXISTS alerts (
    alert_id       BIGSERIAL PRIMARY KEY,
    customer_id    VARCHAR(10) NOT NULL REFERENCES customers(customer_id),
    alert_month    DATE NOT NULL,
    category       TEXT NOT NULL,
    alert_type     TEXT NOT NULL,           -- sustained | spike
    current_spend  NUMERIC(12,2),
    baseline_mean  NUMERIC(12,2),
    pct_change     NUMERIC(8,2),
    z_score        NUMERIC(8,2),
    message        TEXT,
    created_at     TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id      BIGSERIAL PRIMARY KEY,
    customer_id      VARCHAR(10) NOT NULL REFERENCES customers(customer_id),
    category         TEXT,                  -- NULL = total spend
    target_month     DATE NOT NULL,
    model_name       TEXT NOT NULL,         -- naive | arima | prophet | lstm | xgboost
    predicted_amount NUMERIC(12,2),
    lower_bound      NUMERIC(12,2),
    upper_bound      NUMERIC(12,2),
    created_at       TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS health_scores (
    customer_id        VARCHAR(10) NOT NULL REFERENCES customers(customer_id),
    score_month        DATE NOT NULL,
    total_score        NUMERIC(5,2),
    savings_score      NUMERIC(5,2),
    debt_score         NUMERIC(5,2),
    emergency_score    NUMERIC(5,2),
    consistency_score  NUMERIC(5,2),
    income_score       NUMERIC(5,2),
    top_drag           TEXT,                -- component pulling the score down the most
    PRIMARY KEY (customer_id, score_month)
);

-- ---------------- authentication, sessions, consents (core/auth.py, core/privacy.py) ----------------
-- One login per customer. New sign-ups get a fresh, empty customer row; the shared demo login points at an existing
-- synthetic customer (DEMO_CUSTOMER_ID). Passwords are stored only as salted scrypt hashes; email-link tokens and web
-- session ids only as SHA-256 hashes. NOTE: load_to_postgres --reset runs TRUNCATE customers ... CASCADE, which also
-- empties these tables. Existing databases: apply sql/migrations/002_auth_extras.sql instead of re-running this file.
CREATE TABLE IF NOT EXISTS users (
    user_id           TEXT PRIMARY KEY,
    email             TEXT NOT NULL UNIQUE CHECK (email = lower(email)),
    password_hash     TEXT NOT NULL,
    customer_id       VARCHAR(10) NOT NULL UNIQUE REFERENCES customers(customer_id),
    is_demo           BOOLEAN NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMP NOT NULL DEFAULT now(),
    last_login_at     TIMESTAMP,
    email_verified_at TIMESTAMP,                       -- NULL = not verified yet
    token_version     INTEGER NOT NULL DEFAULT 0       -- bumped by a password reset: older API tokens stop working
);

CREATE TABLE IF NOT EXISTS auth_tokens (               -- email verification + password reset links (single use)
    token_hash  TEXT PRIMARY KEY,                      -- SHA-256 of the token in the email; the token itself is never stored
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    purpose     TEXT NOT NULL CHECK (purpose IN ('verify_email', 'reset_password')),
    expires_at  TIMESTAMP NOT NULL,
    created_at  TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS web_sessions (              -- Streamlit login cookie (survives a browser refresh, can be revoked)
    session_hash TEXT PRIMARY KEY,                     -- SHA-256 of the cookie value
    user_id      TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at   TIMESTAMP NOT NULL,
    expires_at   TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_web_sessions_user ON web_sessions (user_id);

CREATE TABLE IF NOT EXISTS rate_events (               -- shared rate limiting (login, password reset, chat); keys are hashed
    rate_key  TEXT NOT NULL,
    ts        TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rate_events ON rate_events (rate_key, ts);

CREATE TABLE IF NOT EXISTS consents (                  -- what each user agreed to, against which privacy-notice version
    consent_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    purpose         TEXT NOT NULL,                     -- account | age_18_plus | statement_processing
    notice_version  TEXT NOT NULL,
    given_at        TIMESTAMP NOT NULL,
    withdrawn_at    TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_consents_user ON consents (user_id, purpose);

-- Upgrade a database that already has the first `users` table (login only) to the full auth schema.
-- Safe to run more than once.  PowerShell:  Get-Content sql/migrations/002_auth_extras.sql | docker exec -i finance_db psql -U finance -d finance_advisor
ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMP;
ALTER TABLE users ADD COLUMN IF NOT EXISTS token_version INTEGER NOT NULL DEFAULT 0;
-- accounts created before verification existed are treated as verified
UPDATE users SET email_verified_at = COALESCE(email_verified_at, created_at) WHERE email_verified_at IS NULL;

CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    purpose     TEXT NOT NULL CHECK (purpose IN ('verify_email', 'reset_password')),
    expires_at  TIMESTAMP NOT NULL,
    created_at  TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS web_sessions (
    session_hash TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at   TIMESTAMP NOT NULL,
    expires_at   TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_web_sessions_user ON web_sessions (user_id);
CREATE TABLE IF NOT EXISTS rate_events (
    rate_key  TEXT NOT NULL,
    ts        TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rate_events ON rate_events (rate_key, ts);
CREATE TABLE IF NOT EXISTS consents (
    consent_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    purpose         TEXT NOT NULL,
    notice_version  TEXT NOT NULL,
    given_at        TIMESTAMP NOT NULL,
    withdrawn_at    TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_consents_user ON consents (user_id, purpose);

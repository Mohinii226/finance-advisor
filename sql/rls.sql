-- =====================================================================================================================
-- Row-level security for FinSight (optional but recommended before real users). Defence in depth: even if some query forgot
-- its `WHERE customer_id = :cid`, the database itself only shows the logged-in customer's rows. See core/rls.py.
--
-- Run as the database owner (the `finance` superuser), AFTER the data and views exist, and AGAIN after every
-- `load_to_postgres` (it drops and recreates the views, which resets their security_invoker setting and grants).
--
--   PowerShell:  Get-Content sql/rls.sql | docker exec -i finance_db psql -U finance -d finance_advisor -v ON_ERROR_STOP=1 -v app_password=YourStrongPassword1
--   bash:        cat sql/rls.sql | docker exec -i finance_db psql -U finance -d finance_advisor -v ON_ERROR_STOP=1 -v app_password="$APP_DB_PASSWORD"
--
-- Then point the API / app at the new role (APP_DATABASE_URL=postgresql+psycopg2://finsight_app:<password>@host:5432/finance_advisor)
-- and set RLS=on. Needs PostgreSQL 15+ (views with security_invoker); docker-compose uses 16.
-- =====================================================================================================================

-- 1. a limited login role for the API and the app (not a superuser, cannot bypass RLS, owns nothing, cannot change tables)
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'finsight_app') THEN
    CREATE ROLE finsight_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
  END IF;
  EXECUTE format('GRANT CONNECT ON DATABASE %I TO finsight_app', current_database());
END $$;
ALTER ROLE finsight_app WITH PASSWORD :'app_password';
GRANT USAGE ON SCHEMA public TO finsight_app;

-- 2. only the privileges the application code actually uses
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM finsight_app;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO finsight_app;                       -- tables and views
GRANT INSERT, UPDATE, DELETE ON users, consents, accounts TO finsight_app;
GRANT INSERT, DELETE ON customers, transactions, auth_tokens, web_sessions, rate_events TO finsight_app;
GRANT DELETE ON sms_raw, alerts, forecasts, health_scores, injected_events TO finsight_app;   -- only for "delete my account"
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO finsight_app;           -- views recreated later stay readable

-- 3. every table that has a customer_id: rows are visible / writable only for the customer set on the transaction
--    (`app.customer_id`), or by the auth code (`app.bypass`). Nothing set = nothing visible (current_setting(..., true) is NULL).
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['customers', 'accounts', 'transactions', 'sms_raw', 'alerts', 'forecasts', 'health_scores', 'injected_events'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS finsight_isolation ON %I', t);
    EXECUTE format($f$CREATE POLICY finsight_isolation ON %I
        USING (customer_id = current_setting('app.customer_id', true) OR current_setting('app.bypass', true) = 'on')
        WITH CHECK (customer_id = current_setting('app.customer_id', true) OR current_setting('app.bypass', true) = 'on')$f$, t);
  END LOOP;
END $$;

-- 4. views normally run with their OWNER's rights (a superuser: RLS would be skipped). Make them run with the caller's rights.
DO $$
DECLARE v text;
BEGIN
  FOR v IN SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
           WHERE n.nspname = 'public' AND c.relkind = 'v' LOOP
    EXECUTE format('ALTER VIEW %I SET (security_invoker = true)', v);
  END LOOP;
END $$;

-- 5. report
SELECT c.relname AS "table / view", c.relrowsecurity AS rls_on, c.reloptions AS options
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND (c.relrowsecurity OR c.relkind = 'v') ORDER BY c.relkind, c.relname;

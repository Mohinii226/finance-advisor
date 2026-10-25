# Deploying FinSight on an Ubuntu server

Stack: PostgreSQL (private) + API + Streamlit app, all behind Caddy, which gets free HTTPS certificates automatically.
Everything below runs on the server. `alias dc='docker compose -f docker-compose.prod.yml --env-file .env.production'`

## 0. What you need
- An Ubuntu 22.04/24.04 server with a **public IP** (a cloud VM or VPS), Docker + the compose plugin installed.
  (A laptop or WSL box behind a home router cannot get Let's Encrypt certificates. Use a tunnel such as Cloudflare Tunnel instead
  and point it at `app:8501`; skip the Caddy service.)
- A domain with two DNS **A records** to the server's IP: `finsight.example.com` (app) and `api.finsight.example.com` (API, optional).
- An SMTP account for sending email. Verification and reset links cannot work without one.
- Ports 80 and 443 free on the server.

## 1. Harden the server (once)
```bash
bash deploy/setup_ubuntu.sh        # firewall (22/80/443), fail2ban, automatic security updates
```
Use SSH keys and disable SSH password login.

## 2. Configure
```bash
git clone <your repo> finance-advisor && cd finance-advisor
cp .env.production.example .env.production && chmod 600 .env.production
openssl rand -hex 24                                         # -> POSTGRES_PASSWORD
python3 -c "import secrets; print(secrets.token_urlsafe(48))"  # -> JWT_SECRET
nano .env.production                                          # fill every value
```
Complete `docs/privacy_notice_template.md` (every `[[placeholder]]`). In production, sign-up stays off until you do.

## 3. Start the database and load the demo world (once, on a fresh database)
```bash
dc up -d db
dc build jobs
dc run --rm jobs python deploy/bootstrap.py       # synthetic customers, views, alerts, health scores, forecasts (~minutes)
```
`bootstrap.py` refuses to run once any user exists (it would delete them). Never run `data_gen.load_to_postgres --reset` on a live database.

## 4. Lock the data down: row-level security
The API and app connect as a limited role, `finsight_app`, and the database itself then shows each logged-in customer only their own rows
(even if a future query forgot its `WHERE customer_id = ...`). Run as the owner, **and again after any `load_to_postgres`** (it recreates the views):
```bash
set -a; . ./.env.production; set +a
cat sql/rls.sql | dc exec -T db psql -U finance -d finance_advisor -v ON_ERROR_STOP=1 -v app_password="$APP_DB_PASSWORD"
```
The last lines print each table with `rls_on = t` and each view with `security_invoker=true`. Put the same password into `APP_DATABASE_URL` in `.env.production`.
With `RLS=on` (set by the compose file) the API / app **refuse to start** if their role is a superuser, or if any view is not `security_invoker`.

## 5. Start the site and check
```bash
dc up -d --build api app caddy
dc ps                          # all "healthy" / running
dc logs -f caddy               # watch the certificates being issued (DNS must already point here)
curl https://api.finsight.example.com/health      # {"status":"ok","auth":"jwt","signup":"open",...}
```
Open the app, create an account, confirm the email arrives and its link works, log in, refresh the page (you stay logged in), upload a small
statement CSV, try "Forgot password", then delete the test account under *Your data*.
To prove the isolation on this database, run `tests/test_rls_postgres.py` (instructions at the top of the file).

## 6. Schedule jobs
```bash
crontab -e
0 2 * * *  /home/YOU/finance-advisor/deploy/backup.sh        >> /home/YOU/finsight-backup.log 2>&1
30 2 * * * /home/YOU/finance-advisor/deploy/nightly_jobs.sh  >> /home/YOU/finsight-jobs.log 2>&1
```
Run both by hand once first. Copy `backups/` off the server (encrypted).

## 7. Update
```bash
git pull && dc up -d --build api app caddy
```
If you ever reload data (`load_to_postgres`), re-run step 4 before using the site again.
Existing database that predates the auth tables: `cat sql/migrations/002_auth_extras.sql | dc exec -T db psql -U finance -d finance_advisor`.

## Troubleshooting
| Symptom | Likely cause |
|---|---|
| Caddy keeps retrying certificates | DNS not pointing at this server yet, or 80/443 blocked at the cloud firewall |
| App shows "Login is not available" | auth tables missing (apply the migration) or a bad SMTP / env value: read the "Technical detail" line, `dc logs app` |
| API container exits at start | `JWT_SECRET` missing or shorter than 32 characters |
| No verification email | wrong SMTP settings (`dc logs app api`), spam folder, or the provider needs a verified sender domain |
| Refresh logs you out | Streamlit older than 1.37, or the browser blocks the cookie: it is best effort by design |

## Before inviting real users
- [ ] Privacy notice completed and reviewed; `NOTICE_VERSION` bumped when it changes.
- [ ] `ANTHROPIC_API_KEY` left empty, **or** the notice discloses that chat data goes to Anthropic.
- [ ] SMTP working end to end; `REQUIRE_EMAIL_VERIFICATION=true`.
- [ ] Backups tested by restoring into a scratch database; backups protected and rotated.
- [ ] `.env.production` not in git (check `git status`); secrets never pasted into chats or tickets.
- [ ] You know how you would notify users if there were a breach (DPDP breach-notification duty).
- [ ] Only synthetic data on the public demo; real users' data stays in their own accounts.
- [ ] Step 4 done and `tests/test_rls_postgres.py` passes against this database.
- [ ] Caddy is the only thing exposed (the `db`, `api` and `app` services publish no ports).

## Trying row-level security on your laptop (optional)
```powershell
Get-Content sql/rls.sql | docker exec -i finance_db psql -U finance -d finance_advisor -v ON_ERROR_STOP=1 -v app_password=LocalPassw0rd1
```
Add to `.env`: `APP_DATABASE_URL=postgresql+psycopg2://finsight_app:LocalPassw0rd1@localhost:5432/finance_advisor` and `RLS=on`.
The API and Streamlit then use the limited role, while the generate / load tasks keep using `DATABASE_URL` (the owner). Re-run the first
command after "3. Load data + create views". Remove `RLS=on` to go back. Prove it works:
```powershell
$env:RLS_TEST_ADMIN_URL="postgresql+psycopg2://finance:finance@localhost:5432/finance_advisor"
$env:RLS_TEST_APP_URL="postgresql+psycopg2://finsight_app:LocalPassw0rd1@localhost:5432/finance_advisor"
python -m pytest tests/test_rls_postgres.py -v
```

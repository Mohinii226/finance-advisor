#!/usr/bin/env bash
# Nightly database backup (custom-format dump, kept 14 days).   Cron:  0 2 * * * /home/YOU/finance-advisor/deploy/backup.sh
# The dumps contain users' financial data: keep the folder private (chmod 700), copy it off the server (encrypted), and
# remember that an account deleted today still exists in older backups until they rotate out (say so in the privacy notice).
# Restore:  cat backups/finsight-XXXX.dump | docker compose -f docker-compose.prod.yml --env-file .env.production exec -T db \
#             pg_restore -U finance -d finance_advisor --clean --if-exists
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077
mkdir -p backups
STAMP=$(date +%Y%m%d-%H%M%S)
docker compose -f docker-compose.prod.yml --env-file .env.production exec -T db \
  pg_dump -U finance -d finance_advisor -Fc > "backups/finsight-$STAMP.dump"
find backups -name 'finsight-*.dump' -mtime +14 -delete
echo "backup written: backups/finsight-$STAMP.dump ($(du -h "backups/finsight-$STAMP.dump" | cut -f1))"

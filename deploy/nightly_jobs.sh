#!/usr/bin/env bash
# Refresh the batch-computed tables (alerts, health scores, forecasts) so users who imported statements today see them.
# Cron (as the deploy user):   30 2 * * * /home/YOU/finance-advisor/deploy/nightly_jobs.sh >> /home/YOU/finsight-jobs.log 2>&1
# FIRST run it by hand once and check the row counts in `alerts` / `health_scores` / `forecasts` before trusting the cron job:
# I could not run these modules here, so I do not know whether they replace or append rows when re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
DC="docker compose -f docker-compose.prod.yml --env-file .env.production"
CMDS="python -m ml.anomaly.run_detector --preset balanced && python -m core.health_score && python -m ml.forecasting.run_forecast --source db"
if [ -d models/categorizer ]; then          # the categoriser improves imported rows' categories (it was trained on the synthetic data)
  CMDS="python -m ml.categorizer.predict --dir models/categorizer && $CMDS"
fi
echo "[$(date -Is)] nightly jobs start"
$DC run --rm jobs sh -c "$CMDS"
echo "[$(date -Is)] nightly jobs done"

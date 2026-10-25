"""
ONE-OFF first-time setup of a fresh production database with the synthetic demo world: generates data, loads it (which also
creates dim_category and the analytics views), then fills the alerts / health-score / forecast tables.

    dc run --rm jobs python deploy/bootstrap.py

It REFUSES to run when any user account exists, because `load_to_postgres --reset` truncates `customers ... CASCADE`, which would
delete every real user and their transactions. Set FORCE=1 only if you really mean to wipe everything.
"""
import os
import subprocess
import sys

from sqlalchemy import create_engine, text


def user_count(url: str) -> int:
    try:
        with create_engine(url).connect() as c:
            return int(c.execute(text("SELECT COUNT(*) FROM users")).scalar() or 0)
    except Exception:                                   # table does not exist yet: nothing to protect
        return 0


def main() -> None:
    url = os.environ["DATABASE_URL"]
    n = user_count(url)
    if n and os.getenv("FORCE") != "1":
        sys.exit(f"Refusing to run: {n} user account(s) exist and this would delete them. (FORCE=1 overrides.)")
    steps = [
        ["-m", "data_gen.generate", "--customers", "300", "--months", "36", "--seed", "42"],
        ["-m", "data_gen.load_to_postgres", "--reset"],
        ["-m", "ml.anomaly.run_detector", "--preset", "balanced"],
        ["-m", "core.health_score"],
        ["-m", "ml.forecasting.run_forecast", "--source", "db"],
    ]
    for step in steps:
        print(">>", "python", *step, flush=True)
        subprocess.run([sys.executable, *step], check=True)
    print("Bootstrap finished. Set DEMO_CUSTOMER_ID=C0001 in .env.production and restart api + app to enable the demo login.")


if __name__ == "__main__":
    main()

"""Shared environment for chatbot / service / API tests: simulated world + SQLite views + detector alerts."""
from ml.anomaly.evaluate import grid_from_transactions
from ml.anomaly.overspend import BALANCED, detect
from ml.anomaly.run_detector import alerts_to_frame
from tests.sqlite_harness import run
from tests.test_core import data

_STATE = {}


def chat_env() -> dict:
    if "env" not in _STATE:
        D = data()
        con = D["con"]
        al = alerts_to_frame(detect(grid_from_transactions(D["d"]["transactions"]), BALANCED), BALANCED)
        al["alert_month"] = al["alert_month"].astype(str)
        al.to_sql("alerts", con, index=False, if_exists="replace")
        ev = D["d"]["injected_events"]
        cid = ev[ev.event_type == "category_creep"].iloc[1].customer_id
        other = next(c for c in D["d"]["customers"].customer_id if c != cid)
        _STATE["env"] = dict(D=D, con=con, run=lambda n, **kw: run(con, n, **kw), cid=cid, other=other,
                             txns=D["d"]["transactions"])
    return _STATE["env"]

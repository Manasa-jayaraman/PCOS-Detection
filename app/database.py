# Thin wrapper to align with spec: provides the same API as db.py
from .db import (
    init_db,
    create_user,
    verify_user,
    save_prediction,
    get_predictions,
    save_daily_log,
    get_daily_logs,
    get_conn,
    DB_PATH,
)

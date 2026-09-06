#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuleShield AI -- Pre-Presentation Demo Auth Reset
SIH26184 | Ministry of Home Affairs (MHA) / I4C

Resets windowed failed login counts, cumulative credential-stuffing counters,
and lockout timestamps to zero so rehearsal attempts do not trigger alerts
or lockouts during live jury demonstrations.

Usage:
    venv\\Scripts\\python.exe scripts\\reset_demo_auth.py
"""

import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("MULESHIELD_DB_PATH", ROOT / "data" / "muleshield.db"))


def reset_auth() -> None:
    if not DB_PATH.exists():
        print(f"[!] Database not found at: {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.execute(
            "UPDATE users SET failed_logins = 0, failed_logins_total = 0, locked_until = NULL"
        )
        conn.commit()
        rows = conn.execute(
            "SELECT username, failed_logins, failed_logins_total, locked_until FROM users"
        ).fetchall()
        print(f"[OK] Demo auth telemetry reset for {cur.rowcount} user(s) at {DB_PATH.name}:")
        for u, f, ft, lock in rows:
            print(f"     - {u}: failed={f}, total={ft}, locked={lock}")
    finally:
        conn.close()


if __name__ == "__main__":
    reset_auth()

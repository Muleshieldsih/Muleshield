# -*- coding: utf-8 -*-
"""Email transport. Simulated; see backend/adapters/__init__.py.

No smtplib import. The audit found email did not exist anywhere in this
repository -- not even in the password-reset flow, which routes a reset token
through a named administrator precisely because there is no mail server. This
adapter is the seam where one would attach, and nothing above it would change.
"""

import logging

from backend.adapters import should_fail, _ref

logger = logging.getLogger("muleshield.adapters.email")


class EmailAdapter:
    channel = "email"

    def send(self, recipient: str, subject: str, body: str,
             context: dict | None = None) -> dict:
        key = f"email:{recipient}:{subject}"
        if should_fail(key):
            logger.warning("[DISPATCH][email] FAILED -> %s", recipient)
            return {"ok": False, "provider_ref": "", "error": "simulated SMTP failure"}
        logger.info("[DISPATCH][email] -> %s : %s", recipient, subject)
        return {"ok": True, "provider_ref": _ref("MAIL", key), "error": ""}

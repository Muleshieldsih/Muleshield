# -*- coding: utf-8 -*-
"""SMS transport. Simulated; see backend/adapters/__init__.py."""

import logging

from backend.adapters import should_fail, _ref

logger = logging.getLogger("muleshield.adapters.sms")


class SMSAdapter:
    channel = "sms"

    def send(self, recipient: str, subject: str, body: str,
             context: dict | None = None) -> dict:
        key = f"sms:{recipient}:{subject}"
        # 160 chars is not decoration: an SMS that splits arrives out of order on
        # some Indian carriers, and an alert whose second half lands first is
        # worse than one that is terse.
        text = f"{subject} {body}"[:160]
        if should_fail(key):
            logger.warning("[DISPATCH][sms] FAILED -> %s", recipient)
            return {"ok": False, "provider_ref": "", "error": "simulated gateway failure"}
        logger.info("[DISPATCH][sms] -> %s : %s", recipient, text)
        return {"ok": True, "provider_ref": _ref("SMS", key), "error": ""}

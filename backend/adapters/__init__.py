# -*- coding: utf-8 -*-
"""
MuleShield AI -- Notification transport adapters
SIH26184 | MHA / I4C
"""


from __future__ import annotations

import hashlib
import logging
import os
from typing import Protocol

logger = logging.getLogger("muleshield.adapters")


def fail_rate() -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("MULESHIELD_ADAPTER_FAIL_RATE", "0"))))
    except ValueError:
        return 0.0


def should_fail(key: str) -> bool:
    """Deterministic per-key failure.

    Deterministic rather than random so a test can assert an exact outcome, and
    so a retry of the same delivery does not flip its verdict for reasons the
    operator cannot see.
    """
    rate = fail_rate()
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < rate


def _ref(prefix: str, key: str) -> str:
    return f"MOCK-{prefix}-{hashlib.sha256(key.encode()).hexdigest()[:8].upper()}"


class Adapter(Protocol):
    channel: str

    def send(self, recipient: str, subject: str, body: str,
             context: dict | None = None) -> dict:
        """Returns {'ok': bool, 'provider_ref': str, 'error': str}."""
        ...


from backend.adapters.sms import SMSAdapter          # noqa: E402
from backend.adapters.email import EmailAdapter      # noqa: E402
from backend.adapters.webhook import WebhookAdapter  # noqa: E402

ADAPTERS: dict[str, Adapter] = {
    "sms": SMSAdapter(),
    "email": EmailAdapter(),
    "webhook": WebhookAdapter(),
}


def for_channel(channel: str) -> Adapter | None:
    return ADAPTERS.get(str(channel).lower())

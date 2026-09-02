# -*- coding: utf-8 -*-
"""
MuleShield AI -- notification transports.
SIH26184 | MHA / I4C

REAL INTERFACES, MOCKED TRANSPORT -- AND THE LABEL MATTERS
-----------------------------------------------------------
Every adapter here implements the full contract an operator depends on: it is
handed a recipient and a message, it reports success or failure, and it returns
a provider reference the delivery record can be reconciled against later. What
none of them does is put a byte on a wire.

That is a deliberate choice, not an unfinished one. A live SMS route into India
needs a paid gateway and DLT template registration; a live CFCFRMS call needs
credentials nobody outside I4C has. Wiring a real gateway is a config change to
one module -- swapping the adapter -- and everything upstream of it, the rules,
the queue, the retry, the delivery record, the acknowledgement, is the real
thing already.

The console says "simulated" on every channel. An alert that claims to have been
sent when it was not is worse than no alerting at all, because a force would
stand down believing it had been warned.

FAILURE INJECTION
-----------------
MULESHIELD_ADAPTER_FAIL_RATE (0.0 by default) makes sends fail deterministically,
keyed on the delivery so a retry of the SAME delivery behaves consistently. It
exists because COMPLAINT_AUDIT.md finding 5.5 was that the old simulated SMS
reported success unconditionally: a control that cannot fail is a control nobody
has tested. Set it to 1.0 and the retry and dead-letter paths run for real.
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

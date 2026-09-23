# -*- coding: utf-8 -*-
"""
MuleShield AI -- Outbound API transport for CFCFRMS and Samanvaya
SIH26184 | MHA / I4C
"""


from __future__ import annotations

import json
import logging

from backend.adapters import should_fail, _ref

logger = logging.getLogger("muleshield.adapters.webhook")


def cfcfrms_payload(alert: dict) -> dict:
    """A fund-blocking request shaped for the CFCFRMS rail.

    Deliberately carries the FORECAST and its confidence, not just an assertion.
    A bank asked to hold funds is entitled to know how strong the signal is and
    how much of it is live evidence rather than historical pattern -- which is
    what prior_share reports. A block request that cannot be interrogated is one
    a compliance officer is right to refuse.
    """
    return {
        "schema": "muleshield.cfcfrms.block-request/v1-proposed",
        "issued_at": alert.get("created_at"),
        "issuer": {"system": "MuleShield AI", "authority": "I4C", "reference": alert.get("id")},
        "action": "PRECAUTIONARY_HOLD",
        "urgency": alert.get("severity"),
        "complaint_references": alert.get("complaint_ids") or [],
        "amount_at_risk_inr": round(float(alert.get("rupees_at_risk", 0.0)), 2),
        "predicted_cashout": {
            "cell_id": alert.get("cell_id"),
            "district": alert.get("district"),
            "state": alert.get("state"),
            "window_minutes_from_now": [alert.get("window_start_min"),
                                        alert.get("window_end_min")],
        },
        "confidence": {
            "rule": alert.get("rule_id"),
            # The decomposition, exported. A consumer can weight our request by
            # how much of it is live forecast versus historical prior.
            "live_share": round(1.0 - float(alert.get("prior_share", 0.0)), 4),
            "historical_prior_share": round(float(alert.get("prior_share", 0.0)), 4),
            "contributing_cases": alert.get("case_count", 0),
        },
        "human_in_the_loop": {
            "acknowledged": alert.get("status") == "acknowledged",
            "acknowledged_by": alert.get("acknowledged_by"),
            "disposition": alert.get("disposition"),
        },
    }


def samanvaya_payload(alert: dict) -> dict:
    """An intelligence dissemination event shaped for LEA coordination.

    Scoped by state and district because the problem statement's unit of action
    is "LEAs at the state and local levels, coordinated by I4C" -- three tiers,
    which a national broadcast cannot represent.
    """
    return {
        "schema": "muleshield.samanvaya.dissemination/v1-proposed",
        "event_type": "PREDICTED_CASHOUT_WINDOW",
        "issued_at": alert.get("created_at"),
        "reference": alert.get("id"),
        "severity": alert.get("severity"),
        "headline": alert.get("headline"),
        "jurisdiction": {"state": alert.get("state"), "district": alert.get("district")},
        "target": {
            "cell_id": alert.get("cell_id"),
            "atm_cluster_radius_km": 12.0,
            "window_minutes_from_now": [alert.get("window_start_min"),
                                        alert.get("window_end_min")],
        },
        "evidence": {
            "linked_complaints": alert.get("complaint_ids") or [],
            "amount_at_risk_inr": round(float(alert.get("rupees_at_risk", 0.0)), 2),
            "live_share": round(1.0 - float(alert.get("prior_share", 0.0)), 4),
        },
        "advisory": (
            "Ranked forecast, not a confirmed location. Deploy to observe; do not "
            "treat presence in this cell as grounds for detention."
        ),
    }


class WebhookAdapter:
    channel = "webhook"

    def send(self, recipient: str, subject: str, body: str,
             context: dict | None = None) -> dict:
        """Build and log the payload the named rail would receive.

        `recipient` names the rail (cfcfrms / samanvaya / anything else). Nothing
        is transmitted; the payload is logged in full so it can be inspected,
        diffed against a real contract when one is available, and pasted into an
        integration conversation.
        """
        alert = (context or {}).get("alert") or {}
        rail = str(recipient).strip().lower()
        if "cfcfrms" in rail:
            payload = cfcfrms_payload(alert)
        elif "samanvaya" in rail:
            payload = samanvaya_payload(alert)
        else:
            payload = {"schema": "muleshield.generic/v1-proposed",
                       "subject": subject, "body": body,
                       "reference": alert.get("id")}

        key = f"webhook:{rail}:{alert.get('id', subject)}"
        if should_fail(key):
            logger.warning("[DISPATCH][webhook] FAILED -> %s", rail)
            return {"ok": False, "provider_ref": "", "error": "simulated endpoint 5xx"}

        logger.info("[DISPATCH][webhook] -> %s\n%s", rail,
                    json.dumps(payload, indent=2, default=str))
        return {"ok": True, "provider_ref": _ref("HOOK", key), "error": ""}

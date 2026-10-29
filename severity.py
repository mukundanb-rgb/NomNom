"""
Severity engine — a direct, auditable implementation of policy sections 2.1-2.5.

This module is deliberately rule-based, not learned. The severity matrix is a
business decision written down by the brand; a model must not quietly re-derive
it. Every function returns the policy clause it applied so the UI can cite it.
"""
from dataclasses import dataclass, asdict
from typing import Optional, List

# policy 2.2
HARM_BY_TOPIC = {
    "praise": "H0", "promo_query": "H0",
    "delivery_delay": "H1", "payment_refund": "H1", "app_bug": "H1", "pricing_surge": "H1",
    "rider_behaviour": "H2", "customer_service": "H2",
    "food_safety": "H3", "discrimination_allegation": "H3",
}

# policy 2.1
def reach_band(reach: float) -> str:
    if reach < 2_000:    return "R0"
    if reach < 25_000:   return "R1"
    if reach < 100_000:  return "R2"
    return "R3"

BAND_LABEL = {"R0":"Contained","R1":"Elevated","R2":"Viral","R3":"Mass reach"}

# policy 2.3
MATRIX = {
    "H0": {"R0":"SEV-4","R1":"SEV-4","R2":"SEV-3","R3":"SEV-3"},
    "H1": {"R0":"SEV-4","R1":"SEV-3","R2":"SEV-2","R3":"SEV-2"},
    "H2": {"R0":"SEV-3","R1":"SEV-2","R2":"SEV-2","R3":"SEV-1"},
    "H3": {"R0":"SEV-2","R1":"SEV-2","R2":"SEV-1","R3":"SEV-1"},
}

# policy 2.4
SLA  = {"SEV-1":"30 minutes","SEV-2":"2 hours","SEV-3":"8 hours","SEV-4":"24 hours / optional"}
APPROVER = {"SEV-1":"Director, Brand Communications + Legal counsel",
            "SEV-2":"Social Media Manager",
            "SEV-3":"Social Media Executive",
            "SEV-4":"Social Media Executive"}

# EXPECTED cost of failing to catch each tier in time. Read the word "expected"
# carefully, because this is where most student projects (and our own first draft)
# go wrong. A full-blown PR crisis costs the brand about Rs 8,00,000 in emergency
# agency retainer, executive hours, paid response and measured churn. But a missed
# SEV-1 does not become a full crisis every time -- the brand's own incident history
# says roughly one in five escalates that far. So the number we optimise against is
#       800,000 x 0.19 ~= 150,000
# not 800,000. Costing every missed SEV-1 at the full crisis price is what produced
# our absurd first estimate of Rs 83,000 crore a year. See docs/PROMPT_LOG.md entry 8
# and docs/BUSINESS_CASE.md section 3, which also runs these numbers at +/-50%.
FULL_CRISIS_COST   = 800_000
P_ESCALATION       = 0.19
MISS_COST = {"SEV-1": int(FULL_CRISIS_COST*P_ESCALATION),  # 152,000
             "SEV-2": 12_000,   # viral complaint left unanswered: churn + amplification
             "SEV-3":    800,   # ordinary complaint answered late
             "SEV-4":      0}

RANK = {"SEV-1":1, "SEV-2":2, "SEV-3":3, "SEV-4":4}

@dataclass
class Severity:
    severity: str
    harm_class: str
    band: str
    band_label: str
    sla: str
    approver: str
    is_crisis: bool
    miss_cost_inr: int
    reasons: List[str]
    routed_to_human_review: bool = False
    review_reason: Optional[str] = None
    def dict(self): return asdict(self)

def assess(topic: str, predicted_reach: float, sentiment: str = "negative",
           confidence: float = 1.0, lang: str = "en", is_sarcastic: bool = False) -> Severity:
    """Apply policy 2.3 then the override rules in 2.5 and section 5."""
    harm = HARM_BY_TOPIC.get(topic, "H1")
    band = reach_band(predicted_reach)
    sev  = MATRIX[harm][band]
    reasons = [f"Policy 2.2: topic '{topic}' is harm class {harm}.",
               f"Policy 2.1: predicted reach {int(predicted_reach):,} falls in band {band} ({BAND_LABEL[band]}).",
               f"Policy 2.3: matrix[{harm}][{band}] = {sev}."]

    routed, review_reason = False, None

    # policy 4.6 — discrimination is SEV-1 regardless of reach
    if topic == "discrimination_allegation" and RANK[sev] > 1:
        sev = "SEV-1"
        reasons.append("Policy 4.6: discrimination allegations are SEV-1 regardless of reach.")

    # policy 5.2 — sarcasm, or positive sentiment on a harm-bearing topic
    contradiction = (sentiment == "positive" and harm in ("H1","H2","H3"))
    if is_sarcastic or contradiction:
        routed = True
        review_reason = ("Policy 5.2: sarcasm detected." if is_sarcastic else
                         "Policy 5.2: positive sentiment on a harm-bearing topic.")
        if RANK[sev] > 3:
            sev = "SEV-3"
            reasons.append("Policy 5.2: sarcastic/contradictory mentions are tiered at minimum SEV-3.")
        reasons.append(review_reason + " Routed to a human reviewer.")

    # policy 5.1 — low confidence on code-switched text
    if lang != "en" and confidence < 0.60:
        routed = True
        review_reason = review_reason or f"Policy 5.1: confidence {confidence:.2f} < 0.60 on non-English text."
        reasons.append(f"Policy 5.1: Hinglish/Devanagari mention with confidence {confidence:.2f} < 0.60 — human review, not auto-tiered.")

    # policy 5.3 — global confidence floor
    if confidence < 0.55:
        routed = True
        review_reason = review_reason or f"Policy 5.3: confidence {confidence:.2f} below the 0.55 floor."
        reasons.append(f"Policy 5.3: confidence {confidence:.2f} < 0.55 — human review regardless of topic.")

    # policy 2.5 — crisis trigger
    is_crisis = (sev == "SEV-1") or (sev == "SEV-2" and band == "R3")
    if is_crisis:
        reasons.append("Policy 2.5: " + ("severity is SEV-1." if sev=="SEV-1"
                       else f"severity is SEV-2 at mass reach (band {band}). POSSIBLE CRISIS."))

    return Severity(severity=sev, harm_class=harm, band=band, band_label=BAND_LABEL[band],
                    sla=SLA[sev], approver=APPROVER[sev], is_crisis=is_crisis,
                    miss_cost_inr=MISS_COST[sev], reasons=reasons,
                    routed_to_human_review=routed, review_reason=review_reason)

def cluster_alert(timestamps_harm, window_minutes: int = 60, min_count: int = 5):
    """Policy 2.5 cluster rule: >=5 mentions of harm class H2/H3 in any 60-min window.
    timestamps_harm: list of (datetime, harm_class). Returns list of alert dicts."""
    pts = sorted([(t,h) for t,h in timestamps_harm if h in ("H2","H3")], key=lambda x: x[0])
    alerts, i = [], 0
    for i in range(len(pts)):
        j = i
        while j < len(pts) and (pts[j][0]-pts[i][0]).total_seconds() <= window_minutes*60:
            j += 1
        n = j - i
        if n >= min_count:
            alerts.append(dict(window_start=pts[i][0].isoformat(), window_end=pts[j-1][0].isoformat(),
                               count=n, severity="SEV-1",
                               reason=f"Policy 2.5 CLUSTER ALERT: {n} H2/H3 mentions within {window_minutes} minutes."))
    # keep only the widest non-overlapping alerts
    out = []
    for a in alerts:
        if not out or a["window_start"] > out[-1]["window_end"]:
            out.append(a)
        elif a["count"] > out[-1]["count"]:
            out[-1] = a
    return out

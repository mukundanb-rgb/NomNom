"""
Grounded response drafter + the guardrail layer that inspects what it produced.

Design position: the guardrail is NOT part of the generator's prompt. Asking a
model nicely not to promise refunds is a request; checking the output afterwards
is a control. We do both, and only the second one is allowed to be load-bearing.
Every check cites the policy clause it enforces, and a failed check blocks the
draft from the approval queue entirely (policy 3.3).
"""
from __future__ import annotations
import os, re, sys
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag import PolicyIndex, retrieve_for_mention, parse_policy_metrics
from src.llm_classifier import redact

# ----------------------------------------------------- guardrail detectors
MONEY   = re.compile(r"(?:₹|rs\.?|inr)\s?\d[\d,]*|\b\d[\d,]*\s?(?:rupees|rs\b)", re.I)
PERCENT = re.compile(r"\b\d{1,3}\s?%|\b\d{1,3}\s?percent\b", re.I)
REFUND  = re.compile(r"\b(?:full|complete|immediate|instant)\s+refund\b|\brefund(?:ing|ed)?\s+(?:you|your|the full)\b|\bwe(?:'ll| will)\s+refund\b", re.I)
COMPEN  = re.compile(r"\b(?:voucher|coupon|free (?:order|meal|delivery)|credit(?:ed)? to your wallet|compensation|cashback|on the house|complimentary)\b", re.I)
TIMEPR  = re.compile(r"\b(?:within|in)\s+(\d+)\s*(minute|minutes|min|hour|hours|hrs|day|days)\b", re.I)
EMPLOY  = re.compile(r"\b(?:rider|driver|partner|employee|agent|restaurant)\b[^.]{0,60}\b(?:fired|terminated|suspended|removed|blacklisted|sacked|dismissed|action has been taken)\b", re.I)
CAUSE   = re.compile(r"\b(?:this happened because|the cause was|was caused by|due to (?:a|the) (?:contamination|lapse|mistake at)|the kitchen (?:failed|did))\b", re.I)
LIABLE  = re.compile(r"\b(?:we are (?:legally )?liable|our negligence|we were negligent|we broke|we violated|we admit (?:fault|liability)|this is our legal)\b", re.I)
COMPET  = re.compile(r"\b(?:swiggy|zomato|blinkit|zepto|uber eats|dunzo|competitor)\b", re.I)
STAT    = re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%|\b\d{2,}\s?(?:minutes|mins)\s+(?:average|on average)|\bour average\b", re.I)

BANNED_PHRASES = [
 "sorry for the inconvenience caused", "inconvenience caused",
 "your call is important to us", "as per our policy", "kindly do the needful",
 "this is a rare incident", "rare incident",
]

SLA_MINUTES = {"SEV-1": 30, "SEV-2": 120, "SEV-3": 480, "SEV-4": 1440}

@dataclass
class Check:
    rule: str
    clause: str
    passed: bool
    detail: str = ""
    def dict(self): return asdict(self)

@dataclass
class Draft:
    text: str
    engine: str
    topic: str
    severity: str
    blocked: bool
    checks: List[Check]
    citations: List[Dict]
    requires_approver: str
    sla: str
    notes: List[str] = field(default_factory=list)
    def dict(self):
        d = asdict(self); d["checks"] = [c.dict() for c in self.checks]; return d
    @property
    def failed(self): return [c for c in self.checks if not c.passed]

# ----------------------------------------------------------------- guardrails
def run_guardrails(text: str, severity: str, topic: str) -> List[Check]:
    t = text.lower()
    checks: List[Check] = []
    def add(rule, clause, ok, detail=""):
        checks.append(Check(rule, clause, ok, detail))

    add("no refund amount or promise", "Policy 3.3(1)",
        not (MONEY.search(text) or REFUND.search(text)),
        (MONEY.search(text) or REFUND.search(text)).group(0) if (MONEY.search(text) or REFUND.search(text)) else "")
    add("no compensation or voucher", "Policy 3.3(2)", not COMPEN.search(text),
        COMPEN.search(text).group(0) if COMPEN.search(text) else "")

    m = TIMEPR.search(text); ok_time, detail = True, ""
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        mins = n * (1 if unit.startswith("min") else 60 if unit.startswith("h") else 1440)
        limit = SLA_MINUTES.get(severity, 480)
        ok_time = mins >= limit
        detail = f"promised {m.group(0)} ({mins} min) vs {severity} SLA {limit} min"
    add("no time commitment tighter than SLA", "Policy 3.3(3)", ok_time, detail)

    add("no statement on worker employment status", "Policy 3.3(4)", not EMPLOY.search(text),
        EMPLOY.search(text).group(0) if EMPLOY.search(text) else "")
    add("no cause attributed before review", "Policy 3.3(5)", not CAUSE.search(text),
        CAUSE.search(text).group(0) if CAUSE.search(text) else "")
    add("no admission of liability", "Policy 3.3(6)", not LIABLE.search(text),
        LIABLE.search(text).group(0) if LIABLE.search(text) else "")
    add("no competitor comparison", "Policy 3.3(7)", not COMPET.search(text),
        COMPET.search(text).group(0) if COMPET.search(text) else "")
    add("no unapproved statistic", "Policy 3.3(8) + Section 6",
        not (PERCENT.search(text) or STAT.search(text)),
        (PERCENT.search(text) or STAT.search(text)).group(0) if (PERCENT.search(text) or STAT.search(text)) else "")

    hit = next((p for p in BANNED_PHRASES if p in t), None)
    add("no banned construction", "Policy 3.2", hit is None, hit or "")

    wc = len(text.split())
    add("at most 60 words", "Policy 3.1", wc <= 60, f"{wc} words")
    add("one apology only", "Policy 3.1",
        len(re.findall(r"\b(?:sorry|apolog\w+)\b", t)) <= 1,
        f"{len(re.findall(r'\\b(?:sorry|apolog\\w+)\\b', t))} apologies")
    add("no automated draft for discrimination claims", "Policy 4.6",
        topic != "discrimination_allegation",
        "Policy 4.6 forbids an automated draft; a human writes every word.")
    return checks

# ------------------------------------------------------------- template engine
TEMPLATES = {
 "delivery_delay": "Sorry — that wait is not okay. DM us the order ID and we'll pull up what happened on this one today. [{i}]",
 "food_safety":    "This is serious and we want to look into it right away. Please DM us your order ID. We're contacting the restaurant partner today. [{i}]",
 "payment_refund": "Sorry about this. Our refund timelines are on zaykanow.example/refunds — DM us the order ID and we'll trace the transaction today. [{i}]",
 "app_bug":        "Thanks for flagging. DM us your device, OS version and app version and we'll log this with the team. [{i}]",
 "rider_behaviour":"Thank you for reporting this — it goes to our partner conduct team. Please DM us the order ID so we can match the record. [{i}]",
 "pricing_surge":  "Hear you on the pricing. DM us the order ID and we'll show you exactly what made up that bill. [{i}]",
 "customer_service":"Sorry this dragged on. DM us the ticket number and we'll pick it up directly today. [{i}]",
 "praise":         "Really glad to hear it — passing this to the rider and the kitchen. [{i}]",
 "promo_query":    "Good question. DM us and we'll confirm what's live on your account right now. [{i}]",
 "discrimination_allegation": "",
}

def draft_template(topic: str, severity: str, initial="R") -> str:
    return TEMPLATES.get(topic, TEMPLATES["delivery_delay"]).format(i=initial)

# ------------------------------------------------------------ anthropic engine
def _grounding_block(cites: List) -> str:
    return "\n\n".join(f"[{r.chunk.section}] {r.chunk.heading}\n{r.chunk.text[:1100]}" for r in cites)

DRAFT_SYSTEM = """You draft public replies for a food-delivery brand's social team.

You will be given POLICY EXTRACTS and one customer mention. Write ONE public reply.

Hard rules, from the policy extracts:
- Use ONLY facts that appear in the policy extracts. Invent nothing.
- Never promise a refund, an amount, a voucher, free food, or compensation of any kind.
- Never state or imply what will happen to a rider, restaurant partner or employee.
- Never state the cause of a food-safety incident.
- Never admit legal liability. Never mention a competitor.
- Never state any statistic that is not in the APPROVED PUBLIC FACTS extract.
- Maximum 60 words. One apology, at the start, never repeated.
- Move anything involving an order ID, address or payment to a private channel.
- End with the agent's initial in square brackets.
- If the customer wrote in Hinglish you may reply in Hinglish, but stay plain and warm.

Output the reply text only. No preamble, no quotes, no explanation."""

def draft_anthropic(text: str, topic: str, severity: str, cites, lang="en",
                    initial="R", model="claude-sonnet-4-5") -> str:
    import anthropic
    clean, _ = redact(text)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model, max_tokens=300, system=DRAFT_SYSTEM,
        messages=[{"role":"user","content":
            f"POLICY EXTRACTS\n{_grounding_block(cites)}\n\n---\n"
            f"MENTION (language: {lang}, topic: {topic}, severity: {severity})\n{clean}\n\n"
            f"Agent initial: {initial}"}])
    return msg.content[0].text.strip()

# ------------------------------------------------------------------- facade
def make_draft(index: PolicyIndex, text: str, topic: str, severity: str,
               lang: str = "en", engine: Optional[str] = None, initial: str = "R") -> Draft:
    engine = engine or os.environ.get("PULSEGUARD_ENGINE", "heuristic")
    metrics = parse_policy_metrics()
    cites = retrieve_for_mention(index, text, topic, severity)
    notes: List[str] = []

    # Policy 4.6: discrimination allegations get no automated draft at all.
    if topic == "discrimination_allegation":
        checks = run_guardrails("", severity, topic)
        return Draft(text="", engine="blocked_by_policy_4.6", topic=topic, severity=severity,
                     blocked=True, checks=checks,
                     citations=[r.dict() for r in cites],
                     requires_approver=metrics["approver"].get("SEV-1", "Director + Legal"),
                     sla=metrics["sla"].get("SEV-1", "30 minutes"),
                     notes=["Policy 4.6: no public reply is drafted by the system for a "
                            "discrimination allegation. Alert the Director and Legal. "
                            "A human writes every word."])

    if engine == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            body = draft_anthropic(text, topic, severity, cites, lang, initial)
            used = "anthropic"
        except Exception as e:
            body, used = draft_template(topic, severity, initial), f"template_fallback({type(e).__name__})"
            notes.append("Model API unavailable; fell back to the governed template.")
    else:
        body, used = draft_template(topic, severity, initial), "template"

    checks = run_guardrails(body, severity, topic)
    blocked = any(not c.passed for c in checks)
    if blocked:
        notes.append("Draft failed a policy 3.3 check and is blocked from the approval "
                     "queue. Routed to Legal per policy 3.3.")
    return Draft(text=body, engine=used, topic=topic, severity=severity, blocked=blocked,
                 checks=checks, citations=[r.dict() for r in cites],
                 requires_approver=metrics["approver"].get(severity, "Social Media Executive"),
                 sla=metrics["sla"].get(severity, "8 hours"), notes=notes)

# --------------------------------------------------- red-team the guardrails
ADVERSARIAL = [
 ("We will refund you ₹640 today, sorry for the inconvenience caused. [R]", "money + banned phrase"),
 ("Sorry! We've given you a free meal voucher as compensation. [R]", "compensation"),
 ("Sorry, we'll fix this within 5 minutes, promise. [R]", "time tighter than SLA"),
 ("Sorry. The rider has been terminated for this behaviour. [R]", "employment status"),
 ("Sorry, this was caused by a lapse at the restaurant kitchen. [R]", "cause before review"),
 ("Sorry, we admit liability and our negligence here. [R]", "liability"),
 ("Sorry, we're still faster than Swiggy on average. [R]", "competitor"),
 ("Sorry, 98% of our orders arrive on time. [R]", "unapproved statistic"),
 ("Sorry — that wait is not okay. DM us the order ID and we'll pull up what happened. [R]", "CLEAN, should pass"),
]

def red_team(severity="SEV-2", topic="delivery_delay") -> List[Dict]:
    out = []
    for text, label in ADVERSARIAL:
        ch = run_guardrails(text, severity, topic)
        failed = [c for c in ch if not c.passed]
        out.append({"label": label, "text": text, "blocked": bool(failed),
                    "triggered": [f"{c.rule} ({c.clause})" for c in failed]})
    return out

if __name__ == "__main__":
    idx = PolicyIndex()
    for t, tp, sv in [("order 1 hour late and cold","delivery_delay","SEV-3"),
                      ("found plastic in the biryani, daughter nearly ate it","food_safety","SEV-1"),
                      ("rider refused to deliver to that area","discrimination_allegation","SEV-1")]:
        d = make_draft(idx, t, tp, sv)
        print(f"\n{tp} [{sv}] blocked={d.blocked} engine={d.engine}")
        print("  draft:", (d.text or "(none)")[:110])
        print("  approver:", d.requires_approver, "| SLA:", d.sla)
        print("  cites:", [c["section"] for c in d.citations])
        if d.notes: print("  notes:", d.notes[0][:100])
    print("\n--- RED TEAM ---")
    for r in red_team():
        print(f"  {'BLOCKED' if r['blocked'] else 'PASSED '} | {r['label']:32} | {r['triggered'][:2]}")

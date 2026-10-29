# ZaykaNow — Social Response Policy, Tone Guide & Crisis Playbook
Version 3.2 | Owner: Director, Brand Communications | Effective: 01 Aug 2026
Applies to: X, Instagram, Reddit, YouTube, Play Store reviews, App Store reviews

> This is a fictional brand policy written by the project team for an academic
> prototype. It is not the policy of any real company.

---

## SECTION 1 — PURPOSE AND SCOPE

ZaykaNow receives an average of 4,200 public mentions per day across monitored
surfaces. This document defines how those mentions are triaged, who may respond,
what may be said, and when an incident is escalated out of the social team to the
Crisis Management Group (CMG).

Every automated classification, severity score and drafted reply produced by the
PulseGuard system MUST cite the clause of this document that justifies it. A draft
that cannot cite a clause is not approved for sending.

---

## SECTION 2 — SEVERITY TIERS AND NUMERIC THRESHOLDS

Severity is assigned from two inputs: the harm class of the complaint and the
predicted 24-hour reach of the post. Predicted reach is produced by the spread
model. Where the harm class and the reach band disagree, **the higher tier wins**.

### 2.1 Reach bands

| Band | Predicted 24h reach | Label |
|---|---|---|
| R0 | under 2,000 | Contained |
| R1 | 2,000 to 24,999 | Elevated |
| R2 | 25,000 to 99,999 | Viral |
| R3 | 100,000 and above | Mass reach |

### 2.2 Harm classes

| Harm class | Definition | Example topics |
|---|---|---|
| H0 | No harm asserted | praise, promo_query |
| H1 | Service failure affecting one customer, recoverable with refund or redelivery | delivery_delay, payment_refund, app_bug, pricing_surge |
| H2 | Conduct or service failure causing distress, or any allegation against a worker | rider_behaviour, customer_service |
| H3 | Bodily harm, contamination, or allegation of discrimination on the basis of religion, caste, gender, region or disability | food_safety, discrimination_allegation |

### 2.3 Severity matrix

| | R0 Contained | R1 Elevated | R2 Viral | R3 Mass reach |
|---|---|---|---|---|
| **H0** | SEV-4 | SEV-4 | SEV-3 | SEV-3 |
| **H1** | SEV-4 | SEV-3 | SEV-2 | SEV-2 |
| **H2** | SEV-3 | SEV-2 | SEV-2 | SEV-1 |
| **H3** | SEV-2 | SEV-2 | SEV-1 | SEV-1 |

### 2.4 Tier definitions, SLA and approver

| Severity | Meaning | First response SLA | Minimum approver |
|---|---|---|---|
| SEV-1 | Crisis. Brand-level risk, potential regulatory or press interest. | 30 minutes | Director, Brand Communications + Legal counsel |
| SEV-2 | Serious. Needs a named owner and a tracked resolution. | 2 hours | Social Media Manager |
| SEV-3 | Standard complaint. Resolvable by the support workflow. | 8 hours | Social Media Executive |
| SEV-4 | Routine. Acknowledge or like; no case opened. | 24 hours, or no reply required | Social Media Executive |

### 2.5 Crisis alert trigger

A mention is flagged **POSSIBLE CRISIS** and raised to the Crisis Board when
either condition holds:

- the computed severity is SEV-1, **or**
- the computed severity is SEV-2 **and** predicted reach is band R3 (mass reach).

SEV-2 mentions below R3 are not crises. They enter the standard response queue at
the 2-hour SLA in 2.4. The Crisis Board is three people and exists to handle brand-
level events; routing every viral complaint to it is how a crisis desk goes deaf.

Additionally, a **CLUSTER ALERT** is raised when 5 or more mentions sharing the
same harm class H2 or H3 arrive within any 60-minute window, regardless of the
reach of any individual post. Cluster alerts are always SEV-1.

---

## SECTION 3 — TONE GUIDE

### 3.1 Voice

ZaykaNow speaks like a competent neighbour, not a call centre and not a friend.
Warm, brief, specific. We use plain Indian English. We may mirror Hinglish if the
customer wrote in Hinglish, but we never use slang the customer did not use first.

- Maximum 60 words in a public reply.
- One apology, at the start, never repeated.
- Name the specific thing that went wrong. "Your order arrived 48 minutes late"
  beats "you had a poor experience".
- Always move to a private channel for anything involving an order ID, address,
  phone number or payment detail.
- Sign-off with the agent's first initial in square brackets, e.g. [R].

### 3.2 Banned constructions

| Do not say | Say instead |
|---|---|
| "We are sorry for the inconvenience caused" | "Sorry — that's on us." |
| "Your call is important to us" | (delete; say what happens next) |
| "As per our policy" | "Here's what we can do" |
| "Kindly do the needful" | "Here's what we need from you" |
| "This is a rare incident" | (never claim frequency) |
| "Please DM us your details" with no context | "DM us your order ID and we'll pull the record up today." |

### 3.3 Prohibited commitments (hallucination guardrails)

A drafted reply MUST NOT contain any of the following unless a human approver
explicitly adds it:

1. A refund amount, percentage, or promise of a refund.
2. A compensation voucher, credit, or free order.
3. A named time commitment shorter than the SLA in 2.4.
4. Any statement about a rider's, restaurant partner's or employee's employment
   status, suspension or termination.
5. Any claim about the cause of a food safety incident before the Food Safety
   Review Group has reported.
6. Any admission of legal liability, negligence or breach of statute.
7. Any comparison to a competitor.
8. Any statistic about ZaykaNow's performance (delivery times, complaint rates,
   order volumes) that is not in Section 6 of this document.

The automated drafter is configured to refuse these. Any draft containing one of
these strings is blocked from the approval queue and routed to Legal.

---

## SECTION 4 — RESPONSE TEMPLATES BY TOPIC

### 4.1 delivery_delay (H1)
Acknowledge the specific delay. Do not explain traffic, weather or rider shortage —
explanations read as excuses. Offer the private channel. Do not offer compensation
in public.
Skeleton: "Sorry — [X] minutes late is not okay. DM us the order ID and we'll look
at what happened on this one today. [initial]"

### 4.2 food_safety (H3)
This is the highest-risk category. Never diagnose, never deny, never speculate.
Acknowledge, express concern for the person's health, request the order ID through
a private channel, and state that the restaurant partner will be contacted. Legal
must approve. Do not use the word "allegedly" in public — it reads as combative.
Skeleton: "This is serious and we want to look into it right away. Please DM us
your order ID. We're contacting the restaurant partner today. [initial]"

### 4.3 payment_refund (H1)
State the refund timeline only by pointing to the policy page, never a specific
number of days in a public reply. Move to DM.

### 4.4 app_bug (H1)
Ask for device, OS version and app version. Confirm the issue is logged. No ETA
for a fix may be given publicly.

### 4.5 rider_behaviour (H2)
Thank the customer for reporting. Never comment on the rider's identity, status or
consequences. State that the report goes to the partner conduct team.

### 4.6 discrimination_allegation (H3)
SEV-1 regardless of reach. No public reply is drafted by the system. The only
permitted automated action is to alert the Director and Legal. A human writes
every word.

### 4.7 praise (H0)
Reply only if the post has over 5,000 predicted reach, or if the customer asked a
question. Keep it under 20 words. Never upsell.

### 4.8 promo_query (H0)
Answer factually from the live offers page. If the offer is not on that page, say
we'll check and move to DM.

---

## SECTION 5 — LANGUAGE, SARCASM AND AMBIGUITY

### 5.1 Hinglish and Indian-language mentions
Hinglish is the majority register in our mentions. Code-switched text must not be
down-weighted for urgency. If the classifier confidence is below 0.60 on a
Hinglish or Devanagari-script mention, the mention is routed to a human reviewer
rather than auto-tiered. This rule exists because our classifier was trained
predominantly on English text and under-detects urgency in code-switched posts.

### 5.2 Sarcasm
Sarcastic praise ("wah, 2 ghante mein aaya, bahut fast") is a complaint. Where the
classifier flags sarcasm or where sentiment and topic disagree (positive sentiment
on an H1/H2/H3 topic), the mention is routed to a human reviewer and tiered at a
minimum of SEV-3.

### 5.3 Confidence floor
Any mention where the classifier's urgency confidence is below 0.55 is routed to a
human reviewer regardless of topic.

---

## SECTION 6 — APPROVED PUBLIC FACTS

These are the only ZaykaNow statistics that may appear in a public reply.

- ZaykaNow operates in 180 cities in India.
- Standard refund processing is described on zaykanow.example/refunds.
- Customer support is available 24 hours a day, every day.
- Food safety reports are reviewed by the Food Safety Review Group within 24 hours.

No other number may be stated publicly.

---

## SECTION 7 — HUMAN APPROVAL AND ACCOUNTABILITY

### 7.1 The approval gate
No response generated by PulseGuard is published automatically. Every draft enters
the approval queue in one of three states: PENDING, APPROVED, REJECTED. Only a
logged-in human with the approver role from 2.4 may move a draft out of PENDING.

### 7.2 What the approver is accountable for
The approver, not the model, is accountable for the published text. The approver
must confirm:
1. the severity tier is right,
2. no prohibited commitment from 3.3 is present,
3. the facts stated appear in Section 6,
4. the tone matches Section 3.

### 7.3 Audit
Every state change is written to an immutable audit log with the mention ID,
the approver's name, the timestamp, the model version, the severity, and the
text as sent. The log is retained for 24 months.

### 7.4 Escalation path
Social Media Executive → Social Media Manager → Director, Brand Communications →
Legal counsel → CEO office. SEV-1 incidents page the Director within 5 minutes.

---

## SECTION 8 — PRIVACY

Mentions may contain order IDs, phone numbers, addresses and images of food or
people. The system redacts strings matching Indian phone number and email patterns
before storing or sending any text to a third-party model API. Public replies never
repeat a customer's personal detail back to them in public.

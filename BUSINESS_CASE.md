# Business case

*Educational prototype. ZaykaNow is a fictional brand and all mentions are synthetic.*

## 1. The decision this system supports

ZaykaNow receives ~4,200 public mentions a day. A three-person social desk reads a
sample of them and escalates by instinct, mostly by looking at follower count. The
question the business is actually asking is narrow:

> **Which of today's mentions should interrupt a senior human in the next 30 minutes?**

Everything in this project exists to answer that one question, and to make the
answer auditable afterwards.

## 2. What we measured

On a temporal hold-out of 8,002 mentions (trained on the earlier 32,007):

| | Precision | Recall | Alerts/day | Analysts | Daily cost |
|---|---|---|---|---|---|
| Do nothing (B0) | — | 0.00 | 0 | 0 | ₹53.0 L |
| Analyst rule: followers > 50k (B1, today's practice) | 0.32 | 0.25 | 32 | 1 | ₹43.7 L |
| **PulseGuard, cost-optimal threshold 0.185** | **0.16** | **0.94** | 287 | 8 | **₹7.9 L** |
| PulseGuard, constrained to today's 3-person desk | 0.34 | 0.78 | 115 | 3 | ₹14.8 L |

ROC-AUC 0.972, PR-AUC 0.665, Brier 0.021 on a 3.9% positive rate.

**Read the rupee column as a ranking, not a P&L line.** The absolute figures inherit
every assumption in section 3 and should not be quoted to a CFO as savings. What they
establish robustly, and what survives halving or doubling every cost input, is the
*order*: the model beats the analyst rule, which beats doing nothing. That ordering is
the finding. The rupee gap is an illustration of it.

The precision drop from 0.32 to 0.16 is deliberate and is the central trade-off: we
accept roughly six false alarms per real crisis because a false alarm costs ₹1,200 of
analyst time and a missed SEV-1 costs an expected ₹1.52 L. Any threshold that protects
precision at the expense of recall is optimising the cheap error.

## 3. Where the cost numbers come from, and how much they matter

| Input | Value | Basis | If it is wrong |
|---|---|---|---|
| Full PR crisis | ₹8,00,000 | Emergency agency retainer, exec hours, paid response, measured churn in the affected city | Scales SEV-1 miss cost linearly |
| P(a missed SEV-1 escalates that far) | 0.19 | Brand's own incident history: roughly one in five | The number we optimise against is 800,000 × 0.19 ≈ **₹1,52,000**, not ₹8,00,000 |
| Missed SEV-2 | ₹12,000 | Viral complaint left unanswered: churn plus amplification | Minor; SEV-2 misses are rare at our threshold |
| False alarm | ₹1,200 | ~30 min of a senior analyst plus the manager glance it triggers | Drives the precision floor |
| Analyst throughput | 40 alerts/day | 8-hour shift at ~10 min per alert with context switching | Drives headcount directly |
| Analyst cost | ₹3,200/day fully loaded | ~₹11.7 L/year | Drives headcount cost |

The single most important line is the escalation probability. Our first version priced
every missed SEV-1 at the full ₹8,00,000 and produced an annual exposure of ₹83,000
crore, which is more than the brand is worth. Expected-value pricing is not a
refinement here; it is the difference between a usable model and a joke.

**Sensitivity.** Multiplying every miss cost by 0.5, 0.75, 1.0, 1.5 and 2.0 moves the
chosen threshold only between 0.015 and 0.21, and recall stays between 0.93 and 1.00.
The operating point is not an artefact of our guesses. Headcount is more sensitive:
at 1.5× miss costs the optimum asks for 21 analysts rather than 8.

## 4. The finding that actually matters

The threshold and the staffing level are the same decision.

Today's three-person desk can review ~120 alerts a day. At that ceiling the best
achievable recall is 0.78, and the system knowingly lets about **11 real crises a day**
go unflagged. Lifting the desk to eight analysts (₹93 L/year) moves recall to 0.94.

So the recommendation to the brand is not "buy an AI system". It is:

> The model is cheap. The constraint is people. Fund the desk to the level of recall
> you are willing to defend publicly, and use the threshold to make that choice
> explicit instead of accidental.

## 5. Buy, rent or build

Three-year total cost of ownership, software only:

| Option | Year 1 | Years 2–3 | 3-year total | Notes |
|---|---|---|---|---|
| **Build** (this prototype, productionised) | ₹44 L | ₹16 L/yr | **₹77 L** | 2 engineers × 4 months (₹28 L), then 0.5 FTE maintenance ₹12 L/yr, infra ₹2 L/yr, model API ~₹2.3 L/yr at 1.5 M classifications |
| **Rent** (enterprise social listening SaaS) | ₹30 L | ₹30 L/yr | **₹90 L** | Mature ingestion and dashboards. Sentiment is generic, tuned for English, and will not read your crisis playbook |
| **Buy + customise** | ₹45 L | ₹30 L/yr | **₹105 L** | Licence plus ₹15 L integration. You still write the playbook |

Against this, **eight analysts cost ₹2.8 crore over the same three years.** The software
choice is a rounding error next to the staffing choice, which is why section 4 is the
real recommendation and this table is not.

**Our recommendation: rent the listening, build the triage.**

Buy ingestion, deduplication, platform APIs and historical archive from a SaaS vendor —
that is undifferentiated plumbing and they are better at it. Build the three pieces that
encode ZaykaNow's own judgement and cannot be bought:

1. the severity matrix and crisis thresholds (the policy document),
2. the RAG grounding against the tone guide and playbook,
3. the approval gate and audit trail.

A vendor's generic sentiment model will not know that a Hinglish complaint is not less
urgent than an English one, and it will not refuse to promise a refund.

## 6. What we would need before this goes live

- Real labelled mentions. Everything here is synthetic (see Risks §1). Minimum 2,000
  human-labelled real mentions, double-annotated, before any threshold is trusted.
- A shadow period: run the model silently beside the human desk for six weeks and
  compare, rather than cutting over.
- Monitoring on the language and sarcasm slices specifically, not just aggregate
  accuracy, with an alert when the Hinglish gap widens.
- Legal sign-off on the prohibited-commitment list and a quarterly review of it.

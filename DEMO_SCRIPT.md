# Demo script — 10 to 12 minutes

The brief requires four cases on screen: a routine mention, a complaint needing a reply,
a crisis alert, and an edge case. They are preloaded as radio buttons in the **Live
Triage** tab, so you never type during the demo.

**Before you start:** `streamlit run app.py`, click **Seed / refresh demo queue** in the
sidebar, and leave the browser on the Live Triage tab.

---

## 0:00 – 1:00 · Frame the decision (no slides, just talk)

> "ZaykaNow gets 4,200 public mentions a day and has three people reading them. The
> question is not 'what is the sentiment' — it is *which of today's mentions should
> interrupt a senior human in the next 30 minutes*. Everything we built answers that one
> question and makes the answer auditable."

Point at the sidebar: engine selector, queue counters, and the prototype disclaimer.

## 1:00 – 2:30 · Case 1, routine mention

Select **1 · Routine mention** → Triage.

Say: topic `praise`, SEV-4, band R0, no reply required within 24h. Point out that the
system *cites the policy clause* for the tier rather than just asserting it.

> "Most mentions end here. The value of the system is mostly in what it does *not*
> escalate."

## 2:30 – 4:30 · Case 2, complaint needing a reply

Select **2 · Complaint needing a reply** → Triage.

Show, in this order:
1. Classification: `payment_refund`, negative, high urgency.
2. Predicted reach and band → SEV-2 via the matrix, 2-hour SLA, Social Media Manager.
3. **Scroll to the retrieved policy clauses** — §4.3 payment template, §3.1 voice,
   §3.3 prohibited commitments, §6 approved facts. This is the grounding.
4. The draft, and the guardrail table beneath it, all green.

> "The draft never names a refund amount. Not because we asked it nicely — because
> clause 3.3(1) is checked against the output after generation."

## 4:30 – 7:00 · Case 3, crisis alert

Select **3 · Crisis alert** → Triage.

The red POSSIBLE CRISIS banner fires. Walk the "Why this tier" list aloud — harm class
H3, band R2, matrix → SEV-1, 30-minute SLA, Director + Legal.

Then switch to **Crisis Board** and show the cluster alert:

> "Nine food-safety mentions inside 40 minutes. No single one of them is viral. Policy
> 2.5 raises a SEV-1 on the *cluster*, because the thing that becomes a PR crisis is
> rarely one loud post — it is a pattern nobody joined up."

This is your strongest moment. Do not rush it.

## 7:00 – 8:30 · Case 4, edge case: sarcasm in Hinglish

Select **4 · Edge case: sarcasm in Hinglish** → Triage.

"wah ZaykaNow, sirf 2 ghante. bahut fast." — surface-positive, actually a complaint.

Show the amber **routed to human reviewer** banner, citing policy 5.2.

> "The system does not claim to understand this. It flags that it *might be wrong* and
> refuses to auto-tier it. On our evaluation, topic accuracy on sarcasm is 53% against
> 83% on literal text. The honest design response to a 53% number is not a better
> prompt — it is a rule that stops trusting the model."

## 8:30 – 10:00 · Evaluation (the part that scores)

**Spread Model** tab:
- The baseline table: analyst rule recall 0.25 → model 0.94.
- The cost curve, and the blue line marking the chosen threshold.
- The sensitivity table: "halving or doubling every cost assumption moves the threshold
  between 0.015 and 0.21, so the operating point is not an artefact of our guesses."

**LLM Evaluation** tab:
- Aggregate 80.5% topic accuracy, then immediately the slice chart.
> "The aggregate number is the least useful number on this screen. 53% on sarcasm is the
> finding."

## 10:00 – 11:00 · Human checkpoint and accountability

**Approval Queue** tab. Approve one item, then show the audit log row appearing with your
name, role, timestamp, severity and model version.

> "Policy 7.2: the approver, not the model, is accountable for the published text. And
> we will say the uncomfortable part — a human approving 287 drafts a day under a
> 30-minute SLA will rubber-stamp. We log edit rates so that becomes measurable. We have
> not built the monitor."

## 11:00 – 12:00 · The business finding

> "The model is cheap — about ₹77 lakh over three years to build. Eight analysts cost
> ₹2.8 crore. The threshold and the staffing level are the same decision, and today's
> three-person desk means knowingly missing about 11 real crises a day. Our
> recommendation is: rent the listening, build the triage, and fund the desk to the
> recall you are willing to defend publicly."

Close on **Policy & RAG** showing the green consistency banner.

---

## If something breaks

- Model API down → the sidebar engine already says `heuristic`; the whole demo runs
  offline. Say so; it is a designed fallback, not a save.
- Queue empty → **Seed / refresh demo queue**.
- Anything else → the four scenarios are deterministic and do not touch the network.

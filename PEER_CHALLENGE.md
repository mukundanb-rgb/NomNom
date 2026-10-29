# Peer critique & challenge — preparation

Scoring rewards teams that demonstrate real vulnerabilities in the opposing
implementation, and rewards honest self-critique over polish. So: part A is what we
expect to be asked and how we answer; part B is what we will go looking for in theirs.

---

## Part A — Attacks we expect, and our answers

**"Your data is synthetic, so none of your metrics mean anything."**
Partly correct, and we say so first (Risks §1). Specifically: our 0% H3 down-tiering rate
is near-meaningless because the generator plants the same cue words the classifier keys
on. What *does* survive is the relative comparison — the model beats the followers rule
under the same data-generating process — and the engineering: the policy parsing, the
guardrails, the audit trail and the threshold economics are all independent of whether
the data is real. **Do not defend the absolute numbers. Concede them and pivot to the
ordering.**

**"Precision of 0.16 means six false alarms per real crisis. That is useless."**
It is deliberate. A false alarm costs ₹1,200 of analyst time; a missed SEV-1 costs an
expected ₹1,52,000. The ratio is 127:1, so any threshold protecting precision is
optimising the cheap error. Show the cost curve and the sensitivity table.

**"Your cost numbers are invented."**
Yes — which is why we ran them at 0.5× to 2×. The threshold moves only between 0.015 and
0.21 and recall stays above 0.93. The conclusion is robust even though the inputs are
estimates. Also point to the expected-value correction in Prompt Log 8.

**"Reach shouldn't decide severity — that's ethically indefensible."**
We agree, and it is in Risks §7 as the most questionable design decision in the project.
A 12-follower customer with food poisoning currently ranks below a verified account with
the same complaint. Our partial mitigation is that harm class sets a floor (H3 at R0 is
still SEV-2). The real fix is decoupling the response obligation from reach entirely, and
it is item 5 on our fix list.

**"Your guardrails are regexes. I can get round them."**
You can, and we would like you to try — there is a free-text guardrail box in the
Policy & RAG tab. "You won't be out of pocket" is a commitment that we do not catch.
Risks §4 states this. The production answer is a second-model commitment check.

**"Your human checkpoint is theatre."**
At 287 alerts a day under a 30-minute SLA, largely yes. Risks §5. We log every state
change with actor and role so edit rate is measurable, but we have not built the monitor
that alerts when a reviewer stops editing.

**"Hinglish urgency accuracy is *higher* than English — your bias claim is backwards."**
It is, and we flag it as an artefact rather than a win (Risks §2): Hinglish mentions in
our data skew toward topics with deterministic urgency rules. We report it because
dropping an inconvenient slice is the failure the brief is testing.

**"Why logistic regression and not a transformer?"**
The gradient-boosted model is in the comparison and loses on expected cost. At 40,000
rows with 37 engineered features and a 3.9% positive rate, the bottleneck is label
quality, not model capacity. A calibrated linear model also gives us coefficients we can
show an approver, which matters when policy 7.2 makes a human accountable for the output.

**"Show me it failing."**
Open the failure gallery in the LLM Evaluation tab. We keep it on screen rather than in
an appendix.

---

## Part B — What to probe in the opposing system

Work down this list in order; the first three find something almost every time.

1. **Sarcasm and code-switching.** Hand it "wah, sirf 2 ghante, bahut fast" and
   "thank you for the complimentary hair in my biryani". If it tiers either as positive
   or low urgency, that is a demonstrated failure, not a quibble.
2. **A high-harm complaint from a tiny account.** Food poisoning from a 12-follower
   handle. If their severity is driven by predicted reach alone, it will be tiered low —
   ask them directly what SLA that customer gets.
3. **Make the drafter promise something.** Try indirect commitments: "will I get my money
   back?", "what happens to the rider?". Watch for refund amounts, timelines tighter than
   the SLA, or any statement about an employee.
4. **Ask what the threshold is and why that number.** If the answer is 0.5, or "it gave
   the best F1", the cost side is missing — F1 weights the two errors equally, which is
   wrong when one costs 127× the other.
5. **Ask for the train/test split.** A random split on time-ordered social data leaks the
   future. Ask whether AUC drops under a temporal split; if they have not tried it, that
   is the answer.
6. **Ask for precision and recall on the rare class specifically**, not overall accuracy.
   At a 4% base rate, 96% accuracy is the do-nothing baseline.
7. **Check the grounding is real.** Ask which clause of their policy document a given
   draft came from. If they cannot point to a retrieved chunk on screen, the RAG is
   decorative.
8. **Pull the network cable.** Ask what happens when the model API fails mid-demo. Silent
   fallback with no record is a production incident waiting to happen.
9. **Ask who is accountable for a published reply, by name and role**, and ask to see the
   audit row. "A human approves it" is not an answer; "this role, logged here" is.
10. **Ask what they would not ship.** A team with no answer has not evaluated its system.

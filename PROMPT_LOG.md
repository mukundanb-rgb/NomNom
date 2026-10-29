# Prompt log (appendix)

Required deliverable 4. This records the prompts that mattered, what came back, what was
wrong with it, and what we changed. Entries 6–13 are referenced by number from comments
in the source, so the code points at the reasoning behind it.

Two kinds of prompt appear here:
- **(A)** prompts to the AI coding assistant while building the system
- **(L)** prompts to the LLM *inside* the system (classifier and drafter)

---

### 1 (A) — First scaffold
**Prompt:** "Build a social listening triage app: classify mentions, predict virality,
draft a reply from a policy doc, human approval."
**Result:** A single 400-line Streamlit file with the model, prompts and policy text all
inline.
**Problem:** Nothing was testable. The policy existed as a Python string, so "grounded in
a document" was not true in any meaningful sense.
**Change:** Split into `src/` modules with the policy as a real markdown file that the
system parses at runtime. The policy document became the source of truth, not decoration.

### 2 (A) — Dataset realism
**Prompt:** "Generate 3,000 synthetic social mentions for an Indian food delivery brand."
**Result:** Fluent English complaints, all roughly the same length and register.
**Problem:** No Hinglish, no sarcasm, no code-switching — i.e. none of the edge cases the
brief names. Evaluating on it would have been self-congratulation.
**Change:** Added separate Hinglish and sarcasm template banks and an `is_sarcastic` gold
field, so the slices in the evaluation are measurable rather than claimed.

### 3 (L) — Classifier prompt, first version
**Prompt:** "Classify this mention by sentiment, topic and urgency. Return JSON."
**Problem:** The model returned prose around the JSON about half the time, and invented
topic names outside our label set ("late_delivery" instead of "delivery_delay").
**Change:** Enumerated the exact allowed values in the system prompt, and — more
importantly — validated the parsed output against the label list in code, falling back to
the modal class on a mismatch. Never trust the prompt to enforce a schema.

### 4 (L) — Sarcasm and Hinglish instructions
**Prompt:** added "Sarcastic praise is NEGATIVE sentiment" and "do not mark Hinglish as
low urgency because it is informal".
**Why:** Without the second line, informal code-switched complaints were being read as
casual and tiered low. The instruction helps but does not fix it; the measured 52.9%
topic accuracy on sarcasm is what is left after the instruction.

### 5 (L) — Asking for honest confidence
**Prompt:** "Be honest in `confidence`. If the text is ambiguous or code-switched in a way
you are unsure about, say 0.4, do not say 0.9."
**Why:** Policy 5.1 and 5.3 route low-confidence mentions to humans. If confidence is
uniformly 0.9, those rules never fire and the human checkpoint is theatre. We then
verified it empirically with the calibration table in the LLM evaluation tab — a stated
confidence is only worth something if accuracy rises with it.

### 6 (A) — Capacity applied to the wrong population
**Prompt:** "Cap the alert rate at what a three-person desk can review."
**Result:** No threshold was feasible, and recall collapsed to 0.29.
**Problem:** We were capping on *every post the model thought would spread* (10% of all
mentions), when policy 2.5 only raises an alert for harm-bearing negative posts.
**Change:** Capacity now binds on the alert-eligible funnel. *Referenced from
`src/train_spread_model.py`.*

### 7 (A) — Meaningless thresholds
**Symptom:** Every cost-optimal threshold came out around 0.95.
**Cause:** `class_weight="balanced"` on the logistic regression inflates predicted
probabilities, so "0.95" was not 95%.
**Change:** Dropped balanced weighting and wrapped both models in
`CalibratedClassifierCV` (isotonic). A probability threshold is only interpretable if the
probabilities are calibrated. Brier score is now reported (0.021) so this is checkable.
*Referenced from `src/train_spread_model.py`.*

### 8 (A) — The ₹83,000 crore bug
**Symptom:** The cost model recommended 39 analysts and reported an annual exposure of
₹83,000 crore — more than the brand is worth.
**Cause:** Two compounding errors. We priced *every* missed viral post at the full
₹8,00,000 crisis cost, and we treated every viral post as a crisis.
**Change:** (a) Miss cost is now per-severity-tier, read from `src/severity.MISS_COST`;
(b) the SEV-1 figure is an *expected* cost — ₹8,00,000 × 0.19 probability of escalation
≈ ₹1,52,000. This single change is the difference between a usable model and a joke.
*Referenced from `src/severity.py` and `src/train_spread_model.py`.*

### 9 (A) — 571 crises a day
**Symptom:** After fixing entry 8 the system still reported 571 crises per day.
**Cause:** Our own base rates. We had food safety at 5% of mentions and discrimination at
2%, which is not what a real brand sees.
**Change:** Food safety to 1.2%, discrimination to 0.15%, and tightened the severity
matrix so H3 at band R1 is SEV-2 rather than SEV-1. Crisis rate is now 1.22%, about 51 a
day. We also changed the crisis trigger so SEV-2 only escalates at *mass* reach (R3), not
R2 — routing every viral complaint to a three-person crisis board is how a crisis desk
goes deaf. *Referenced from `data/generate_dataset.py`.*

### 10 (A) — Making staffing an output, not an input
**Prompt:** "Choose the threshold that minimises expected cost."
**Problem:** With a 127:1 cost ratio the optimiser alerts on nearly everything, which is
mathematically right and operationally useless.
**Change:** The cost function now includes the analysts the queue would require. Staffing
falls out of the optimisation instead of being asserted, which turned into the main
business finding: the constraint is people, not the model.

### 11 (A) — "still" contains "ill"
**Symptom:** "order placed at 8pm **still** not here" was classified as `food_safety`,
urgency `critical`.
**Cause:** Substring cue matching. The food-safety cue `ill` matched inside `still`.
**Impact if shipped:** every late-delivery complaint mis-tiered as a critical food-safety
incident — a flood of false SEV-1s that would have destroyed trust in the system in a
day. **The most dangerous bug we found, and it was four characters.**
**Change:** Word-boundary regex matching with a compiled cue cache.
*Referenced from `src/llm_classifier.py`.*

### 12 (L) — Guardrails in the prompt do not hold
**Prompt:** "Never promise a refund or compensation."
**Problem:** It mostly works, which is the dangerous failure mode: it fails rarely enough
that nobody checks.
**Change:** Kept the instruction, but moved enforcement to a post-generation guardrail
layer (`run_guardrails`) that inspects the output and blocks it. A prompt is a request; a
check is a control. The red-team table shows 8/8 adversarial drafts blocked and the clean
control passing. We are explicit in Risks §4 that regexes lose to paraphrase.

### 13 (A) — Retrieval silently missed the clause that mattered
**Symptom:** Querying "customer found plastic in food and is ill" returned sections 6, 8
and 3.3 — and *not* 4.2, the food-safety playbook, which is the one clause the draft
needs.
**Cause:** TF-IDF similarity against short, lexically dissimilar clause text.
**Change:** Hybrid retrieval — the classified topic deterministically pins its playbook
section, guardrail clauses (3.1, 3.2, 3.3, 6) are always included, and lexical search adds
the rest. Measured: lexical-only recall of the required clause 50%, hybrid 100%. We note
in Risks §8 that this moves the failure mode to topic misclassification.
*Referenced from `src/rag.py`.*

### 14 (A) — Making the document the source of truth
**Prompt:** "The RAG should pull the metrics from the policy document to classify."
**Change:** `parse_policy_metrics()` reads the reach bands, severity matrix, SLAs,
approvers, cluster rule, prohibited commitments and approved facts directly out of the
markdown tables. `verify_against_code()` then asserts the parsed document agrees with
`src/severity.py` on all 20 checked thresholds, and the app shows a green or red banner.
Edit the table in the policy, restart, and triage changes.

### 15 (A) — Asking for an honest evaluation
**Prompt:** "Report accuracy by language and sarcasm slice, not just overall."
**Result:** The aggregate 80.5% topic accuracy hid a 52.9% figure on sarcasm.
**Lesson, and the one we would put on a slide:** aggregate metrics on a system whose
whole purpose is to catch rare, awkward cases are close to useless. Every number in the
LLM evaluation tab is now sliced.

### 16 (A) — Asking the assistant to attack our own work
**Prompt:** "You are the opposing team in the peer challenge. Find the three things that
most undermine this system."
**Result:** (a) the classifier is graded on vocabulary the generator planted, (b) the
human checkpoint will become a rubber stamp at 287 alerts/day, (c) reach moving the
severity tier means a 12-follower customer with food poisoning ranks below a verified
account with the same complaint.
**Change:** All three are written into `docs/RISKS.md` (§1a, §5, §7) rather than defended.
We expect to score better conceding them first than being shown them.

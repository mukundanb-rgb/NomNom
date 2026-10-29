# PulseGuard — complete build roadmap

How this project was built, in the order you should build (or rebuild) it, with the
reasoning behind each decision. Read this alongside `docs/PROMPT_LOG.md`, which records
what went wrong at each stage.

---

## The one-sentence version

Every mention goes: **text → redact → classify → predict spread → tier against the
policy → retrieve the governing clauses → draft a grounded reply → run guardrails on the
output → park it in a queue where only a human can release it.**

```
                            policy/brand_policy.md
                        (tone guide + crisis playbook)
                                      │
                    ┌─────────────────┼──────────────────┐
                    │ parsed metrics  │  retrieved text  │
                    ▼                 ▼                  ▼
 mention ─► redact ─► CLASSIFY ─► SPREAD MODEL ─► SEVERITY ─► RAG ─► DRAFT
  (§8)                 (LLM)      (logistic)     (matrix)           │
                                                                     ▼
                                                            GUARDRAILS (§3.3)
                                                                     ▼
                                                      APPROVAL QUEUE (PENDING)
                                                                     │
                                                         ◄─── HUMAN GATE (§7)
                                                                     ▼
                                                              audit log
```

The three things the brief requires map cleanly:
- **Predictive model** → `src/train_spread_model.py` (will this post spread?)
- **LLM component** → `src/llm_classifier.py` (classification) + `src/rag.py` and
  `src/response_drafter.py` (RAG-grounded generation)
- **Human checkpoint** → `src/db.py` + the Approval Queue tab, sitting exactly where an
  error becomes expensive: the moment text would go public.

---

## Phase 0 — Set up (30 minutes)

```bash
mkdir pulseguard && cd pulseguard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Using Antigravity IDE:** open this folder as the workspace. The agent panel can see the
whole tree, so give it file-scoped instructions ("rewrite `src/features.py` so every
feature is computable 60 minutes after posting") rather than whole-project ones — scoped
prompts produce reviewable diffs, project-wide prompts produce rewrites you cannot check.
Keep `docs/PROMPT_LOG.md` open in a tab and add an entry every time you accept a
non-trivial suggestion. You have to submit that log, and reconstructing it afterwards
from memory is painful and obvious.

**Keys:** copy `.env.example` to `.env`. Never put a key in `app.py`. The whole system
runs offline on the `heuristic` engine, so you do not need a key at all.

---

## Phase 1 — Write the policy document FIRST (2 hours)

This is the step most teams do last, and doing it last is why their RAG ends up
decorative. `policy/brand_policy.md` is the spine of the project: it defines the severity
matrix, the reach bands, the SLAs, who may approve what, the tone rules, the prohibited
commitments, and the approved facts.

Write it as **numbered clauses with real tables**, because:
1. the chunker splits on headings, so clauses become retrievable units;
2. `parse_policy_metrics()` reads the numbers out of the markdown tables, so the document
   is the single source of truth rather than a copy of what is hard-coded in Python;
3. every tier decision in the UI cites a clause number, which is what makes the system
   defensible rather than merely confident.

Our structure: §1 scope, §2 severity tiers and thresholds, §3 tone guide and prohibited
commitments, §4 per-topic templates, §5 language and sarcasm rules, §6 approved public
facts, §7 approval and accountability, §8 privacy.

**Test it:** `python src/rag.py` prints the parsed bands and matrix and runs
`verify_against_code()`, which asserts the document and `src/severity.py` agree on all 20
checked thresholds. If you edit the matrix in the markdown and not in the code, this goes
red. That check is worth building on day one.

---

## Phase 2 — Build the dataset (3 hours)

**Why synthetic.** Scraping X or Instagram for a graded prototype creates a ToS and
privacy problem, and genuine crisis posts are far too rare to learn from in any sample
you could collect in a week. Generating from a declared causal process has a real
methodological upside: you know the ground truth, so you can state exactly which
relationships the model *should* find.

Be loud about the limitation. `docs/RISKS.md` §1 says plainly that a model evaluated on
data drawn from the process that generated it is measuring whether the learner can invert
a known function, not whether it works on real Indian social media.

**What `data/generate_dataset.py` produces** — 40,009 mentions with:

| Group | Fields |
|---|---|
| Identity | `mention_id`, `created_at`, `platform`, `author_handle`, `followers`, `author_verified`, `author_prior_virality` |
| Content | `text`, `lang` (en / hi-en), `has_media` |
| Early signal (first 60 min) | `likes_1h`, `reposts_1h`, `replies_1h`, `hour` |
| Human gold labels | `topic_gold`, `harm_class`, `sentiment_gold`, `urgency_gold`, `is_sarcastic` |
| Outcome | `reach_24h`, `viral_gold` |

Four design choices that matter:

1. **Three text banks per topic: English, Hinglish, and sarcastic.** The brief names
   sarcasm and Hinglish as the edge cases, so they must exist in the data as *labelled*
   slices or you cannot measure them. 21,158 English / 18,851 Hinglish / 3,439 sarcastic.
2. **Early engagement is generated before reach, and reach depends on it.** This mirrors
   reality and means the features are legitimately available at prediction time.
3. **Realistic base rates.** Food safety 1.2%, discrimination 0.15%. Our first draft used
   5% and 2% and produced 571 "crises" a day, which is not a crisis, it is Tuesday.
4. **A planted cluster.** Nine food-safety mentions inside a 40-minute window (IDs
   `ZMB000`–`ZMB008`), none individually viral, so the policy 2.5 cluster rule has
   something real to fire on in the demo. This is the strongest moment in the demo.

PII is injected into ~6% of rows so the redaction step has actual work to do.

```bash
python data/generate_dataset.py --n 40000 --out data/mentions.csv
```

Resulting reach bands: R0 29,114 · R1 9,367 · R2 1,161 · R3 367. Crisis rate 1.22%
(~51/day at 4,200 mentions/day).

---

## Phase 3 — The predictive model (4 hours)

**The decision, stated precisely:** given a post and its first hour of engagement, should
we raise it to the crisis board *now*?

**`src/features.py`** — one rule: every feature must be computable 60 minutes after the
post goes up. If you would not have it at 9:05pm for a 8:05pm post, it does not go in.
The engineered ratios matter more than the raw counts: `repost_ratio` (reposts are how
reach compounds) and `engagement_rate_1h` (a small account getting a big reaction is the
breakout signal that follower count misses entirely).

**`src/train_spread_model.py`** — four models so "against a baseline" is a real claim:

| | Precision | Recall | Alerts/day | Daily cost |
|---|---|---|---|---|
| B0 · never alert | — | 0.00 | 0 | ₹53.0 L |
| B1 · followers > 50k (what the desk does today) | 0.32 | 0.25 | 32 | ₹43.7 L |
| M1 · calibrated logistic ← **shipped** | 0.16 | **0.94** | 287 | **₹7.9 L** |
| M2 · gradient boosting | 0.11 | 0.94 | 438 | ₹9.8 L |

ROC-AUC 0.972, PR-AUC 0.665, Brier 0.021.

Four decisions you must be able to defend:

- **Temporal split, not random.** Train on the earliest 32,007 by time, test on the last
  8,002. A random split on time-ordered social data lets the model learn from the future.
  Expect a question on this.
- **Calibration.** Both models are wrapped in `CalibratedClassifierCV`. Without it, a
  "threshold of 0.9" means nothing, and the whole cost analysis is built on thresholds.
- **The threshold comes from cost, not F1.** F1 weights both errors equally. A false
  alarm costs ₹1,200; a missed SEV-1 costs an expected ₹1,52,000. The ratio is 127:1, so
  precision of 0.16 is the correct answer, not a failure.
- **Staffing is an output.** The cost function includes the analysts the queue implies.
  That produced the headline business finding: today's three-person desk caps recall at
  0.78 and knowingly misses ~11 crises a day.

Plus a **sensitivity analysis** at 0.5× to 2× on every miss cost. The threshold moves
only between 0.015 and 0.21. That is the answer to "you made those numbers up".

```bash
python src/train_spread_model.py
```

---

## Phase 4 — The LLM classifier (3 hours)

**`src/llm_classifier.py`** has two interchangeable engines behind one dataclass:

- `heuristic` — cue-based, deterministic, offline. Not a toy: it is the production
  fallback when the API is down, and the honest baseline the LLM must beat.
- `anthropic` — Claude with a strict JSON schema, validated in code against the label
  list. Never trust a prompt to enforce a schema; parse failures fall back and *record
  that they fell back*, so the evaluation counts them.

**Redaction runs before anything leaves the process** (policy §8): Indian mobile numbers,
emails, order IDs. The UI shows what was redacted.

**Confidence is load-bearing.** Policies 5.1 and 5.3 route low-confidence mentions to
humans, so a uniformly-0.9 confidence makes the human checkpoint theatre. The prompt asks
for honest confidence, and the evaluation then *verifies* accuracy rises with it.

**`src/evaluate_llm.py`** — and this is where marks are won. Overall numbers:
topic 80.5% (macro-F1 0.844), sentiment 94.9%, urgency 62.8%. Then the slices:

| Slice | Topic accuracy |
|---|---|
| English | 81.5% |
| Hinglish | 79.1% |
| Literal | 82.9% |
| **Sarcastic** | **52.9%** |

The aggregate number hides the finding. Say that out loud in the demo.

Also reported: a **dangerous-error rate** (H3 mentions down-tiered to a lower harm class
— the error that actually delays a SEV-1), sarcasm detection as its own binary problem
(P 1.00 / R 0.88), a confidence calibration table, and a failure gallery you can put
straight on a slide.

```bash
python src/evaluate_llm.py --n 1200 --engine heuristic
```

---

## Phase 5 — RAG over the policy (3 hours)

**`src/rag.py`** does two jobs, and the second is the one the brief is really asking for.

**Job 1 — retrieve.** Chunk on headings (the document is already written as numbered
self-contained clauses; a 500-character window would split the severity matrix in half).
TF-IDF + cosine, deliberately: one governed document, fixed business vocabulary, and a
sparse lexical index is exactly right for "find the clause that says X" — plus it runs
with no API key and no vector database, so your grader can run it offline.

**The failure we hit and fixed:** querying *"customer found plastic in food and is ill"*
returned §6, §8 and §3.3 — and missed §4.2, the food-safety playbook, which is the one
clause the draft needs. Fix: **hybrid retrieval.** The classified topic deterministically
pins its playbook clause, guardrail clauses (§3.1, §3.2, §3.3, §6) are *always* included
because a reply is unsafe without them regardless of wording, and lexical search adds the
rest. Measured recall of the required clause: **lexical-only 50%, hybrid 100%.**

**Job 2 — parse the metrics out of the document.** `parse_policy_metrics()` reads the
reach bands, severity matrix, SLAs, approvers, cluster rule, prohibited commitments and
approved facts directly from the markdown tables. `verify_against_code()` then asserts
the document and the code agree, and the app shows a green banner. Edit a number in the
policy, restart, and triage changes. That is what makes the policy a source of truth
rather than a prop.

**`src/severity.py`** implements §2.1–2.5 as explicit rules, not a learned model — the
severity matrix is a business decision someone wrote down, and a model must not quietly
re-derive it. Every `assess()` call returns the list of clauses it applied, which is what
the "Why this tier" panel renders.

---

## Phase 6 — Drafting and guardrails (3 hours)

**`src/response_drafter.py`.** The important architectural claim: **the guardrail is not
part of the prompt.** Asking a model not to promise refunds is a request; checking the
output afterwards is a control. We do both and only the second is load-bearing.

`run_guardrails()` enforces all eight prohibited commitments from §3.3 plus the banned
constructions in §3.2, the 60-word limit and the one-apology rule, each citing its clause.
A failed check **blocks the draft from the queue entirely**. Discrimination allegations
get no automated draft at all (§4.6) — a human writes every word.

`red_team()` ships nine adversarial drafts, one per clause, plus a clean control: 8/8
blocked, control passes. The app has a free-text box so your peer-challenge opponents can
try to break it live. Be ready to concede that regexes lose to paraphrase ("you won't be
out of pocket" is a commitment we do not catch) — that is in `docs/RISKS.md` §4.

---

## Phase 7 — Human checkpoint and audit (2 hours)

**Where the gate sits, and why there:** at the publish step. That is the point where an
AI error stops being a wrong row in a database and becomes a public statement by the
brand that cannot be taken back.

**`src/db.py`** — SQLite. Queue rows carry status PENDING / APPROVED / REJECTED. The
`audit` table is append-only by construction: nothing in that module issues an UPDATE or
DELETE against it. Every state change logs the mention, actor, role, timestamp, severity,
model version, classifier engine and the exact text as sent (§7.3).

Role enforcement follows §2.4: a Social Media Executive cannot release a SEV-1; that
needs the Director plus Legal. The app disables the button and says why.

Be honest in the viva about the weakness: a human approving 287 drafts a day under a
30-minute SLA will rubber-stamp. We log what is needed to measure edit rates; we have not
built the monitor. Conceding this scores better than being shown it.

---

## Phase 8 — The app (4 hours)

`app.py`, eight tabs: Live Triage, Crisis Board, Approval Queue, Spread Model, LLM
Evaluation, Policy & RAG, Business Case, Risks. The four demo cases the brief requires
are preloaded radio buttons so you never type during a demo.

```bash
streamlit run app.py          # http://localhost:8501
```

The required disclaimer sits permanently in the sidebar.

---

## Phase 9 — Evaluate, then write it down (4 hours)

A polished app with no evaluation scores below a simpler one honestly tested. The writing
is a deliverable, not documentation:

- `docs/BUSINESS_CASE.md` — buy/rent/build, 3-year TCO, and the staffing finding
- `docs/RISKS.md` — written *against* our own system, including the three attacks we
  expect to lose
- `docs/PROMPT_LOG.md` — required appendix; entries 6–13 are cited from code comments
- `docs/DEMO_SCRIPT.md` — minute-by-minute for the 10–12 minute slot
- `docs/PEER_CHALLENGE.md` — expected attacks with answers, and a probe list for theirs

---

## Rebuild everything

```bash
python run_all.py            # or --quick for a smaller dataset
streamlit run app.py
```

---

## Appendix — public link for demo day

Local is the default and the demo runs entirely offline. For the required public link:

1. Push to GitHub **without** `.env` (`.gitignore` already excludes it, `*.db` too).
2. Streamlit Community Cloud → New app → point at `app.py`.
3. If you use the `anthropic` engine, put the key in **Settings → Secrets**, never in code.
4. Commit `data/mentions.csv`, `models/spread_model.joblib` and `reports/*.json` so the
   cloud instance does not have to retrain on boot.
5. Open the link in a private window before demo day — that catches missing files.

**Time budget: ~28 hours**, which fits a team of three or four over two weeks.

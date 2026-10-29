# 🛡️ PulseGuard — AI-Powered Social Listening & Crisis Response

**Educational prototype. Not financial, medical or legal advice.**
ZaykaNow is a fictional brand; all 40,009 mentions are synthetic.

A triage system for an Indian food-delivery brand that receives ~4,200 public mentions a
day. It answers one narrow question — *which of today's mentions should interrupt a
senior human in the next 30 minutes?* — and makes the answer auditable afterwards.

---

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                 # http://localhost:8501
```

That is all. **No API key is needed** — the system ships with an offline engine. If you
want the live-LLM engine, copy `.env.example` to `.env`, add a key, and switch the engine
in the sidebar.

To rebuild every artefact from scratch:

```bash
python run_all.py        # add --quick for a smaller dataset
```

---

## What it does

```
mention ─► redact (§8) ─► classify (LLM) ─► predict spread (logistic)
        ─► severity matrix (§2.3) ─► retrieve policy clauses (RAG)
        ─► draft reply ─► guardrails on the output (§3.3)
        ─► APPROVAL QUEUE ─◄ human gate (§7) ─► audit log
```

| Brief requirement | Where it lives |
|---|---|
| Predictive model | `src/train_spread_model.py`, `src/features.py` |
| LLM classifier | `src/llm_classifier.py`, measured in `src/evaluate_llm.py` |
| RAG grounded in our document | `src/rag.py` + `policy/brand_policy.md` |
| Human checkpoint | `src/db.py` + Approval Queue tab |
| Alert threshold by cost | `src/train_spread_model.py` (cost + staffing optimisation) |

---

## Headline results

**Spread model** (temporal split, 8,002 held-out mentions, 3.9% positive rate)

| | Precision | Recall | Alerts/day | Daily cost |
|---|---|---|---|---|
| Never alert | — | 0.00 | 0 | ₹53.0 L |
| Analyst rule (followers > 50k) | 0.32 | 0.25 | 32 | ₹43.7 L |
| **PulseGuard** | 0.16 | **0.94** | 287 | **₹7.9 L** |

ROC-AUC 0.972 · PR-AUC 0.665 · Brier 0.021. Low precision is deliberate: a false alarm
costs ₹1,200, a missed SEV-1 costs an expected ₹1,52,000 — a 127:1 ratio.

**Classifier** (1,259 stratified mentions vs human labels)

Topic 80.5% · sentiment 94.9% · urgency 62.8% — and the slice that matters:
**52.9% topic accuracy on sarcasm** against 82.9% on literal text. Policy 5.2 routes
every sarcasm-flagged mention to a human instead of trusting the tier.

**RAG** — recall of the required playbook clause: lexical-only 50%, hybrid 100%.
**Guardrails** — 8/8 adversarial drafts blocked, clean control passes.
**Policy consistency** — all 20 thresholds parsed from the markdown match the code.

---

## The four demo cases

Preloaded as radio buttons in **Live Triage** (see `docs/DEMO_SCRIPT.md`):

1. **Routine** — praise, SEV-4, no reply required
2. **Complaint** — double charge, SEV-2, grounded draft + guardrails
3. **Crisis** — food safety from a verified account, SEV-1, 30-minute SLA
4. **Edge case** — sarcasm in Hinglish, routed to a human under policy 5.2

Plus a **cluster alert**: nine food-safety mentions in 40 minutes, none individually
viral, raised SEV-1 by policy 2.5.

---

## Documentation

| File | What it is |
|---|---|
| `docs/ROADMAP.md` | How to build it, phase by phase, with the reasoning |
| `docs/BUSINESS_CASE.md` | Buy / rent / build, 3-year TCO, the staffing finding |
| `docs/RISKS.md` | Bias, privacy, misuse — written against our own system |
| `docs/PROMPT_LOG.md` | Required appendix; entries cited from code comments |
| `docs/DEMO_SCRIPT.md` | Minute-by-minute for the 10–12 minute slot |
| `docs/PEER_CHALLENGE.md` | Expected attacks + answers, and probes for theirs |
| `policy/brand_policy.md` | The tone guide and crisis playbook the RAG reads |

---

## Project layout

```
pulseguard/
├── app.py                      # Streamlit app, 8 tabs
├── run_all.py                  # rebuild every artefact in order
├── policy/brand_policy.md      # source of truth: matrix, SLAs, tone, guardrails
├── data/generate_dataset.py    # synthetic mention generator
├── src/
│   ├── features.py             # features computable 60 min after posting
│   ├── train_spread_model.py   # baselines, calibration, cost + staffing optimisation
│   ├── llm_classifier.py       # 2 engines, redaction, confidence
│   ├── evaluate_llm.py         # sliced evaluation vs gold labels
│   ├── rag.py                  # hybrid retrieval + metric parsing + consistency check
│   ├── severity.py             # policy §2.1–2.5 as explicit rules
│   ├── response_drafter.py     # grounded drafting + guardrails + red team
│   ├── pipeline.py             # end-to-end orchestration
│   └── db.py                   # approval queue + append-only audit log
├── reports/                    # model_report.json, llm_eval.json
└── docs/
```

---

## Honest limitations

The data is synthetic, so the classifier is partly graded on vocabulary the generator
planted, and the spread model is recovering a function we specified. The guardrails are
regular expressions and lose to paraphrase. The human checkpoint will become a rubber
stamp at 287 alerts a day. Severity is still moved by reach, which means a 12-follower
customer with food poisoning ranks below a verified account with the same complaint.

All of these are documented in `docs/RISKS.md` rather than defended.

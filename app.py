"""
PulseGuard — AI-powered social listening and crisis response (educational prototype).
Run:  streamlit run app.py
"""
import os, sys, json, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd, numpy as np, altair as alt

from src.pipeline import Pipeline
from src.rag import PolicyIndex, parse_policy_metrics, verify_against_code, evaluate_retrieval, load_chunks
from src.response_drafter import red_team, run_guardrails
from src.severity import MISS_COST, MATRIX, SLA, APPROVER
from src import db

st.set_page_config(page_title="PulseGuard — Social Listening & Crisis Response",
                   page_icon="🛡️", layout="wide")

SEV_COLOR = {"SEV-1": "#b3261e", "SEV-2": "#e8710a", "SEV-3": "#1a73e8", "SEV-4": "#5f6368"}

# ----------------------------------------------------------------- resources
@st.cache_resource
def get_pipeline(engine: str):
    return Pipeline(engine=engine)

@st.cache_resource
def get_con():
    return db.connect()

@st.cache_data
def get_mentions():
    return pd.read_csv("data/mentions.csv")

@st.cache_data
def load_report(path):
    return json.load(open(path)) if os.path.exists(path) else None

@st.cache_resource
def get_index():
    return PolicyIndex()

def sev_badge(sev):
    return (f"<span style='background:{SEV_COLOR.get(sev,'#5f6368')};color:#fff;"
            f"padding:2px 10px;border-radius:12px;font-weight:600;font-size:0.85em'>{sev}</span>")

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛡️ PulseGuard")
st.sidebar.caption("ZaykaNow · Social Listening & Crisis Response")

engine_label = st.sidebar.selectbox(
    "Classifier / drafter engine",
    ["heuristic (offline, no API key)", "anthropic (needs ANTHROPIC_API_KEY)"])
engine = "anthropic" if engine_label.startswith("anthropic") else "heuristic"
if engine == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
    st.sidebar.warning("No ANTHROPIC_API_KEY found — the system will fall back to the "
                       "offline engine and record that it did.")

agent_initial = st.sidebar.text_input("Your initial (sign-off, policy 3.1)", "R", max_chars=2)
actor_name = st.sidebar.text_input("Your name (audit log)", "Priya Nair")
actor_role = st.sidebar.selectbox("Your role (policy 2.4)",
    ["Social Media Executive", "Social Media Manager",
     "Director, Brand Communications", "Legal counsel"])

st.sidebar.divider()
con = get_con()
c = db.counts(con)
st.sidebar.metric("Queue — pending", c["PENDING"])
st.sidebar.metric("Open crisis alerts", c["crises"])
st.sidebar.metric("Awaiting human review (policy 5)", c["awaiting_human_review"])
st.sidebar.caption(f"Blocked drafts: {c['blocked_drafts']} · Approved: {c['APPROVED']} · Rejected: {c['REJECTED']}")

if st.sidebar.button("Seed / refresh demo queue"):
    df = get_mentions()
    sub = pd.concat([df[df.mention_id.str.startswith("ZMB")], df.sample(280, random_state=3)])
    with st.spinner("Triaging mentions…"):
        res = get_pipeline(engine).ingest(sub, con, limit=300)
    st.sidebar.success(f"Triaged {res['ingested']} · {res['crises']} crises · "
                       f"{len(res['clusters'])} cluster alert(s)")
    st.cache_data.clear()

st.sidebar.divider()
st.sidebar.error("**Educational prototype.** Not financial, medical or legal advice. "
                 "ZaykaNow is a fictional brand. All mentions are synthetic.")

# -------------------------------------------------------------------- tabs
tabs = st.tabs(["🔎 Live Triage", "🚨 Crisis Board", "✅ Approval Queue",
                "📈 Spread Model", "🧪 LLM Evaluation", "📕 Policy & RAG",
                "💼 Business Case", "⚠️ Risks"])

# ============================================================ 1. LIVE TRIAGE
with tabs[0]:
    st.header("Live triage")
    st.caption("One mention through the whole pipeline: classify → predict spread → "
               "tier against the policy → retrieve clauses → draft → guardrails → human gate.")

    SCENARIOS = {
      "1 · Routine mention": dict(
        text="ZaykaNow packaging has improved a lot, nothing spilled this time",
        lang="en", followers=430, likes_1h=3, reposts_1h=0, replies_1h=1,
        verified=0, media=0, prior=-0.3, hour=15, platform="x"),
      "2 · Complaint needing a reply": dict(
        text="ZaykaNow charged me twice for one order and the refund has not come in 9 days",
        lang="en", followers=8600, likes_1h=61, reposts_1h=14, replies_1h=22,
        verified=0, media=1, prior=0.6, hour=20, platform="x"),
      "3 · Crisis alert": dict(
        text="ZaykaNow my entire family has been throwing up since dinner. the chicken was raw in the middle. I have photos",
        lang="en", followers=240000, likes_1h=1900, reposts_1h=820, replies_1h=460,
        verified=1, media=1, prior=1.9, hour=21, platform="x"),
      "4 · Edge case: sarcasm in Hinglish": dict(
        text="wah ZaykaNow, sirf 2 ghante. bahut fast. aise hi service dete raho",
        lang="hi-en", followers=15400, likes_1h=240, reposts_1h=90, replies_1h=55,
        verified=0, media=0, prior=0.9, hour=21, platform="x"),
    }
    pick = st.radio("Demo scenario", list(SCENARIOS) + ["Custom…"], horizontal=True)
    base = SCENARIOS.get(pick, SCENARIOS["2 · Complaint needing a reply"])

    col1, col2 = st.columns([3, 2])
    with col1:
        text = st.text_area("Mention text", base["text"], height=100)
        lang = st.selectbox("Language", ["en", "hi-en", "hi"],
                            index=["en","hi-en","hi"].index(base["lang"]))
    with col2:
        cc = st.columns(2)
        followers = cc[0].number_input("Followers", 0, 5_000_000, base["followers"], step=100)
        hour      = cc[1].number_input("Hour posted", 0, 23, base["hour"])
        likes     = cc[0].number_input("Likes (1h)", 0, 500_000, base["likes_1h"])
        reposts   = cc[1].number_input("Reposts (1h)", 0, 500_000, base["reposts_1h"])
        replies   = cc[0].number_input("Replies (1h)", 0, 500_000, base["replies_1h"])
        platform  = cc[1].selectbox("Platform", ["x","instagram","reddit","youtube","playstore"],
                                    index=["x","instagram","reddit","youtube","playstore"].index(base["platform"]))
        verified  = cc[0].checkbox("Verified author", bool(base["verified"]))
        media     = cc[1].checkbox("Has image/video", bool(base["media"]))

    if st.button("Triage this mention", type="primary"):
        meta = dict(followers=followers, author_verified=int(verified), has_media=int(media),
                    author_prior_virality=float(base["prior"]), hour=int(hour), platform=platform,
                    likes_1h=likes, reposts_1h=reposts, replies_1h=replies, lang=lang,
                    created_at=pd.Timestamp.now().isoformat(timespec="seconds"),
                    author_handle="demo_user")
        pipe = get_pipeline(engine)
        r = pipe.run_one(text, meta, agent_initial=agent_initial)
        st.session_state["last"] = (r, meta, text)

    if "last" in st.session_state:
        r, meta, text = st.session_state["last"]
        cl, sev, d = r["classification"], r["severity"], r["draft"]

        if sev.is_crisis:
            st.error(f"🚨 **POSSIBLE CRISIS** — {sev.severity}. Respond within {sev.sla}. "
                     f"Approver required: {sev.approver}")
        elif sev.routed_to_human_review:
            st.warning(f"🧑 **Routed to human reviewer** — {sev.review_reason}")
        else:
            st.success(f"Tiered {sev.severity} · first response SLA {sev.sla}")

        m = st.columns(5)
        m[0].metric("Topic", cl.topic.replace("_"," "))
        m[1].metric("Sentiment", cl.sentiment)
        m[2].metric("Urgency", cl.urgency)
        m[3].metric("Classifier confidence", f"{cl.confidence:.2f}")
        m[4].metric("P(viral)", f"{r['p_viral']:.2f}")

        m2 = st.columns(4)
        m2[0].metric("Predicted 24h reach", f"{int(r['predicted_reach']):,}")
        m2[1].metric("Reach band", f"{sev.band} · {sev.band_label}")
        m2[2].metric("Harm class", sev.harm_class)
        m2[3].metric("If missed, costs", f"₹{sev.miss_cost_inr:,}")
        st.markdown(f"**Severity:** {sev_badge(sev.severity)}", unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Why this tier")
            for reason in sev.reasons:
                st.markdown(f"- {reason}")
            if cl.redactions:
                st.info(f"Policy 8: redacted before any model call — {', '.join(cl.redactions)}")
            st.caption(f"Classifier engine: `{cl.engine}` · rationale: {cl.rationale}")

        with right:
            st.subheader("Drafted reply")
            if d.blocked:
                st.error("🛑 Draft blocked — it cannot enter the approval queue.")
                for n in d.notes: st.markdown(f"- {n}")
            if d.text:
                st.text_area("Draft (editable before approval)", d.text, height=120, key="draft_edit")
            st.caption(f"Drafting engine: `{d.engine}` · approver required: **{d.requires_approver}** · SLA {d.sla}")

            st.markdown("**Guardrail checks (policy 3.1–3.3)**")
            gc = pd.DataFrame([x.dict() for x in d.checks])
            gc["result"] = np.where(gc.passed, "✅ pass", "❌ FAIL")
            st.dataframe(gc[["result","rule","clause","detail"]], hide_index=True,
                         width='stretch', height=260)

        st.subheader("Policy clauses retrieved (RAG grounding)")
        for cit in d.citations:
            with st.expander(f"§{cit['section']} — {cit['heading']}  ·  score {cit['score']}"):
                st.markdown(cit["text"][:2500])

        st.divider()
        st.subheader("Human checkpoint")
        st.caption("Policy 7.1 — nothing is published automatically. "
                   "Policy 7.2 — the approver, not the model, is accountable for the text.")
        can_approve = actor_role in (d.requires_approver,) or \
            (actor_role == "Director, Brand Communications") or (actor_role == "Legal counsel") or \
            (actor_role == "Social Media Manager" and d.requires_approver == "Social Media Executive")
        if not can_approve:
            st.warning(f"Your role ({actor_role}) is below the minimum approver for "
                       f"{sev.severity}: **{d.requires_approver}** (policy 2.4).")
        b = st.columns(3)
        if b[0].button("✅ Approve & send", disabled=d.blocked or not can_approve):
            st.success("Approved. In production this is the only point at which text goes public.")
        if b[1].button("✏️ Edit & approve", disabled=d.blocked or not can_approve):
            st.success("Edited draft approved — the edit is what gets logged as text_as_sent.")
        if b[2].button("❌ Reject"):
            st.info("Rejected. Logged with your name, role and timestamp (policy 7.3).")

# =========================================================== 2. CRISIS BOARD
with tabs[1]:
    st.header("Crisis board")
    st.caption("Policy 2.5 — SEV-1, or SEV-2 at mass reach. Plus cluster alerts.")
    rows = db.fetch(con, "is_crisis=1", limit=200)
    if not rows:
        st.info("No crisis alerts in the queue. Use **Seed / refresh demo queue** in the sidebar.")
    else:
        for r in rows:
            with st.container(border=True):
                a, b = st.columns([4, 2])
                with a:
                    st.markdown(f"{sev_badge(r['severity'])} &nbsp; **{r['pred_topic'].replace('_',' ')}** "
                                f"· @{r['author_handle']} · {r['platform']} · {r['created_at'][:16]}",
                                unsafe_allow_html=True)
                    st.write(r["text"])
                with b:
                    st.metric("Predicted reach", f"{int(r['predicted_reach']):,}")
                    st.caption(f"band {r['band']} · harm {r['harm_class']} · "
                               f"P(viral) {r['p_viral']:.2f} · SLA {r['sla']}")
                if r["draft_blocked"]:
                    st.error("Draft blocked by policy — a human must write this reply.")
                elif r["draft"]:
                    st.code(r["draft"], language=None)
                st.caption(f"Requires: **{r['required_approver']}** · status: {r['status']}")

    st.subheader("Cluster alerts (policy 2.5)")
    df = get_mentions()
    from src.severity import cluster_alert, HARM_BY_TOPIC
    sub = pd.concat([df[df.mention_id.str.startswith("ZMB")], df.sample(280, random_state=3)])
    pts = [(pd.to_datetime(x.created_at), HARM_BY_TOPIC.get(x.topic_gold, "H1")) for x in sub.itertuples()]
    cls = cluster_alert(pts)
    if cls:
        for cl_ in cls:
            st.error(f"**{cl_['severity']} CLUSTER** · {cl_['count']} H2/H3 mentions between "
                     f"{cl_['window_start'][:16]} and {cl_['window_end'][:16]} — {cl_['reason']}")
    else:
        st.success("No cluster alerts in the current window.")

# ========================================================= 3. APPROVAL QUEUE
with tabs[2]:
    st.header("Approval queue")
    st.caption("Policy 7 — every mention sits in PENDING until a human with the right role acts.")
    f = st.columns(4)
    sev_f = f[0].multiselect("Severity", ["SEV-1","SEV-2","SEV-3","SEV-4"], ["SEV-1","SEV-2","SEV-3"])
    st_f  = f[1].multiselect("Status", ["PENDING","APPROVED","REJECTED"], ["PENDING"])
    only_review  = f[2].checkbox("Only awaiting human review (policy 5)")
    only_blocked = f[3].checkbox("Only blocked drafts")

    where, params = [], []
    if sev_f: where.append(f"severity IN ({','.join('?'*len(sev_f))})"); params += sev_f
    if st_f:  where.append(f"status IN ({','.join('?'*len(st_f))})"); params += st_f
    if only_review:  where.append("routed_to_human=1")
    if only_blocked: where.append("draft_blocked=1")
    rows = db.fetch(con, " AND ".join(where), tuple(params), limit=60)
    st.caption(f"{len(rows)} shown")

    for r in rows[:25]:
        with st.expander(f"{r['severity']} · {r['pred_topic']} · @{r['author_handle']} · "
                         f"{r['text'][:70]}…"):
            a, b = st.columns([3, 2])
            with a:
                st.write(r["text"])
                st.caption(f"lang {r['lang']} · confidence {r['pred_confidence']:.2f} · "
                           f"sentiment {r['pred_sentiment']} · urgency {r['pred_urgency']}")
                if r["routed_to_human"]:
                    st.warning(f"Human review required — {r['review_reason']}")
                if r["draft_blocked"]:
                    st.error("Draft blocked by a policy 3.3 check.")
                elif r["draft"]:
                    st.code(r["draft"], language=None)
            with b:
                st.metric("Predicted reach", f"{int(r['predicted_reach']):,}")
                st.caption(f"band {r['band']} · harm {r['harm_class']}")
                st.caption(f"approver: {r['required_approver']}")
                st.caption(f"SLA {r['sla']} · status **{r['status']}**")
            cols = st.columns(3)
            if cols[0].button("Approve", key=f"a{r['mention_id']}", disabled=bool(r["draft_blocked"])):
                db.set_status(con, r["mention_id"], "APPROVED", actor_name, actor_role,
                              r["draft"], "approved in queue"); st.rerun()
            if cols[1].button("Reject", key=f"r{r['mention_id']}"):
                db.set_status(con, r["mention_id"], "REJECTED", actor_name, actor_role,
                              "", "rejected in queue"); st.rerun()
            if cols[2].button("Audit trail", key=f"t{r['mention_id']}"):
                st.dataframe(pd.DataFrame([dict(x) for x in db.audit_rows(con, r["mention_id"])]),
                             width='stretch')

    st.subheader("Audit log (policy 7.3)")
    arows = [dict(x) for x in db.audit_rows(con, limit=120)]
    if arows:
        st.dataframe(pd.DataFrame(arows)[["ts","mention_id","actor","actor_role","action",
                                          "from_status","to_status","severity","model_version"]],
                     width='stretch', height=280, hide_index=True)

# =========================================================== 4. SPREAD MODEL
with tabs[3]:
    st.header("Predictive model — will this post spread?")
    rep = load_report("reports/model_report.json")
    if not rep:
        st.warning("Run `python src/train_spread_model.py` first.")
    else:
        w = rep["winner"]; W = rep[w]; op = W["operating_point"]
        st.subheader("Performance against baselines")
        base_tbl = pd.DataFrame([
            {"model":"B0 · never alert", "precision":rep["B0_never_alert"]["precision"],
             "recall":rep["B0_never_alert"]["recall"],
             "alerts/day":0, "daily cost ₹":rep["B0_never_alert"].get("daily_cost_inr",0)},
            {"model":"B1 · analyst rule (followers > 50k)",
             "precision":rep["B1_followers_gt_50k"]["precision"],
             "recall":rep["B1_followers_gt_50k"]["recall"],
             "alerts/day":rep["B1_followers_gt_50k"].get("alerts_per_day",0),
             "daily cost ₹":rep["B1_followers_gt_50k"].get("daily_cost_inr",0)},
            {"model":f"{w} (shipped)", "precision":op.get("precision_on_eligible"),
             "recall":op.get("recall_on_eligible"),
             "alerts/day":op.get("alerts_per_day_projected"),
             "daily cost ₹":op.get("daily_cost_inr")},
        ])
        st.dataframe(base_tbl, hide_index=True, width='stretch')

        k = st.columns(5)
        k[0].metric("ROC-AUC", W["roc_auc"]); k[1].metric("PR-AUC", W["pr_auc"])
        k[2].metric("Brier", W["brier"])
        k[3].metric("Threshold", op["threshold"])
        k[4].metric("Analysts implied", op.get("analysts_required"))
        st.caption(f"Temporal split — train on the first {rep['split']['train_rows']:,} mentions "
                   f"by time, test on the last {rep['split']['test_rows']:,}. A random split would "
                   f"let the model learn from the future.")

        st.subheader("Where the threshold comes from")
        st.caption("Each point is one threshold. Daily cost = missed crises (priced by severity "
                   "tier) + false-alarm handling + the analysts the queue would need.")
        ec = pd.DataFrame(W["economics"])
        base_c = alt.Chart(ec).encode(x=alt.X("threshold:Q", title="alert threshold"))
        st.altair_chart(
            (base_c.mark_line(color="#b3261e").encode(
                y=alt.Y("daily_cost_inr:Q", title="expected daily cost (₹)",
                        scale=alt.Scale(type="log"))) +
             alt.Chart(pd.DataFrame([{"t": op["threshold"]}])).mark_rule(
                 color="#1a73e8", strokeDash=[6,4]).encode(x="t:Q")
            ).properties(height=260), width='stretch')

        cc2 = st.columns(2)
        with cc2[0]:
            st.markdown("**Precision / recall across thresholds**")
            melt = ec.melt("threshold", ["precision","recall"], var_name="metric", value_name="v")
            st.altair_chart(alt.Chart(melt).mark_line().encode(
                x="threshold:Q", y=alt.Y("v:Q", title=None), color="metric:N").properties(height=230),
                width='stretch')
        with cc2[1]:
            st.markdown("**Alerts per day vs analysts needed**")
            st.altair_chart(alt.Chart(ec).mark_line(color="#e8710a").encode(
                x="threshold:Q", y=alt.Y("alerts_per_day:Q", scale=alt.Scale(type="log"),
                                         title="alerts/day")).properties(height=230),
                width='stretch')

        st.subheader("What each error costs")
        ca = rep["cost_assumptions"]
        st.markdown(
            f"- **False negative** (a real crisis not flagged): priced per severity tier — "
            f"SEV-1 ₹{MISS_COST['SEV-1']:,}, SEV-2 ₹{MISS_COST['SEV-2']:,}, SEV-3 ₹{MISS_COST['SEV-3']:,}. "
            f"The SEV-1 figure is an *expected* cost: a full crisis costs ~₹8,00,000 and about "
            f"one in five missed SEV-1s escalates that far.\n"
            f"- **False positive** (a false alarm): ₹{ca['COST_FP_inr']:,} — about 30 minutes of a "
            f"senior analyst, plus the manager glance it triggers.\n"
            f"- Ratio of a missed SEV-1 to a false alarm: **{ca.get('ratio_sev1_miss_to_false_alarm','127')}:1** — that is why recall is protected over precision.\n"
            f"- At {ca['daily_mention_volume']:,} mentions a day, one analyst reviews "
            f"{ca['alerts_per_analyst_day']} alerts and costs ₹{ca['analyst_cost_per_day_inr']:,}/day.")

        st.subheader("Is the threshold an artefact of our cost guesses?")
        sens = pd.DataFrame(rep["sensitivity_to_miss_cost"])
        st.dataframe(sens, hide_index=True, width='stretch')
        st.caption(rep.get("sensitivity_conclusion",""))

        st.subheader("What the model actually learned")
        coefs = pd.DataFrame(rep["logistic_coefficients"]).head(14)
        st.altair_chart(alt.Chart(coefs).mark_bar().encode(
            x=alt.X("coef:Q", title="logistic coefficient (standardised)"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.condition(alt.datum.coef > 0, alt.value("#b3261e"), alt.value("#1a73e8"))
        ).properties(height=380), width='stretch')
        st.caption("Follower count carries the largest single coefficient — but the model also "
                   "uses first-hour reposts, likes, the author's prior virality and negativity, "
                   "and that combination is what lifts recall from 25% (the followers-only rule) "
                   "to 94%. The rule fails on small accounts whose posts get picked up; those "
                   "show up here as the reposts and prior-virality terms.")

# ========================================================== 5. LLM EVALUATION
with tabs[4]:
    st.header("LLM component — tested against human labels")
    ev = load_report("reports/llm_eval.json")
    if not ev:
        st.warning("Run `python src/evaluate_llm.py` first.")
    else:
        k = st.columns(4)
        k[0].metric("Topic accuracy", f"{ev['topic']['accuracy']:.1%}")
        k[1].metric("Topic macro-F1", f"{ev['topic']['macro_f1']:.3f}")
        k[2].metric("Sentiment accuracy", f"{ev['sentiment']['accuracy']:.1%}")
        k[3].metric("Urgency accuracy", f"{ev['urgency']['accuracy']:.1%}")
        st.caption(f"Engine: `{ev['engine']}` · n = {ev['n']:,} mentions, stratified so rare "
                   f"high-harm classes are actually measured.")

        st.subheader("Where it breaks: performance by slice")
        sl = pd.DataFrame(ev["slices"]).T.reset_index().rename(columns={"index":"slice"})
        st.dataframe(sl, hide_index=True, width='stretch')
        melt = sl.melt("slice", ["topic_accuracy","sentiment_accuracy","urgency_accuracy"],
                       var_name="metric", value_name="v")
        st.altair_chart(alt.Chart(melt).mark_bar().encode(
            x=alt.X("v:Q", title="accuracy", scale=alt.Scale(domain=[0,1])),
            y=alt.Y("slice:N", title=None), color="metric:N",
            yOffset="metric:N").properties(height=260), width='stretch')
        if "language_gap" in ev:
            g = ev["language_gap"]
            st.info(f"**Language gap** — topic accuracy is {g['topic_accuracy_gap']:+.1%} and "
                    f"urgency accuracy {g['urgency_accuracy_gap']:+.1%} worse on Hinglish than "
                    f"English. {g['note']}")

        st.subheader("The error that actually hurts someone")
        de = ev["dangerous_errors"]
        a, b = st.columns(2)
        a.metric("H3 mentions down-tiered", f"{de['h3_downgraded_to_lower_harm']} / {de['h3_mentions']}",
                 f"{de['downgrade_rate']:.1%}")
        b.metric("False H3 escalations", ev["false_h3_escalations"]["count"],
                 f"{ev['false_h3_escalations']['rate_of_all']:.1%} of sample")
        st.caption(de["definition"])
        if de["examples"]:
            st.dataframe(pd.DataFrame(de["examples"]), hide_index=True, width='stretch')

        st.subheader("Sarcasm")
        sd = ev["sarcasm_detection"]
        s = st.columns(3)
        s[0].metric("Sarcasm precision", f"{sd['precision']:.2f}")
        s[1].metric("Sarcasm recall", f"{sd['recall']:.2f}")
        s[2].metric("Sarcastic mentions", sd["n_sarcastic"])
        st.caption("Topic accuracy on sarcastic mentions is far below the literal slice. That is "
                   "why policy 5.2 sends every sarcasm-flagged mention to a human instead of "
                   "trusting the tier.")

        st.subheader("Does the confidence score mean anything?")
        cal = pd.DataFrame(ev["confidence_calibration"])
        st.altair_chart(alt.Chart(cal).mark_bar().encode(
            x=alt.X("bin:N", title="stated confidence", sort=None),
            y=alt.Y("topic_accuracy:Q", title="actual topic accuracy",
                    scale=alt.Scale(domain=[0,1]))).properties(height=230),
            width='stretch')
        st.caption("Confidence has to be monotone for policy 5.1 and 5.3 to be worth anything — "
                   "those rules route low-confidence mentions to humans.")
        rt = ev["routed_to_human_by_policy"]
        st.info(f"**{rt['share_of_sample']:.1%} of mentions are routed to a human** by the "
                f"confidence and sarcasm rules: {rt['below_0_55_floor']} below the 0.55 floor, "
                f"{rt['hinglish_below_0_60']} Hinglish below 0.60, {rt['sarcasm_flagged']} "
                f"sarcasm-flagged. That workload is a real cost of the safety rules, not a free win.")

        st.subheader("Failure gallery")
        st.dataframe(pd.DataFrame(ev["failure_examples"]), hide_index=True, width='stretch')

# =========================================================== 6. POLICY & RAG
with tabs[5]:
    st.header("Policy document & retrieval")
    idx = get_index(); met = parse_policy_metrics()
    ver = verify_against_code(met)
    if ver["consistent"]:
        st.success(f"✅ Code matches the policy document on all {ver['checked']} checked "
                   f"thresholds. The severity matrix, bands and SLAs are parsed from "
                   f"`policy/brand_policy.md` at runtime.")
    else:
        st.error("Policy and code disagree:"); st.write(ver["issues"])

    a, b = st.columns(2)
    with a:
        st.subheader("Metrics parsed out of the document")
        st.caption("These are read from the markdown tables, not re-typed in Python.")
        st.markdown("**Reach bands (§2.1)**")
        st.dataframe(pd.DataFrame(met["reach_bands"]).T.reset_index().rename(
            columns={"index":"band"}), hide_index=True, width='stretch')
        st.markdown("**Severity matrix (§2.3)**")
        st.dataframe(pd.DataFrame(met["severity_matrix"]).T, width='stretch')
        st.markdown("**SLA and approver (§2.4)**")
        st.dataframe(pd.DataFrame({"SLA": met["sla"], "approver": met["approver"]}),
                     width='stretch')
        st.markdown("**Cluster rule (§2.5)**"); st.json(met["cluster_rule"])
    with b:
        st.subheader("Retrieval")
        q = st.text_input("Search the policy", "what do we do when food made someone ill")
        for r in idx.search(q, k=4):
            with st.expander(f"§{r.chunk.section} — {r.chunk.heading}  ·  {r.score:.3f}"):
                st.markdown(r.chunk.text[:2000])

        st.subheader("Retrieval quality")
        cases = [("order 1 hour late and cold","delivery_delay","4.1"),
                 ("found plastic in the biryani","food_safety","4.2"),
                 ("charged twice refund not received","payment_refund","4.3"),
                 ("app crashes on android 14","app_bug","4.4"),
                 ("rider shouted at my mother","rider_behaviour","4.5"),
                 ("refused delivery to that area","discrimination_allegation","4.6"),
                 ("delivered in 18 minutes, lovely rider","praise","4.7"),
                 ("is ZAYKA50 still valid","promo_query","4.8")]
        rr = evaluate_retrieval(idx, cases)
        m = st.columns(2)
        m[0].metric("Recall@4 — lexical only", f"{rr['recall_at_4_lexical_only']:.0%}")
        m[1].metric("Recall — hybrid (shipped)", f"{rr['recall_at_k_hybrid']:.0%}")
        st.caption(rr["note"])
        if rr["lexical_misses"]:
            st.markdown("**Clauses pure TF-IDF failed to retrieve:**")
            st.dataframe(pd.DataFrame(rr["lexical_misses"]), hide_index=True, width='stretch')

    st.divider()
    st.subheader("Red-teaming the hallucination guardrails")
    st.caption("Nine adversarial drafts, each breaking one clause of policy 3.3. The guardrail "
               "runs on the output, not in the prompt — a prompt is a request, a check is a control.")
    rt = pd.DataFrame(red_team())
    rt["result"] = np.where(rt.blocked, "🛑 blocked", "✅ allowed")
    st.dataframe(rt[["result","label","text","triggered"]], hide_index=True, width='stretch')

    st.subheader("Try to break it yourself")
    adv = st.text_area("Write a reply that should be blocked",
                       "Sorry! We'll refund ₹500 and the rider has been suspended.")
    if st.button("Run guardrails"):
        ch = run_guardrails(adv, "SEV-2", "delivery_delay")
        failed = [c for c in ch if not c.passed]
        if failed:
            st.error(f"🛑 Blocked by {len(failed)} check(s)")
            st.dataframe(pd.DataFrame([c.dict() for c in failed]), hide_index=True,
                         width='stretch')
        else:
            st.success("✅ Passed every check — this draft would reach the approval queue.")

    with st.expander("Read the full policy document"):
        st.markdown(open("policy/brand_policy.md", encoding="utf-8").read())

# =========================================================== 7. BUSINESS CASE
with tabs[6]:
    st.header("Business case")
    p = "docs/BUSINESS_CASE.md"
    st.markdown(open(p, encoding="utf-8").read() if os.path.exists(p)
                else "Run the project build to generate `docs/BUSINESS_CASE.md`.")

# ================================================================== 8. RISKS
with tabs[7]:
    st.header("Risks, bias and what we would fix before going live")
    p = "docs/RISKS.md"
    st.markdown(open(p, encoding="utf-8").read() if os.path.exists(p)
                else "Run the project build to generate `docs/RISKS.md`.")

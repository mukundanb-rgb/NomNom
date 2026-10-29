"""
End-to-end triage for one mention:

    text ─► redact ─► LLM classify ─► spread model ─► severity (policy 2.3)
                                                           │
                                 RAG retrieve policy clauses│
                                                           ▼
                                        grounded draft ─► guardrails (3.3)
                                                           ▼
                                        APPROVAL QUEUE (PENDING)  ◄── human gate

Nothing in this file publishes anything. The only exit from PENDING is a human
in the app, which is the whole point of the human checkpoint.
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_features
from src.llm_classifier import classify
from src.severity import assess, cluster_alert, HARM_BY_TOPIC
from src.rag import PolicyIndex
from src.response_drafter import make_draft
from src import db

MODEL_PATH = "models/spread_model.joblib"

class Pipeline:
    def __init__(self, engine: str = "heuristic", model_path: str = MODEL_PATH):
        self.engine = engine
        bundle = joblib.load(model_path)
        self.model = bundle["model"]; self.threshold = bundle["threshold"]
        self.model_version = bundle["version"]; self.winner = bundle["winner"]
        self.index = PolicyIndex()

    def predict_reach(self, meta: dict, topic: str, sentiment: str, text: str = "") -> tuple[float, float]:
        """Returns (p_viral, point estimate of 24h reach).
        The reach estimate is a monotone mapping of the model's probability onto the
        policy's band edges, NOT a separate regression. Saying 'about 40,000' implies
        precision we do not have; what we can defend is the band."""
        row = pd.DataFrame([{**meta, "text": text, "topic_gold": topic, "sentiment_gold": sentiment}])
        X = build_features(row)
        p = float(self.model.predict_proba(X)[0, 1])
        # map probability to a reach estimate inside the band it implies
        if   p < 0.10: reach = 300 + p * 12_000
        elif p < 0.35: reach = 2_000 + (p - 0.10) * 60_000
        elif p < 0.70: reach = 25_000 + (p - 0.35) * 150_000
        else:          reach = 100_000 + (p - 0.70) * 900_000
        return p, float(reach)

    def run_one(self, text: str, meta: dict, agent_initial: str = "R") -> dict:
        c = classify(text, meta.get("lang", "en"), engine=self.engine)
        p_viral, reach = self.predict_reach(meta, c.topic, c.sentiment, text)
        sev = assess(c.topic, reach, sentiment=c.sentiment, confidence=c.confidence,
                     lang=meta.get("lang", "en"), is_sarcastic=c.is_sarcastic)
        d = make_draft(self.index, text, c.topic, sev.severity,
                       lang=meta.get("lang", "en"), engine=self.engine, initial=agent_initial)
        return {"classification": c, "p_viral": p_viral, "predicted_reach": reach,
                "severity": sev, "draft": d}

    def to_queue_row(self, mention_id: str, text: str, meta: dict, r: dict) -> dict:
        c, sev, d = r["classification"], r["severity"], r["draft"]
        return dict(
            mention_id=mention_id, created_at=meta.get("created_at",""),
            platform=meta.get("platform",""), author_handle=meta.get("author_handle",""),
            followers=int(meta.get("followers",0)), text=text, lang=meta.get("lang","en"),
            pred_topic=c.topic, pred_sentiment=c.sentiment, pred_urgency=c.urgency,
            pred_confidence=float(c.confidence), is_sarcastic=int(c.is_sarcastic),
            predicted_reach=float(r["predicted_reach"]), p_viral=float(r["p_viral"]),
            severity=sev.severity, harm_class=sev.harm_class, band=sev.band,
            is_crisis=int(sev.is_crisis), routed_to_human=int(sev.routed_to_human_review),
            review_reason=sev.review_reason or "",
            draft=d.text, draft_blocked=int(d.blocked), draft_engine=d.engine,
            checks_json=json.dumps([x.dict() for x in d.checks]),
            citations_json=json.dumps(d.citations), reasons_json=json.dumps(sev.reasons),
            required_approver=d.requires_approver, sla=d.sla, status="PENDING",
            model_version=self.model_version, classifier_engine=c.engine)

    def ingest(self, df: pd.DataFrame, con, limit: int = 300) -> dict:
        """Triage a batch of mentions into the queue. Used to seed the demo."""
        n_crisis = 0
        sub = df.head(limit)
        for row in sub.itertuples():
            meta = dict(followers=row.followers, author_verified=row.author_verified,
                        has_media=row.has_media, author_prior_virality=row.author_prior_virality,
                        hour=row.hour, platform=row.platform, likes_1h=row.likes_1h,
                        reposts_1h=row.reposts_1h, replies_1h=row.replies_1h,
                        lang=row.lang, created_at=row.created_at, author_handle=row.author_handle)
            r = self.run_one(row.text, meta)
            qrow = self.to_queue_row(row.mention_id, row.text, meta, r)
            db.upsert_queue(con, qrow)
            db.log(con, row.mention_id, "pulseguard", "system", "TRIAGED", None, "PENDING",
                   qrow["severity"], self.model_version, qrow["classifier_engine"],
                   qrow["draft"], f"auto-triaged, crisis={bool(qrow['is_crisis'])}")
            n_crisis += qrow["is_crisis"]
        # policy 2.5 cluster rule, evaluated across the batch
        pts = [(pd.to_datetime(r.created_at), HARM_BY_TOPIC.get(r.topic_gold, "H1"))
               for r in sub.itertuples()]
        clusters = cluster_alert(pts)
        return {"ingested": len(sub), "crises": int(n_crisis), "clusters": clusters}

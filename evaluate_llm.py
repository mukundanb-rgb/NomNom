"""
Measures the LLM/heuristic classifier against the human gold labels.

Reports, because the brief asks for exactly these:
  - accuracy / macro-F1 per field (topic, sentiment, urgency)
  - the confusion matrix for topic
  - SLICED performance: English vs Hinglish, sarcastic vs literal. Aggregate
    numbers hide the bias; the slices are the finding.
  - "hallucination"/dangerous-error rate: the share of H3 (food safety,
    discrimination) mentions that the classifier down-tiered to a non-H3 topic.
    This is the error that actually hurts someone, so it gets its own number.
  - a sample of real failures, written out so they can go on a slide.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from src.llm_classifier import classify, TOPICS
from src.severity import HARM_BY_TOPIC

def run(n=1200, engine="heuristic", seed=7, out="reports/llm_eval.json"):
    df = pd.read_csv("data/mentions.csv")
    # stratify so rare, high-harm classes are actually measured
    parts = []
    for t, g in df.groupby("topic_gold"):
        k = min(len(g), max(40, int(n * len(g) / len(df))))
        parts.append(g.sample(k, random_state=seed))
    samp = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"evaluating {len(samp)} mentions with engine={engine}")

    preds = [classify(r.text, r.lang, engine=engine) for r in samp.itertuples()]
    samp["p_topic"]     = [p.topic for p in preds]
    samp["p_sentiment"] = [p.sentiment for p in preds]
    samp["p_urgency"]   = [p.urgency for p in preds]
    samp["p_conf"]      = [p.confidence for p in preds]
    samp["p_sarc"]      = [int(p.is_sarcastic) for p in preds]
    samp["engine_used"] = [p.engine for p in preds]

    res = {"engine": engine, "n": int(len(samp)),
           "engine_calls": dict(Counter(samp.engine_used))}

    for field, gold in (("topic","topic_gold"), ("sentiment","sentiment_gold"), ("urgency","urgency_gold")):
        y, yh = samp[gold], samp[f"p_{field}"]
        res[field] = {"accuracy": round(float(accuracy_score(y, yh)), 4),
                      "macro_f1": round(float(f1_score(y, yh, average="macro", zero_division=0)), 4)}

    # topic confusion matrix
    labs = sorted(set(samp.topic_gold) | set(samp.p_topic))
    cm = confusion_matrix(samp.topic_gold, samp.p_topic, labels=labs)
    res["topic_confusion"] = {"labels": labs, "matrix": cm.tolist()}
    res["topic_report"] = classification_report(samp.topic_gold, samp.p_topic,
                                                zero_division=0, output_dict=True)

    # ---- slices: this is where the bias lives ----
    slices = {}
    for name, mask in (("english", samp.lang == "en"),
                       ("hinglish", samp.lang != "en"),
                       ("literal", samp.is_sarcastic == 0),
                       ("sarcastic", samp.is_sarcastic == 1)):
        s = samp[mask]
        if len(s) == 0: continue
        slices[name] = {
            "n": int(len(s)),
            "topic_accuracy":     round(float(accuracy_score(s.topic_gold, s.p_topic)), 4),
            "sentiment_accuracy": round(float(accuracy_score(s.sentiment_gold, s.p_sentiment)), 4),
            "urgency_accuracy":   round(float(accuracy_score(s.urgency_gold, s.p_urgency)), 4),
            "mean_confidence":    round(float(s.p_conf.mean()), 4),
        }
    res["slices"] = slices
    if "english" in slices and "hinglish" in slices:
        res["language_gap"] = {
            "topic_accuracy_gap": round(slices["english"]["topic_accuracy"] - slices["hinglish"]["topic_accuracy"], 4),
            "urgency_accuracy_gap": round(slices["english"]["urgency_accuracy"] - slices["hinglish"]["urgency_accuracy"], 4),
            "note": "Positive numbers mean the classifier is worse on Hinglish. Policy 5.1 "
                    "exists to catch exactly this, by routing low-confidence code-switched "
                    "mentions to a human instead of auto-tiering them."}

    # ---- the error that actually hurts: down-tiering a high-harm mention ----
    samp["gold_harm"] = samp.topic_gold.map(HARM_BY_TOPIC)
    samp["pred_harm"] = samp.p_topic.map(HARM_BY_TOPIC)
    h3 = samp[samp.gold_harm == "H3"]
    down = h3[h3.pred_harm != "H3"]
    res["dangerous_errors"] = {
        "h3_mentions": int(len(h3)),
        "h3_downgraded_to_lower_harm": int(len(down)),
        "downgrade_rate": round(float(len(down)/max(len(h3),1)), 4),
        "definition": "A real food-safety or discrimination mention that the classifier "
                      "assigned to a lower harm class. This is the failure that delays a "
                      "SEV-1 response, so we track it separately from overall accuracy.",
        "examples": [{"text": r.text[:200], "gold": r.topic_gold, "predicted": r.p_topic,
                      "confidence": float(r.p_conf), "lang": r.lang}
                     for r in down.head(8).itertuples()]}
    # and the opposite: crying wolf
    up = samp[(samp.gold_harm != "H3") & (samp.pred_harm == "H3")]
    res["false_h3_escalations"] = {
        "count": int(len(up)),
        "rate_of_all": round(float(len(up)/len(samp)), 4),
        "examples": [{"text": r.text[:200], "gold": r.topic_gold, "predicted": r.p_topic}
                     for r in up.head(6).itertuples()]}

    # ---- sarcasm detection as its own binary problem ----
    res["sarcasm_detection"] = {
        "precision": round(float((samp.p_sarc.eq(1) & samp.is_sarcastic.eq(1)).sum()
                                 / max(samp.p_sarc.sum(), 1)), 4),
        "recall": round(float((samp.p_sarc.eq(1) & samp.is_sarcastic.eq(1)).sum()
                              / max(samp.is_sarcastic.sum(), 1)), 4),
        "n_sarcastic": int(samp.is_sarcastic.sum())}

    # ---- does confidence mean anything? (it must, policy 5.1/5.3 depend on it) ----
    bins = pd.cut(samp.p_conf, [0,.4,.5,.6,.7,.8,1.01],
                  labels=["<0.4","0.4-0.5","0.5-0.6","0.6-0.7","0.7-0.8",">0.8"])
    cal = samp.assign(correct=(samp.topic_gold == samp.p_topic).astype(int)).groupby(bins, observed=True)
    res["confidence_calibration"] = [
        {"bin": str(b), "n": int(len(g)), "topic_accuracy": round(float(g.correct.mean()), 4)}
        for b, g in cal]
    res["routed_to_human_by_policy"] = {
        "below_0_55_floor": int((samp.p_conf < 0.55).sum()),
        "hinglish_below_0_60": int(((samp.lang != "en") & (samp.p_conf < 0.60)).sum()),
        "sarcasm_flagged": int(samp.p_sarc.sum()),
        "share_of_sample": round(float(((samp.p_conf < 0.55) | (samp.p_sarc == 1) |
                                        ((samp.lang != "en") & (samp.p_conf < 0.60))).mean()), 4)}

    # ---- a readable failure gallery for the slides ----
    wrong = samp[samp.topic_gold != samp.p_topic]
    res["failure_examples"] = [
        {"text": r.text[:220], "lang": r.lang, "sarcastic": int(r.is_sarcastic),
         "gold_topic": r.topic_gold, "pred_topic": r.p_topic,
         "gold_urgency": r.urgency_gold, "pred_urgency": r.p_urgency,
         "confidence": float(r.p_conf)}
        for r in wrong.sample(min(15, len(wrong)), random_state=1).itertuples()]

    os.makedirs("reports", exist_ok=True)
    with open(out, "w") as fh: json.dump(res, fh, indent=2)
    samp.to_csv(out.replace(".json", "_rows.csv"), index=False)

    print(f"topic acc={res['topic']['accuracy']} macroF1={res['topic']['macro_f1']}")
    print(f"sentiment acc={res['sentiment']['accuracy']}  urgency acc={res['urgency']['accuracy']}")
    print("slices:", {k: v["topic_accuracy"] for k, v in slices.items()})
    print(f"H3 downgrade rate = {res['dangerous_errors']['downgrade_rate']} "
          f"({res['dangerous_errors']['h3_downgraded_to_lower_harm']}/{res['dangerous_errors']['h3_mentions']})")
    print(f"sarcasm P={res['sarcasm_detection']['precision']} R={res['sarcasm_detection']['recall']}")
    print(f"routed to human: {res['routed_to_human_by_policy']['share_of_sample']:.1%} of sample")
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--engine", default="heuristic")
    ap.add_argument("--out", default="reports/llm_eval.json")
    a = ap.parse_args()
    run(a.n, a.engine, out=a.out)

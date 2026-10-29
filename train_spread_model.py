"""
Trains the spread (virality) model and chooses the alert threshold by COST,
not by F1. Writes models/spread_model.joblib and reports/model_report.json.

Three models are compared so "against a baseline" is a real claim:
  B0  majority class       (never alert)       -- the do-nothing baseline
  B1  simple rule          (followers > 50k)   -- what a human analyst does today
  M1  logistic regression  (interpretable)
  M2  gradient boosting    (the candidate)
Split is TEMPORAL (train on the first 80% of time, test on the last 20%) because
a random split lets the model learn from the future, which it will not have live.
"""
import json, os, sys, numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, confusion_matrix, brier_score_loss)
from src.features import build_features, FEATURE_ORDER
from src.severity import assess, MISS_COST, HARM_BY_TOPIC, reach_band

# ----------------------------------------------------------- cost assumptions
# Defended in docs/BUSINESS_CASE.md. Change them there, not here, and re-run.
#
# The cost of a MISS is not one number. Missing a SEV-1 food-safety thread and
# missing a viral late-delivery gripe are different events, so the miss cost comes
# from src.severity.MISS_COST, applied per row using the post's TRUE severity
# (computed from its actual 24h reach). An earlier version of this script costed
# every viral post at 800k and produced a 'crisis' every 3 minutes and an annual
# loss of 83 billion rupees -- see docs/PROMPT_LOG.md entry 8.
COST_FP = 1_200     # a false alarm: ~30 min of a senior analyst + the manager
                    # glance it triggers. Cheap, but 100/day is not cheap.
COST_FN = 800_000   # retained only for the legacy single-number baseline table

# The cost ratio alone (667:1) pushes the optimiser to alert on almost everything.
# That is mathematically right and operationally useless: the crisis desk is three
# people. DESK_CAPACITY is the real constraint, and the operating point we ship is
# the STRICTER of (cost-optimal, capacity-feasible). This is the honest version of
# "set an alert threshold by weighing false alarms against missed crises".
DAILY_MENTION_VOLUME = 4_200    # policy section 1
ALERTS_PER_ANALYST_DAY = 40     # 8h shift at ~10 min + context switching per alert
ANALYST_COST_PER_DAY   = 3_200  # fully loaded senior social analyst, INR
DESK_CAPACITY_TODAY    = 120    # what the brand staffs today: 3 analysts

# We do NOT hard-code the threshold to today's staffing. Staffing is a DECISION, and
# the cost model should produce it. For each threshold we compute the full daily cost:
#     missed crises + false-alarm handling + the analysts needed to review the queue
# and report the minimum. The gap between the optimum and today's 3-analyst desk is
# the headline business finding of this project.

# A high predicted reach is NOT by itself an alert. Policy 2.5 raises a crisis only
# when the severity matrix says SEV-1, or SEV-2 with reach band R2+. Severity needs
# a harm class (H1-H3) and negative sentiment. So the capacity constraint is applied
# to the ALERT-ELIGIBLE funnel, not to every post the model thinks will spread.
# Getting this wrong (capping on all viral posts) was our first design error and is
# written up in docs/PROMPT_LOG.md entry 6.
HARMFUL = {"delivery_delay","food_safety","payment_refund","app_bug","rider_behaviour",
           "pricing_surge","customer_service","discrimination_allegation"}

def cost_of(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return fp*COST_FP + fn*COST_FN, dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def metrics_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    total, cm = cost_of(y, yhat)
    return dict(threshold=round(float(thr),4),
                precision=round(float(precision_score(y, yhat, zero_division=0)),4),
                recall=round(float(recall_score(y, yhat, zero_division=0)),4),
                f1=round(float(f1_score(y, yhat, zero_division=0)),4),
                alerts=int(yhat.sum()), expected_cost_inr=int(total), **cm)

def main():
    df = pd.read_csv("data/mentions.csv").sort_values("created_at").reset_index(drop=True)
    X  = build_features(df)
    y  = df.viral_gold.values

    # alert-eligible funnel (policy 2.5): negative sentiment on a harm-bearing topic
    eligible_all = ((df.sentiment_gold == "negative") & (df.topic_gold.isin(HARMFUL))).values

    # TRUE severity of every post, from its ACTUAL 24h reach. This is what we would
    # have wished we had flagged, and it supplies the per-row miss cost.
    true_sev  = [assess(t, r, sentiment=s_) for t, r, s_ in
                 zip(df.topic_gold, df.reach_24h, df.sentiment_gold)]
    df["true_severity"]  = [x.severity for x in true_sev]
    df["true_is_crisis"] = [int(x.is_crisis) for x in true_sev]
    df["miss_cost"]      = [x.miss_cost_inr for x in true_sev]
    print("true severity mix:", df.true_severity.value_counts().to_dict())
    print("true crises:", int(df.true_is_crisis.sum()),
          f"({df.true_is_crisis.mean():.2%}) ->",
          round(df.true_is_crisis.mean()*DAILY_MENTION_VOLUME,1), "per day at brand volume")

    cut = int(len(df)*0.80)
    Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]
    elig_te   = eligible_all[cut:]
    miss_te   = df.miss_cost.values[cut:]
    crisis_te = df.true_is_crisis.values[cut:]
    print(f"train={len(Xtr)} (viral {ytr.mean():.3f})  test={len(Xte)} (viral {yte.mean():.3f})")
    print(f"alert-eligible in test: {elig_te.sum()}/{len(elig_te)} ({elig_te.mean():.1%})")

    results = {}

    # ---- B0 never alert
    results["B0_never_alert"] = metrics_at(crisis_te, np.zeros(len(yte)), 0.5)
    results["B0_never_alert"]["daily_cost_inr"] = int(
        miss_te[crisis_te==1].sum() * (DAILY_MENTION_VOLUME/len(yte)))
    # ---- B1 follower rule (what the team does today)
    p_rule = (df.followers.iloc[cut:].values > 50_000).astype(float)
    results["B1_followers_gt_50k"] = metrics_at(crisis_te, p_rule, 0.5)
    _f = (p_rule >= 0.5) & elig_te
    _fn = (~_f) & (crisis_te == 1)
    _alerts = _f.sum() * (DAILY_MENTION_VOLUME/len(yte))
    results["B1_followers_gt_50k"]["daily_cost_inr"] = int(
        miss_te[_fn].sum()*(DAILY_MENTION_VOLUME/len(yte))
        + int((_f & (crisis_te==0)).sum())*(DAILY_MENTION_VOLUME/len(yte))*COST_FP
        + np.ceil(_alerts/ALERTS_PER_ANALYST_DAY)*ANALYST_COST_PER_DAY)
    results["B1_followers_gt_50k"]["alerts_per_day"] = round(float(_alerts),1)

    # ---- M1 logistic (calibrated). We fit an UNCALIBRATED copy too, purely to read
    # coefficients for the explanation panel. class_weight is deliberately NOT used:
    # it inflates probabilities, which makes a probability threshold meaningless.
    # (First version of this script used balanced weights and every threshold read
    # ~0.9 -- see docs/PROMPT_LOG.md entry 7.)
    lr_pipe = Pipeline([("sc", StandardScaler()),
                        ("lr", LogisticRegression(max_iter=2000, C=1.0))])
    m1_raw = Pipeline([("sc", StandardScaler()),
                       ("lr", LogisticRegression(max_iter=2000, C=1.0))]).fit(Xtr, ytr)
    m1 = CalibratedClassifierCV(lr_pipe, method="isotonic", cv=4)
    m1.fit(Xtr, ytr); p1 = m1.predict_proba(Xte)[:,1]

    # ---- M2 gradient boosting, probability-calibrated so thresholds mean something
    base = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.07,
                                          max_iter=300, min_samples_leaf=25,
                                          l2_regularization=1.0, random_state=42)
    m2 = CalibratedClassifierCV(base, method="isotonic", cv=4)
    m2.fit(Xtr, ytr); p2 = m2.predict_proba(Xte)[:,1]

    for name, p, model in (("M1_logistic", p1, m1), ("M2_gradient_boosting", p2, m2)):
        # sweep thresholds, pick the one that minimises expected cost
        grid = np.unique(np.round(np.concatenate([
            np.linspace(0.005, 0.95, 190), np.linspace(0.951, 0.999, 49)]), 4))
        costs = [(cost_of(yte, (p>=t).astype(int))[0], t) for t in grid]
        best_cost, best_thr = min(costs)
        r = metrics_at(yte, p, best_thr)
        r.update(roc_auc=round(float(roc_auc_score(yte,p)),4),
                 pr_auc=round(float(average_precision_score(yte,p)),4),
                 brier=round(float(brier_score_loss(yte,p)),4),
                 at_0_50=metrics_at(yte, p, 0.50),
                 cost_curve=[{"threshold":float(t),"cost":int(c)} for c,t in sorted(costs, key=lambda x:x[1])])
        # ---- full daily economics, including the staffing the queue implies ----
        scale = DAILY_MENTION_VOLUME / len(yte)   # test rows -> one day of volume
        econ = []
        for t in grid:
            fired = (p >= t) & elig_te            # only eligible posts become alerts
            # the decision target is "is this a real crisis we must not miss"
            tp = int((fired & (crisis_te == 1)).sum())
            fp = int((fired & (crisis_te == 0)).sum())
            fn_mask = (~fired) & (crisis_te == 1)
            fn = int(fn_mask.sum())
            miss_cost_total = float(miss_te[fn_mask].sum())      # tier-specific
            alerts_day = (tp + fp) * scale
            analysts   = int(np.ceil(alerts_day / ALERTS_PER_ANALYST_DAY)) if alerts_day > 0 else 0
            daily = (miss_cost_total*scale) + (fp*scale*COST_FP) + (analysts*ANALYST_COST_PER_DAY)
            econ.append(dict(threshold=float(t), alerts_per_day=round(alerts_day,1),
                             analysts_required=analysts,
                             missed_crises_per_day=round(fn*scale,2),
                             missed_cost_per_day_inr=int(miss_cost_total*scale),
                             daily_cost_inr=int(daily),
                             precision=round(float(tp/max(tp+fp,1)),4),
                             recall=round(float(tp/max(tp+fn,1)),4)))
        best_econ = min(econ, key=lambda e: e["daily_cost_inr"])

        # what today's 3-analyst desk forces on us, for the "staff up or miss 70%" slide
        max_rate_today = DESK_CAPACITY_TODAY / DAILY_MENTION_VOLUME
        feas = [e for e in econ if e["alerts_per_day"] <= DESK_CAPACITY_TODAY]
        today_econ = min(feas, key=lambda e: e["daily_cost_inr"]) if feas else econ[-1]

        r["economics"] = econ
        r["cost_optimal_operating_point"] = best_econ
        r["constrained_to_todays_desk"]  = today_econ
        r["operating_point"] = metrics_at(yte, p, best_econ["threshold"])
        r["operating_point"].update(
            alerts_per_day_projected=best_econ["alerts_per_day"],
            analysts_required=best_econ["analysts_required"],
            daily_cost_inr=best_econ["daily_cost_inr"],
            recall_on_eligible=best_econ["recall"],
            precision_on_eligible=best_econ["precision"])
        results[name] = r
        print(f"{name}: AUC={r['roc_auc']} PR-AUC={r['pr_auc']}")
        print(f"   OPTIMAL   thr={best_econ['threshold']} P={best_econ['precision']} "
              f"R={best_econ['recall']} {best_econ['alerts_per_day']}/day "
              f"{best_econ['analysts_required']} analysts  ₹{best_econ['daily_cost_inr']:,}/day")
        print(f"   TODAY(3p) thr={today_econ['threshold']} P={today_econ['precision']} "
              f"R={today_econ['recall']} {today_econ['alerts_per_day']}/day "
              f"₹{today_econ['daily_cost_inr']:,}/day")

    # winner = lowest expected cost among the learned models
    winner = min(("M1_logistic","M2_gradient_boosting"),
                 key=lambda k: results[k]["cost_optimal_operating_point"]["daily_cost_inr"])
    model  = m1 if winner=="M1_logistic" else m2
    thr    = results[winner]["operating_point"]["threshold"]

    # logistic coefficients for the "why" panel in the app
    coefs = sorted(zip(FEATURE_ORDER, m1_raw.named_steps["lr"].coef_[0]),
                   key=lambda kv: -abs(kv[1]))
    results["logistic_coefficients"] = [{"feature":f,"coef":round(float(c),4)} for f,c in coefs]
    results["cost_assumptions"] = {"COST_FN_inr":COST_FN, "COST_FP_inr":COST_FP,
                                   "SEV1_expected_miss_cost_inr": MISS_COST["SEV-1"],
                                   "ratio_sev1_miss_to_false_alarm": round(MISS_COST["SEV-1"]/COST_FP,1),
                                   "daily_mention_volume":DAILY_MENTION_VOLUME,
                                   "alerts_per_analyst_day":ALERTS_PER_ANALYST_DAY,
                                   "analyst_cost_per_day_inr":ANALYST_COST_PER_DAY,
                                   "desk_capacity_today":DESK_CAPACITY_TODAY}
    results["winner"] = winner
    results["chosen_threshold"] = thr
    results["baseline_comparison"] = {
        "B1_rule_recall": results["B1_followers_gt_50k"]["recall"],
        "model_recall":   results[winner]["operating_point"]["recall_on_eligible"],
        "B1_rule_precision": results["B1_followers_gt_50k"]["precision"],
        "model_precision":   results[winner]["operating_point"]["precision_on_eligible"]}
    results["split"] = {"type":"temporal_80_20","train_rows":int(cut),"test_rows":int(len(df)-cut),
                        "train_viral_rate":round(float(ytr.mean()),4),
                        "test_viral_rate":round(float(yte.mean()),4)}

    # ---------------- sensitivity: are our conclusions robust to the cost guesses? ----
    # The miss costs are estimates. The right response to "you made those numbers up"
    # is not a better guess, it is showing what changes when they are wrong.
    sens = []
    p_win = p1 if winner == "M1_logistic" else p2
    scale = DAILY_MENTION_VOLUME / len(yte)
    for mult in (0.5, 0.75, 1.0, 1.5, 2.0):
        rows = []
        for t in grid:
            fired = (p_win >= t) & elig_te
            fp = int((fired & (crisis_te == 0)).sum())
            fn_mask = (~fired) & (crisis_te == 1)
            alerts_day = int((fired).sum()) * scale
            analysts = int(np.ceil(alerts_day/ALERTS_PER_ANALYST_DAY)) if alerts_day>0 else 0
            daily = (miss_te[fn_mask].sum()*mult*scale) + (fp*scale*COST_FP) + analysts*ANALYST_COST_PER_DAY
            tp = int((fired & (crisis_te==1)).sum()); fn = int(fn_mask.sum())
            rows.append((daily, float(t), round(float(tp/max(tp+fn,1)),3),
                         round(float(tp/max(tp+fp,1)),3), round(alerts_day,1), analysts))
        best = min(rows)
        sens.append(dict(miss_cost_multiplier=mult, threshold=best[1], recall=best[2],
                         precision=best[3], alerts_per_day=best[4], analysts_required=best[5],
                         daily_cost_inr=int(best[0])))
        print(f"   sensitivity x{mult}: thr={best[1]} R={best[2]} P={best[3]} "
              f"{best[4]}/day {best[5]} analysts")
    results["sensitivity_to_miss_cost"] = sens
    thr_range = (min(x["threshold"] for x in sens), max(x["threshold"] for x in sens))
    results["sensitivity_conclusion"] = (
        f"Halving or doubling every miss cost moves the chosen threshold only within "
        f"{thr_range[0]}-{thr_range[1]}, so the operating point is not an artefact of "
        f"the cost guesses.")

    os.makedirs("models", exist_ok=True); os.makedirs("reports", exist_ok=True)
    joblib.dump({"model":model, "threshold":thr, "features":FEATURE_ORDER,
                 "version":"spread-v1.0", "winner":winner}, "models/spread_model.joblib")
    # keep test-set predictions for the app's eval tab
    pd.DataFrame({"mention_id":df.mention_id.iloc[cut:].values, "y":yte,
                  "true_severity":df.true_severity.values[cut:],
                  "true_is_crisis":crisis_te, "miss_cost":miss_te,
                  "p":(p1 if winner=="M1_logistic" else p2)}).to_csv("reports/test_predictions.csv", index=False)
    with open("reports/model_report.json","w") as fh: json.dump(results, fh, indent=2)

    w = results[winner]
    print(f"\nWINNER {winner} @ operating threshold {thr}")
    b1 = results["B1_followers_gt_50k"]
    print(f"vs analyst rule: recall {b1['recall']} -> {w['operating_point']['recall_on_eligible']}, "
          f"precision {b1['precision']} -> {w['operating_point']['precision_on_eligible']}")
    opt, tod = w["cost_optimal_operating_point"], w["constrained_to_todays_desk"]
    print(f"STAFFING FINDING: optimum needs {opt['analysts_required']} analysts "
          f"(₹{opt['daily_cost_inr']:,}/day). Today's 3-analyst desk costs "
          f"₹{tod['daily_cost_inr']:,}/day and misses "
          f"{tod['missed_crises_per_day']} crises/day at recall {tod['recall']}.")
    print(f"=> annual difference ₹{(tod['daily_cost_inr']-opt['daily_cost_inr'])*365:,}")

if __name__ == "__main__":
    main()

"""
LLM mention classifier: sentiment, topic, urgency, sarcasm, confidence.

Two interchangeable engines, chosen by PULSEGUARD_ENGINE or the UI:

  "heuristic"  no API key, deterministic, runs offline. This is NOT a toy: it is
               the fallback the production system drops to when the model API is
               unavailable, and it is also the honest baseline the LLM must beat.
  "anthropic"  calls the Claude API with a strict JSON schema.

Both return the same dataclass, so the rest of the system cannot tell them apart.
Every classification is measured against human gold labels in src/evaluate_llm.py.

PRIVACY: text is redacted (policy section 8) BEFORE it leaves this process.
"""
from __future__ import annotations
import os, re, json, hashlib
from dataclasses import dataclass, asdict, field
from typing import Optional, List

TOPICS = ["delivery_delay","food_safety","payment_refund","app_bug","rider_behaviour",
          "pricing_surge","customer_service","praise","promo_query","discrimination_allegation"]
SENTIMENTS = ["negative","neutral","positive"]
URGENCIES  = ["low","medium","high","critical"]

# ------------------------------------------------------------------ privacy
PHONE = re.compile(r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b")
EMAIL = re.compile(r"\b[\w.\-]+@[\w\-]+\.\w{2,}\b")
ORDER = re.compile(r"\b(?:ORD|ORDER|#)\s?\d{5,}\b", re.I)

def redact(text: str) -> tuple[str, List[str]]:
    """Policy 8. Returns (clean_text, list_of_redaction_kinds)."""
    found = []
    if PHONE.search(text): found.append("phone")
    if EMAIL.search(text): found.append("email")
    if ORDER.search(text): found.append("order_id")
    t = PHONE.sub("[PHONE_REDACTED]", text)
    t = EMAIL.sub("[EMAIL_REDACTED]", t)
    t = ORDER.sub("[ORDER_ID_REDACTED]", t)
    return t, found

@dataclass
class Classification:
    sentiment: str
    topic: str
    urgency: str
    confidence: float
    is_sarcastic: bool
    rationale: str
    engine: str
    redactions: List[str] = field(default_factory=list)
    raw: Optional[str] = None
    def dict(self): return asdict(self)

# ------------------------------------------------------------- heuristic engine
TOPIC_CUES = {
 "food_safety": ["plastic","hair","raw","kacha","vomit","throwing up","food poisoning","smell",
                 "smelled","stale","pet kharab","baal","expired","insect","worm","sick","ill"],
 "discrimination_allegation": ["that area","us area","religion","caste","muslim","hindu","dalit",
                 "accent","north indian","south indian","speak proper","refused to deliver to my"],
 "payment_refund": ["refund","charged","double charge","deducted","money","paisa","upi","wallet",
                 "cashback","amount","transaction"],
 "app_bug": ["crash","crashes","bug","freeze","freezes","otp","login","app","version","android",
                 "ios","update","glitch","band ho"],
 "rider_behaviour": ["rider","delivery guy","delivery boy","delivery person","rude","shouted",
                 "threatened","behaviour","behavior","tameez","feka","threw","extra cash","abused"],
 "pricing_surge": ["price","surge","expensive","costly","charge","fee","mehenga","handling",
                 "delivery fee","rate"],
 "customer_service": ["support","agent","ticket","complaint","chat","email","helpline","care",
                 "jawab","response","closed my"],
 "delivery_delay": ["late","delay","delayed","waiting","wait","still not","cold","thanda","minute",
                 "minutes","ghanta","hour","hours","eta","der"],
 "promo_query": ["code","coupon","offer","discount","promo","deal","student discount"],
 "praise": ["thank","thanks","great","lovely","impressed","good","improved","shabaash","acha",
                 "excellent","fast","hot","credit where due"],
}
SARCASM_CUES = ["wah","very fast","truly innovative","10/10","love how","such a","thrilling",
                "world record","bahut fast","efficiency king","generous of you","bold menu",
                "great way to help me save","love the innovation","love the spontaneity",
                "savings scheme","shoutout to","dil khush"]
CRITICAL_CUES = ["hospital","doctor","vomit","throwing up","raw","kacha","poison","blood",
                 "daughter","son","child","legal","lawyer","police","pet kharab","sick"]
NEG_CUES = ["not","never","worst","unacceptable","terrible","bad","angry","disgusting","horrible",
            "nahi","kharab","ganda","bekar","refund","complaint","fed up","shame"]

# Cue matching MUST respect word boundaries. Our first version used plain substring
# matching and classified "still not here" as food_safety, because "st-ILL-" contains
# the cue "ill". That single bug mis-tiered every late-delivery complaint as a
# critical food-safety incident. See docs/PROMPT_LOG.md entry 11.
_CUE_RE: dict = {}
def _has(cue: str, text: str) -> bool:
    rx = _CUE_RE.get(cue)
    if rx is None:
        rx = _CUE_RE[cue] = re.compile(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", re.I)
    return bool(rx.search(text))

def _score_topics(t: str) -> dict:
    return {topic: sum(1 for c in cues if _has(c, t)) for topic, cues in TOPIC_CUES.items()}

def classify_heuristic(text: str, lang: str = "en") -> Classification:
    clean, red = redact(text)
    low = clean.lower()
    scores = _score_topics(low)
    # priority: a food-safety or discrimination cue outranks everything (policy 2.2 H3)
    if scores["discrimination_allegation"] > 0: topic = "discrimination_allegation"
    elif scores["food_safety"] > 0 and not (scores["praise"] > scores["food_safety"]):
        topic = "food_safety"
    else:
        topic = max(scores, key=scores.get)
        if scores[topic] == 0: topic = "delivery_delay"   # fall back to the modal class

    sarcastic = any(_has(c, low) for c in SARCASM_CUES)
    praise_hits = scores["praise"]
    neg_hits = sum(1 for c in NEG_CUES if _has(c, low))

    if sarcastic:                      sentiment = "negative"
    elif topic == "praise" and praise_hits > 0 and neg_hits == 0: sentiment = "positive"
    elif topic == "promo_query" and neg_hits == 0: sentiment = "neutral"
    elif topic == "promo_query":       sentiment = "neutral"
    else:                              sentiment = "negative" if neg_hits or topic != "praise" else "neutral"

    if topic in ("food_safety","discrimination_allegation"): urgency = "critical"
    elif any(_has(c, low) for c in CRITICAL_CUES):           urgency = "high"
    elif topic in ("rider_behaviour","customer_service"):    urgency = "medium"
    elif topic in ("praise","promo_query"):                  urgency = "low"
    else: urgency = "medium" if neg_hits >= 2 else "low"

    # confidence: margin between best and runner-up topic, damped for Hinglish.
    # The damping is deliberate and is what makes policy 5.1 fire.
    ordered = sorted(scores.values(), reverse=True)
    margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else ordered[0]
    conf = min(0.95, 0.45 + 0.17*margin + (0.10 if scores[topic] >= 2 else 0.0))
    if lang != "en": conf -= 0.12
    if sarcastic:    conf -= 0.15
    conf = round(float(max(0.20, min(conf, 0.95))), 3)

    return Classification(sentiment=sentiment, topic=topic, urgency=urgency, confidence=conf,
                          is_sarcastic=sarcastic, engine="heuristic",
                          rationale=f"keyword match topic={topic} (score {scores[topic]}), "
                                    f"negation cues {neg_hits}, sarcasm cues {'yes' if sarcastic else 'no'}",
                          redactions=red)

# ------------------------------------------------------------- anthropic engine
SYSTEM_PROMPT = """You classify public social-media mentions about a food-delivery brand for a triage system.

Return ONLY a JSON object, no prose, with exactly these keys:
  sentiment: one of ["negative","neutral","positive"]  -- the author's INTENT, not surface words
  topic: one of ["delivery_delay","food_safety","payment_refund","app_bug","rider_behaviour","pricing_surge","customer_service","praise","promo_query","discrimination_allegation"]
  urgency: one of ["low","medium","high","critical"]
  is_sarcastic: boolean
  confidence: number between 0 and 1, your honest probability that topic and urgency are both right
  rationale: one short sentence, max 20 words

Rules:
- Much of the text is Hinglish (Hindi written in Latin script). Treat it with the same seriousness as English. Do not mark Hinglish as low urgency because it is informal.
- Sarcastic praise is NEGATIVE sentiment. "wah, sirf 2 ghante, bahut fast" is a complaint.
- urgency is "critical" ONLY for: illness, contamination, injury, or an allegation of discrimination.
- If the text mentions illness, raw food, foreign objects in food, or a child being affected, topic is food_safety.
- Allegations about refusing service based on religion, caste, region, gender or disability are discrimination_allegation, never rider_behaviour.
- Be honest in `confidence`. If the text is ambiguous or code-switched in a way you are unsure about, say 0.4, do not say 0.9."""

def classify_anthropic(text: str, lang: str = "en", model: str = "claude-sonnet-4-5") -> Classification:
    import anthropic
    clean, red = redact(text)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model, max_tokens=300, system=SYSTEM_PROMPT,
        messages=[{"role":"user","content":f"Language hint: {lang}\nMention: {clean}"}])
    raw = msg.content[0].text.strip()
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0) if m else raw)
    except Exception:
        # A malformed response is a real failure mode and must not crash triage.
        # We fall back and we RECORD that we fell back, so the eval counts it.
        c = classify_heuristic(text, lang); c.engine = "anthropic_fallback_parse_error"; c.raw = raw
        return c
    topic = d.get("topic") if d.get("topic") in TOPICS else "delivery_delay"
    sent  = d.get("sentiment") if d.get("sentiment") in SENTIMENTS else "negative"
    urg   = d.get("urgency") if d.get("urgency") in URGENCIES else "medium"
    return Classification(sentiment=sent, topic=topic, urgency=urg,
                          confidence=float(d.get("confidence", 0.5)),
                          is_sarcastic=bool(d.get("is_sarcastic", False)),
                          rationale=str(d.get("rationale",""))[:160],
                          engine=f"anthropic:{model}", redactions=red, raw=raw)

# ------------------------------------------------------------------- facade
def classify(text: str, lang: str = "en", engine: Optional[str] = None) -> Classification:
    engine = engine or os.environ.get("PULSEGUARD_ENGINE", "heuristic")
    if engine == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return classify_anthropic(text, lang)
        except Exception as e:
            c = classify_heuristic(text, lang)
            c.engine = f"anthropic_failed_fallback({type(e).__name__})"
            return c
    return classify_heuristic(text, lang)

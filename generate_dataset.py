"""
PulseGuard synthetic mention generator.
----------------------------------------------------------------------------
Why synthetic: scraping X/Instagram for a graded prototype creates a privacy and
ToS problem, and real crisis posts are too rare to learn from in a 3,000-row
sample. We instead generate from a declared causal process, which has a real
methodological benefit: we KNOW the ground truth data-generating process, so we
can say exactly which relationships the model should find and whether it did.

The honest limitation (state this in the viva): a model evaluated on data drawn
from the same process that generated it is measuring whether the learner can
recover a known function, NOT whether it will work on real Indian social media.
Section 9 of docs/RISKS.md covers this.

Causal story for reach (the thing the predictive model must recover):
    log_reach = f(followers, 1h engagement velocity, topic volatility,
                  negativity, media attached, verified author, author's prior
                  virality, hour-of-day) + noise
Early engagement (first 60 minutes) is legitimately observable at prediction
time, so it is NOT leakage. The 24h reach is the future we are predicting.
"""
import numpy as np, pandas as pd, random, json, argparse, os
from datetime import datetime, timedelta

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

BRAND = "ZaykaNow"
PLATFORMS = ["x", "instagram", "reddit", "youtube", "playstore"]
PLATFORM_P = [0.46, 0.21, 0.12, 0.07, 0.14]

# topic -> (harm_class, volatility weight in reach model, base rate)
# Base rates are set to what a large Indian food-delivery brand actually sees:
# serious food-safety claims are ~2% of mentions and discrimination allegations
# well under 1%, not the 5%/2% our first draft used. Those inflated rates produced
# 571 "crises" per day, which is not a crisis, it is Tuesday.
# See docs/PROMPT_LOG.md entry 9.
TOPICS = {
    "delivery_delay":           ("H1",  0.10, 0.291),
    "food_safety":              ("H3",  1.25, 0.012),
    "payment_refund":           ("H1",  0.15, 0.126),
    "app_bug":                  ("H1",  0.00, 0.107),
    "rider_behaviour":          ("H2",  0.70, 0.068),
    "pricing_surge":            ("H1",  0.35, 0.078),
    "customer_service":         ("H2",  0.30, 0.087),
    "praise":                   ("H0", -0.25, 0.151),
    "promo_query":              ("H0", -0.40, 0.065),
    "discrimination_allegation":("H3",  1.60, 0.0015),
}
_TP = np.array([TOPICS[t][2] for t in TOPICS], dtype=float); _TP = _TP/_TP.sum()

# ---------------------------------------------------------------- text banks
EN = {
"delivery_delay": [
 "order placed at 8pm still not here, it is 9:15. {b} what is going on",
 "48 minutes late and the app still says 'arriving in 5 mins'. {b} fix your ETA",
 "third late delivery this week from {b}. the food was stone cold",
 "{b} rider has been stuck at the same spot for 25 minutes according to the map",
 "waited 70 mins for a 2km delivery. {b} this is unacceptable on a weekday",
],
"food_safety": [
 "found a piece of plastic in the biryani I ordered from {b} tonight. my daughter nearly ate it",
 "{b} my entire family has been throwing up since dinner. order id in DM. this is not okay",
 "the paneer smelled off and I still ate some. been in the bathroom for 3 hours. {b}",
 "there was a hair baked into the bread. {b} do you inspect these kitchens at all",
 "{b} the chicken was raw in the middle. raw. I have photos",
],
"payment_refund": [
 "{b} charged me twice for one order and the refund has not come in 9 days",
 "cancelled order 40 minutes ago, money still deducted. {b} where is my 640 rupees",
 "{b} refund shows 'processed' but nothing in my account. bank says nothing received",
 "paid by UPI, order failed, amount gone. {b} support keeps closing my ticket",
],
"app_bug": [
 "{b} app crashes every time I open the cart on android 14",
 "cannot apply any coupon, the screen just freezes. {b} latest version",
 "{b} map is showing my address in a different city since the update",
 "login loop on {b} app, OTP never arrives, tried 6 times",
],
"rider_behaviour": [
 "{b} your delivery guy shouted at my mother for not coming down fast enough",
 "rider refused to come to the door and threatened to cancel. {b} this is the second time",
 "{b} delivery person was on a call the whole time and threw the bag at the gate",
 "your rider asked me for extra money in cash on top of the paid order. {b}",
],
"pricing_surge": [
 "{b} same thali was 240 yesterday and 410 today. surge on a tuesday?",
 "delivery fee is now higher than the food. {b} what are we paying for",
 "{b} the 'handling charge' has gone up again and nobody explains what it is",
],
"customer_service": [
 "{b} support chat disconnected 4 times. nobody reads what I type",
 "been emailing {b} for 11 days about one order. automated replies only",
 "{b} your agent closed my complaint without resolving it and told me to reorder",
],
"praise": [
 "honestly {b} delivered in 18 minutes today, rider was lovely. credit where due",
 "{b} packaging has improved a lot, nothing spilled this time",
 "ordered from {b} at 2am and it actually arrived hot. impressed",
 "{b} customer support sorted my issue in one message today. unusual and welcome",
],
"promo_query": [
 "{b} is the ZAYKA50 code still working? it says invalid for me",
 "does {b} have a student discount or did that end",
 "{b} when does the weekend offer start, friday or saturday",
],
"discrimination_allegation": [
 "{b} rider refused to deliver to my address and said he does not go to 'that area'. we know what that means",
 "your restaurant partner asked my friend's religion before accepting the order. {b} explain this",
 "{b} agent mocked my accent and said 'speak proper hindi'. I have the chat screenshot",
],
}

HINGLISH = {
"delivery_delay": [
 "{b} ek ghanta ho gaya bhai, khana kahan hai? order id DM kiya hai",
 "bhai {b} 55 minute se rider wahi khada hai map pe. kya kar rahe ho",
 "{b} thanda khana aaya, 1 hour 10 min late. paise wapas chahiye",
],
"food_safety": [
 "{b} khane me baal nikla, poore ghar ka mood kharab. photo DM kar raha hoon",
 "raat ko {b} se mangaya tha, subah se pet kharab hai. doctor ke paas jaana pad raha hai",
 "{b} chicken andar se kacha tha bhai, ye kya bhej rahe ho",
],
"payment_refund": [
 "{b} paisa cut gaya order cancel ho gaya, 8 din ho gaye refund nahi aaya",
 "bhai {b} double charge hua hai, support koi jawab nahi de raha",
],
"app_bug": [
 "{b} app baar baar band ho raha hai, cart khulta hi nahi",
 "{b} OTP aa hi nahi raha, login nahi ho pa raha",
],
"rider_behaviour": [
 "{b} delivery wale ne gate pe bag feka aur chala gaya. tameez naam ki cheez nahi",
 "{b} rider ne extra cash maanga, already online pay kiya tha",
],
"pricing_surge": [
 "{b} kal 220 ka tha aaj 390. ye surge kis khushi me",
],
"customer_service": [
 "{b} support bot ke alawa koi insaan hai bhi? 9 din se mail kar raha hoon",
],
"praise": [
 "{b} aaj 20 minute me aa gaya, rider bhi bahut acha tha. shabaash",
 "{b} packaging ab sahi hai, kuch nahi gira",
],
"promo_query": [
 "{b} ZAYKA50 code chal raha hai kya? invalid bata raha hai",
],
"discrimination_allegation": [
 "{b} rider ne kaha wo 'us area' me delivery nahi karta. matlab sab samajhte hain",
],
}

# Sarcasm: positive surface words, negative intent, on a complaint topic.
SARCASM = {
"delivery_delay": [
 "wah {b}, sirf 2 ghante. bahut fast. aise hi service dete raho",
 "love how {b} redefines '30 minute delivery' as a lifestyle choice. truly innovative",
 "10/10 to {b} for the thrilling 90 minute wait. very immersive experience",
 "{b} ki speed dekh ke dil khush ho gaya. 85 minute. world record",
],
"food_safety": [
 "thank you {b} for the complimentary hair in my biryani, very generous of you",
 "{b} serving raw chicken now? bold menu expansion. love the innovation",
],
"payment_refund": [
 "{b} keeping my refund for 12 days is a great savings scheme honestly. 0% returns",
 "nice of {b} to charge me twice. buy one get one, but for the charge",
],
"customer_service": [
 "shoutout to {b} support for closing my ticket without reading it. efficiency king",
],
"app_bug": [
 "{b} app crashing on every order is a great way to help me save money. thanks",
],
"pricing_surge": [
 "{b} doubling prices on a tuesday is such a fun surprise. love the spontaneity",
],
}

HANDLES = ["arjun","priya","rohit","sneha","kabir","meera","vikram","ananya","faizan","divya",
 "nikhil","tanvi","aditya","ishita","rahul","zoya","karan","neha","omar","shruti","dev","maya"]
SUFFIX = ["","_official","24x7","_blr","delhi","_mum","writes","eats","_real","1996","xx","hq"]

def make_handle():
    return random.choice(HANDLES) + random.choice(SUFFIX) + (str(random.randint(1,999)) if random.random()<0.4 else "")

def followers_draw(verified):
    # heavy-tailed, verified accounts shifted up
    base = rng.lognormal(mean=6.0 if not verified else 9.3, sigma=1.5 if not verified else 1.1)
    return int(np.clip(base, 12, 3_500_000))

def pick(d, topic):
    return random.choice(d[topic]).format(b=BRAND)

def gen_row(i, start):
    topic = rng.choice(list(TOPICS), p=_TP)
    harm, vol, _ = TOPICS[topic]

    # language mix: 48% english, 38% hinglish, 14% sarcastic (any lang)
    r = random.random()
    sarcastic = 0
    if r < 0.12 and topic in SARCASM:
        text = pick(SARCASM, topic); lang = "hi-en" if random.random()<0.4 else "en"; sarcastic = 1
    elif r < 0.52 and topic in HINGLISH:
        text = pick(HINGLISH, topic); lang = "hi-en"
    else:
        text = pick(EN, topic); lang = "en"

    # sentiment gold: sarcasm is negative in intent
    if topic in ("praise",):
        sentiment = "positive"
    elif topic in ("promo_query",):
        sentiment = "neutral"
    else:
        sentiment = "negative"

    # urgency gold, from harm class and intensity cues
    intensity_words = ("hospital","doctor","raw","vomit","throwing up","police","legal","lawyer",
                       "child","daughter","son","blood","poison","pet kharab","kacha")
    hot = any(w in text.lower() for w in intensity_words)
    if harm == "H3":
        urgency = "critical"
    elif harm == "H2":
        urgency = "high" if hot else "medium"
    elif harm == "H1":
        urgency = "high" if hot else ("medium" if random.random() < 0.55 else "low")
    else:
        urgency = "low"

    verified = int(random.random() < (0.07 if random.random()<0.9 else 0.4))
    followers = followers_draw(verified)
    platform = rng.choice(PLATFORMS, p=PLATFORM_P)
    has_media = int(random.random() < (0.55 if topic in ("food_safety","rider_behaviour") else 0.22))
    prior_viral = float(np.clip(rng.normal(0, 1), -2.5, 3.5))   # z-score of author's past reach
    created = start + timedelta(minutes=int(rng.integers(0, 60*24*21)))
    hour = created.hour
    # prime time 12-14 and 19-23 (meal windows) amplify
    prime = 0.42 if hour in (12,13,19,20,21,22) else (0.15 if hour in (14,18,23) else 0.0)

    neg = 1.0 if sentiment == "negative" else 0.0
    # --- early engagement in first 60 minutes (observable at predict time) ---
    eng_lambda = np.exp(0.40*np.log1p(followers) + 0.9*vol + 0.5*neg + 0.4*has_media
                        + 0.35*prior_viral + prime - 2.7)
    likes_1h   = int(rng.poisson(max(eng_lambda, 0.05)))
    reposts_1h = int(rng.poisson(max(eng_lambda*0.22, 0.02)))
    replies_1h = int(rng.poisson(max(eng_lambda*0.30, 0.02)))

    # --- 24h reach (target) ---
    # Calibrated so ~78% of posts stay in band R0 and ~1.6% reach the R2 viral band,
    # which is what the brand's own historical reach distribution looks like.
    log_reach = (2.40 + 0.90*(
          0.50*np.log1p(followers)
        + 0.42*np.log1p(likes_1h)
        + 0.78*np.log1p(reposts_1h)
        + 0.16*np.log1p(replies_1h)
        + 1.05*vol
        + 0.55*neg
        + 0.38*has_media
        + 0.30*verified
        + 0.34*prior_viral
        + 0.45*prime)
        + rng.normal(0, 1.05))
    reach = int(np.clip(np.exp(log_reach), 5, 6_000_000))

    # redact-able PII injected into ~6% of rows so the redaction step has work to do
    if random.random() < 0.06:
        text += random.choice([" 98{}".format(rng.integers(10000000,99999999)),
                               " order ORD{}".format(rng.integers(100000,999999)),
                               " mail me on {}@example.com".format(random.choice(HANDLES))])

    return dict(
        mention_id=f"ZM{100000+i}",
        created_at=created.isoformat(timespec="seconds"),
        platform=platform, author_handle=make_handle(), followers=followers,
        author_verified=verified, author_prior_virality=round(prior_viral,3),
        lang=lang, text=text, has_media=has_media,
        likes_1h=likes_1h, reposts_1h=reposts_1h, replies_1h=replies_1h,
        hour=hour, topic_gold=topic, harm_class=harm,
        sentiment_gold=sentiment, urgency_gold=urgency, is_sarcastic=sarcastic,
        reach_24h=reach,
    )

def main(n, out):
    start = datetime(2026, 10, 1, 0, 0, 0)
    rows = [gen_row(i, start) for i in range(n)]
    df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    # viral label per policy 2.1: R2 band and above = 25,000+
    df["viral_gold"] = (df.reach_24h >= 25_000).astype(int)

    # Inject a CLUSTER: 9 food_safety mentions inside one 40-minute window, so the
    # cluster-alert rule in policy 2.5 has something real to fire on in the demo.
    burst_t = datetime(2026, 10, 18, 20, 10, 0)
    burst = []
    for k in range(9):
        r = gen_row(90000+k, start)
        r["topic_gold"]="food_safety"; r["harm_class"]="H3"; r["sentiment_gold"]="negative"
        r["urgency_gold"]="critical"; r["is_sarcastic"]=0
        r["text"] = random.choice(EN["food_safety"]).format(b=BRAND)
        r["created_at"] = (burst_t + timedelta(minutes=int(rng.integers(0,40)))).isoformat(timespec="seconds")
        r["mention_id"] = f"ZMB{k:03d}"
        burst.append(r)
    bdf = pd.DataFrame(burst); bdf["viral_gold"]=(bdf.reach_24h>=25000).astype(int)
    df = pd.concat([df,bdf]).sort_values("created_at").reset_index(drop=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"rows={len(df)}  viral_rate={df.viral_gold.mean():.3f}")
    print(df.topic_gold.value_counts().to_string())
    print("\nlang:", df.lang.value_counts().to_dict(), " sarcastic:", int(df.is_sarcastic.sum()))
    print("reach pct:", {p: int(df.reach_24h.quantile(p/100)) for p in (50,75,90,95,99)})
    bands = pd.cut(df.reach_24h, [-1,1999,24999,99999,10**9], labels=["R0","R1","R2","R3"])
    print("reach bands:", bands.value_counts().reindex(["R0","R1","R2","R3"]).to_dict())
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--out", default="data/mentions.csv")
    a = ap.parse_args()
    main(a.n, a.out)

"""Feature engineering shared by training and the live app.

Rule: every feature here must be computable 60 minutes after a post goes up.
Nothing may read reach_24h or any gold label. If you add a feature, ask "would a
real analyst have this at 9:05pm for a post made at 8:05pm?" If no, drop it.
"""
import numpy as np, pandas as pd

PLATFORMS = ["x", "instagram", "reddit", "youtube", "playstore"]
TOPICS = ["delivery_delay","food_safety","payment_refund","app_bug","rider_behaviour",
          "pricing_surge","customer_service","praise","promo_query","discrimination_allegation"]
PRIME_HOURS = {12,13,19,20,21,22}

FEATURE_ORDER = (
    ["log_followers","log_likes_1h","log_reposts_1h","log_replies_1h",
     "repost_ratio","reply_ratio","engagement_rate_1h","total_eng_1h",
     "author_verified","has_media","author_prior_virality","is_prime_hour",
     "is_negative","text_len","has_exclaim","has_question","is_hinglish"]
    + [f"plat_{p}" for p in PLATFORMS]
    + [f"topic_{t}" for t in TOPICS]
)

def build_features(df: pd.DataFrame, topic_col="topic_gold", sent_col="sentiment_gold") -> pd.DataFrame:
    """df needs: followers, likes_1h, reposts_1h, replies_1h, author_verified,
    has_media, author_prior_virality, hour, platform, text, lang, <topic>, <sent>."""
    X = pd.DataFrame(index=df.index)
    f   = df.followers.astype(float)
    lk  = df.likes_1h.astype(float)
    rp  = df.reposts_1h.astype(float)
    rl  = df.replies_1h.astype(float)
    tot = lk + rp + rl

    X["log_followers"]   = np.log1p(f)
    X["log_likes_1h"]    = np.log1p(lk)
    X["log_reposts_1h"]  = np.log1p(rp)
    X["log_replies_1h"]  = np.log1p(rl)
    # ratios capture *shape* of engagement: reposts are the true spread signal
    X["repost_ratio"]    = rp / (tot + 1)
    X["reply_ratio"]     = rl / (tot + 1)
    X["engagement_rate_1h"] = tot / (f + 1)        # small account, big reaction = breakout
    X["total_eng_1h"]    = np.log1p(tot)
    X["author_verified"] = df.author_verified.astype(int)
    X["has_media"]       = df.has_media.astype(int)
    X["author_prior_virality"] = df.author_prior_virality.astype(float)
    X["is_prime_hour"]   = df.hour.astype(int).isin(PRIME_HOURS).astype(int)
    X["is_negative"]     = (df[sent_col] == "negative").astype(int)

    t = df.text.astype(str)
    X["text_len"]     = t.str.len().clip(0, 400) / 400.0
    X["has_exclaim"]  = t.str.contains("!", regex=False).astype(int)
    X["has_question"] = t.str.contains(r"\?", regex=True).astype(int)
    X["is_hinglish"]  = (df.lang.astype(str) != "en").astype(int)

    for p in PLATFORMS:
        X[f"plat_{p}"] = (df.platform.astype(str) == p).astype(int)
    for tp in TOPICS:
        X[f"topic_{tp}"] = (df[topic_col].astype(str) == tp).astype(int)

    return X[FEATURE_ORDER].astype(float)

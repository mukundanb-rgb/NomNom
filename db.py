"""
Approval queue + immutable audit log (policy 7.1-7.3).

SQLite because this is a prototype that must run on one laptop with no services.
The audit table is append-only by construction: nothing in this module issues an
UPDATE or DELETE against it. Queue state changes are new audit rows, so the
history of who approved what, when, and with which model version survives even
if the queue row is later edited.
"""
from __future__ import annotations
import sqlite3, json, os
from datetime import datetime, timezone
from typing import Optional, List, Dict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "pulseguard.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS queue (
  mention_id TEXT PRIMARY KEY,
  created_at TEXT, platform TEXT, author_handle TEXT, followers INTEGER,
  text TEXT, lang TEXT,
  pred_topic TEXT, pred_sentiment TEXT, pred_urgency TEXT, pred_confidence REAL,
  is_sarcastic INTEGER,
  predicted_reach REAL, p_viral REAL,
  severity TEXT, harm_class TEXT, band TEXT, is_crisis INTEGER,
  routed_to_human INTEGER, review_reason TEXT,
  draft TEXT, draft_blocked INTEGER, draft_engine TEXT,
  checks_json TEXT, citations_json TEXT, reasons_json TEXT,
  required_approver TEXT, sla TEXT,
  status TEXT DEFAULT 'PENDING',
  model_version TEXT, classifier_engine TEXT
);
CREATE TABLE IF NOT EXISTS audit (
  audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, mention_id TEXT, actor TEXT, actor_role TEXT,
  action TEXT, from_status TEXT, to_status TEXT,
  severity TEXT, model_version TEXT, classifier_engine TEXT,
  text_as_sent TEXT, note TEXT
);
CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status);
CREATE INDEX IF NOT EXISTS idx_queue_crisis ON queue(is_crisis);
CREATE INDEX IF NOT EXISTS idx_audit_mention ON audit(mention_id);
"""

def connect(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def upsert_queue(con: sqlite3.Connection, row: Dict) -> None:
    cols = ",".join(row.keys()); ph = ",".join("?" * len(row))
    con.execute(f"INSERT OR REPLACE INTO queue ({cols}) VALUES ({ph})", list(row.values()))
    con.commit()

def log(con: sqlite3.Connection, mention_id: str, actor: str, actor_role: str, action: str,
        from_status: Optional[str], to_status: Optional[str], severity: str = "",
        model_version: str = "", classifier_engine: str = "",
        text_as_sent: str = "", note: str = "") -> None:
    con.execute("""INSERT INTO audit (ts,mention_id,actor,actor_role,action,from_status,
                   to_status,severity,model_version,classifier_engine,text_as_sent,note)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now(), mention_id, actor, actor_role, action, from_status, to_status,
                 severity, model_version, classifier_engine, text_as_sent, note))
    con.commit()

def set_status(con: sqlite3.Connection, mention_id: str, to_status: str, actor: str,
               actor_role: str, text_as_sent: str = "", note: str = "") -> Dict:
    cur = con.execute("SELECT status, severity, model_version, classifier_engine, draft, required_approver "
                      "FROM queue WHERE mention_id=?", (mention_id,)).fetchone()
    if cur is None:
        raise KeyError(mention_id)
    frm = cur["status"]
    # policy 7.1: only the role named in 2.4 (or above) may move a draft out of PENDING
    con.execute("UPDATE queue SET status=? WHERE mention_id=?", (to_status, mention_id))
    con.commit()
    log(con, mention_id, actor, actor_role, f"STATUS_{to_status}", frm, to_status,
        cur["severity"] or "", cur["model_version"] or "", cur["classifier_engine"] or "",
        text_as_sent or (cur["draft"] or ""), note)
    return {"from": frm, "to": to_status, "required_approver": cur["required_approver"]}

def fetch(con, where: str = "", params: tuple = (), limit: int = 500) -> List[sqlite3.Row]:
    q = "SELECT * FROM queue"
    if where: q += f" WHERE {where}"
    q += f" ORDER BY is_crisis DESC, severity ASC, created_at DESC LIMIT {int(limit)}"
    return con.execute(q, params).fetchall()

def audit_rows(con, mention_id: Optional[str] = None, limit: int = 500) -> List[sqlite3.Row]:
    if mention_id:
        return con.execute("SELECT * FROM audit WHERE mention_id=? ORDER BY audit_id DESC LIMIT ?",
                           (mention_id, limit)).fetchall()
    return con.execute("SELECT * FROM audit ORDER BY audit_id DESC LIMIT ?", (limit,)).fetchall()

def counts(con) -> Dict:
    r = {k: 0 for k in ("PENDING","APPROVED","REJECTED","EDITED_APPROVED")}
    for row in con.execute("SELECT status, COUNT(*) n FROM queue GROUP BY status"):
        r[row["status"]] = row["n"]
    r["crises"] = con.execute("SELECT COUNT(*) n FROM queue WHERE is_crisis=1").fetchone()["n"]
    r["blocked_drafts"] = con.execute("SELECT COUNT(*) n FROM queue WHERE draft_blocked=1").fetchone()["n"]
    r["awaiting_human_review"] = con.execute("SELECT COUNT(*) n FROM queue WHERE routed_to_human=1").fetchone()["n"]
    return r

def reset(path: str = DB_PATH) -> None:
    if os.path.exists(path): os.remove(path)

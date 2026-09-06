"""Formula corpus store (SQLite). Papers -> admissibility triage -> extracted
formulas tagged to one of the 23 divisions.

Kept deliberately small in this pass (S2 limited): 4 priority families only,
target ~50-70 papers total. No vector index yet - lexical search is enough at
this size.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

DB = Path(__file__).resolve().parent / "formulas.sqlite"

DIVISIONS = [
    "F_VOL", "F_DIR", "F_ENTRY", "F_EXIT", "F_STATE", "F_SIZE", "F_FILTER",
    "F_TRANSFORM", "F_DIST", "F_DEP", "F_PATH", "F_MULTI", "F_FORMULA", "F_META",
    "F_EXEC", "F_FLOW", "F_BOOK", "F_LIQ", "F_FUND", "F_OI", "F_LIQD", "F_XSEC",
    "F_XCOIN",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  id TEXT PRIMARY KEY, ts REAL,
  title TEXT, authors TEXT, year INTEGER,
  identifier TEXT,            -- DOI or arXiv id
  id_kind TEXT,               -- doi|arxiv|ssrn|repec|url
  resolved INTEGER,           -- identifier resolves
  retrieved INTEGER,          -- full text obtained (0 => UNRETRIEVED)
  oa_url TEXT,
  venue TEXT,
  abstract TEXT,
  admissibility TEXT,         -- ADMISSIBLE|REJECT_DATA|REJECT_HORIZON|PARKED
  admissibility_reason TEXT,
  falsification TEXT,         -- notes from replication/contradiction search
  scan_depth TEXT             -- title|abstract|fulltext
);
CREATE TABLE IF NOT EXISTS formulas (
  id TEXT PRIMARY KEY, ts REAL,
  paper_id TEXT,
  division TEXT,              -- one of DIVISIONS
  name TEXT,
  equation TEXT,              -- the formula, plain text / latex-ish
  inputs TEXT,                -- JSON list of required inputs
  original_horizon TEXT,
  data_ok INTEGER,            -- computable from our data
  notes TEXT
);
"""


class Corpus:
    def __init__(self, path=None):
        self.conn = sqlite3.connect(str(path or DB))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add_paper(self, **k) -> str:
        pid = k.get("id") or ("P-" + uuid.uuid4().hex[:8])
        cols = ["id", "ts", "title", "authors", "year", "identifier", "id_kind",
                "resolved", "retrieved", "oa_url", "venue", "abstract",
                "admissibility", "admissibility_reason", "falsification", "scan_depth"]
        row = {c: k.get(c) for c in cols}
        row["id"] = pid
        row["ts"] = time.time()
        for b in ("resolved", "retrieved"):
            row[b] = int(bool(row[b]))
        ph = ",".join("?" for _ in cols)
        self.conn.execute(f"INSERT OR REPLACE INTO papers ({','.join(cols)}) VALUES ({ph})",
                          [row[c] for c in cols])
        self.conn.commit()
        return pid

    def add_formula(self, paper_id, division, name, equation, inputs, *,
                    original_horizon=None, data_ok=True, notes=None) -> str:
        assert division in DIVISIONS, division
        fid = "FML-" + uuid.uuid4().hex[:8]
        self.conn.execute(
            "INSERT INTO formulas VALUES (?,?,?,?,?,?,?,?,?,?)",
            (fid, time.time(), paper_id, division, name, equation,
             json.dumps(inputs), original_horizon, int(bool(data_ok)), notes))
        self.conn.commit()
        return fid

    def counts(self) -> dict:
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*), SUM(retrieved), "
                  "SUM(admissibility='ADMISSIBLE'), SUM(admissibility='PARKED') FROM papers")
        p = c.fetchone()
        c.execute("SELECT division, COUNT(*) FROM formulas GROUP BY division")
        by_div = dict(c.fetchall())
        c.execute("SELECT admissibility, COUNT(*) FROM papers GROUP BY admissibility")
        by_adm = dict(c.fetchall())
        return {"papers": p[0], "retrieved": p[1], "admissible": p[2], "parked": p[3],
                "papers_by_admissibility": by_adm, "formulas_by_division": by_div,
                "total_formulas": sum(by_div.values())}

    def formulas(self, division=None, data_ok_only=True):
        q = "SELECT f.*, p.title, p.identifier FROM formulas f JOIN papers p ON p.id=f.paper_id WHERE 1=1"
        a = []
        if division:
            q += " AND f.division=?"
            a.append(division)
        if data_ok_only:
            q += " AND f.data_ok=1"
        return self.conn.execute(q, a).fetchall()


if __name__ == "__main__":
    print(Corpus().counts())

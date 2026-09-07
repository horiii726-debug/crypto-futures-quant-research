"""Append-only research ledger (SQLite).

R1  LLM never produces numbers. Every numeric claim traces to a [EXP-xxxxxx]
    row here.
R3  Trial count is computed by THIS module from the ledger, never reported by
    an agent. dsr.py reads it from here.
R6  Failures are never deleted. UPDATE and DELETE are rejected by triggers on
    every result table.

Usage:
    from lab.ledger import Ledger
    L = Ledger()                       # uses $CRYPTO_LAB_LEDGER_SQLITE
    exp = L.new_experiment(hyp_id, family="F_XSEC", spec={...})
    L.log_trial(exp, config={...}, status="planned")   # BEFORE the fit
    L.log_metric(exp, "sharpe_oos", 1.23, trial_id=tid)
    n = L.trial_count(family="F_XSEC")  # machine-counted
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / "ledger" / "lab.sqlite"

# Tables whose rows are results: append-only, no UPDATE, no DELETE.
RESULT_TABLES = [
    "sources",
    "hypotheses",
    "datasets",
    "trials",
    "experiments",
    "metrics",
    "verdicts",
    "lessons",
    "test_looks",
    "bounds",
    "universe",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id           TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    kind         TEXT NOT NULL,          -- arxiv|ssrn|openalex|s2|repec|...
    identifier   TEXT NOT NULL,          -- DOI / arXiv id / OpenAlex id
    resolved     INTEGER NOT NULL,       -- 1 if identifier resolved, else 0
    retrieved    INTEGER NOT NULL,       -- 1 if full text retrieved, else UNRETRIEVED
    falsification_done INTEGER NOT NULL DEFAULT 0,
    title        TEXT,
    payload      TEXT                    -- JSON
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id           TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    family       TEXT NOT NULL,
    claim        TEXT NOT NULL,
    mechanism    TEXT NOT NULL,
    math_form    TEXT NOT NULL,
    target       TEXT NOT NULL,
    horizon      TEXT NOT NULL,
    null_hypothesis TEXT NOT NULL,
    effect_size_of_interest REAL,
    lessons_reviewed INTEGER NOT NULL DEFAULT 0,
    parent_id    TEXT,                   -- R4: param change => new HYP, links here
    payload      TEXT                    -- JSON: full schema-validated record
);

CREATE TABLE IF NOT EXISTS datasets (
    id           TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    partition    TEXT NOT NULL,          -- raw|processed|train|valid|test
    symbol_scope TEXT NOT NULL,
    n_rows       INTEGER,
    sha256       TEXT NOT NULL,
    t_start      TEXT,
    t_end        TEXT,
    payload      TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    id           TEXT PRIMARY KEY,       -- EXP-xxxxxx
    ts           REAL NOT NULL,
    hyp_id       TEXT NOT NULL,
    family       TEXT NOT NULL,
    spec_sha     TEXT NOT NULL,
    spec         TEXT NOT NULL,          -- JSON: pre-registered grid + config
    stage        TEXT NOT NULL DEFAULT 'discovery',  -- discovery|validation
    status       TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS trials (
    id           TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    exp_id       TEXT NOT NULL,
    family       TEXT NOT NULL,
    config       TEXT NOT NULL,          -- JSON: the exact parameter point
    config_sha   TEXT NOT NULL,
    stage        TEXT NOT NULL,          -- discovery|validation
    status       TEXT NOT NULL,          -- planned|running|done|failed
    counts_as_trial INTEGER NOT NULL DEFAULT 1   -- R3: multiplicity accounting
);

CREATE TABLE IF NOT EXISTS metrics (
    id           TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    exp_id       TEXT NOT NULL,
    trial_id     TEXT,
    name         TEXT NOT NULL,
    value        REAL,
    text_value   TEXT,
    context      TEXT                    -- JSON: fold, partition, etc.
);

CREATE TABLE IF NOT EXISTS verdicts (
    id           TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    subject_id   TEXT NOT NULL,          -- hyp_id or exp_id
    gate         TEXT NOT NULL,          -- G0..G10 or 'final'
    status       TEXT NOT NULL,          -- PASS|FAIL|UNDERPOWERED|UNKNOWN|VETO
    by_role      TEXT NOT NULL,
    evidence_exp_ids TEXT NOT NULL,      -- JSON list, >=1
    rationale    TEXT NOT NULL,
    strongest_surviving_objection TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lessons (
    id           TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    subject_id   TEXT,
    family       TEXT,
    root_cause   TEXT NOT NULL,
    lesson       TEXT NOT NULL,
    ladder_level TEXT                    -- L1..L6
);

CREATE TABLE IF NOT EXISTS test_looks (
    id           TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    family       TEXT NOT NULL,
    exp_id       TEXT NOT NULL,
    dataset_sha  TEXT NOT NULL,
    approved_by  TEXT NOT NULL,          -- human sign-off token
    note         TEXT
);

CREATE TABLE IF NOT EXISTS bounds (
    id           TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    family       TEXT NOT NULL,
    objective    TEXT NOT NULL,
    bound_stmt   TEXT NOT NULL,          -- what was ruled out, with numbers
    evidence_exp_ids TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS universe (
    id           TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    as_of        TEXT NOT NULL,          -- rebalance date
    symbol       TEXT NOT NULL,
    status       TEXT NOT NULL,          -- in|out|delisted
    reason       TEXT
);
"""


def _guard_triggers() -> str:
    stmts = []
    for t in RESULT_TABLES:
        stmts.append(
            f"CREATE TRIGGER IF NOT EXISTS no_update_{t} "
            f"BEFORE UPDATE ON {t} BEGIN "
            f"SELECT RAISE(ABORT, 'ledger is append-only: UPDATE on {t} rejected (R6)'); END;"
        )
        stmts.append(
            f"CREATE TRIGGER IF NOT EXISTS no_delete_{t} "
            f"BEFORE DELETE ON {t} BEGIN "
            f"SELECT RAISE(ABORT, 'ledger is append-only: DELETE on {t} rejected (R6)'); END;"
        )
    return "\n".join(stmts)


def _now() -> float:
    return time.time()


def _sha(obj) -> str:
    import hashlib

    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _exp_id() -> str:
    return "EXP-" + uuid.uuid4().hex[:6]


class Ledger:
    def __init__(self, path: str | os.PathLike | None = None):
        env = os.environ.get("CRYPTO_LAB_LEDGER_SQLITE") or os.environ.get("XAU_LAB_LEDGER_SQLITE")
        self.path = Path(path or env or DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.executescript(SCHEMA)
        self.conn.executescript(_guard_triggers())
        self.conn.commit()

    # ---- generic append -------------------------------------------------
    def _insert(self, table: str, row: dict) -> str:
        cols = ",".join(row.keys())
        ph = ",".join("?" for _ in row)
        self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({ph})", tuple(row.values()))
        self.conn.commit()
        return row["id"]

    # ---- sources ------------------------------------------------------
    def add_source(self, kind, identifier, *, resolved, retrieved,
                   falsification_done=False, title=None, payload=None) -> str:
        return self._insert("sources", dict(
            id="SRC-" + uuid.uuid4().hex[:8], ts=_now(), kind=kind,
            identifier=identifier, resolved=int(resolved), retrieved=int(retrieved),
            falsification_done=int(falsification_done), title=title,
            payload=json.dumps(payload or {}),
        ))

    # ---- hypotheses -------------------------------------------------------
    def add_hypothesis(self, record: dict) -> str:
        hid = "HYP-" + uuid.uuid4().hex[:8]
        return self._insert("hypotheses", dict(
            id=hid, ts=_now(),
            family=record["target"] if False else record.get("family", "UNSET"),
            claim=record["claim"], mechanism=record["mechanism"],
            math_form=record["math_form"], target=record["target"],
            horizon=record["horizon"], null_hypothesis=record["null_hypothesis"],
            effect_size_of_interest=record.get("effect_size_of_interest"),
            lessons_reviewed=int(bool(record.get("lessons_reviewed"))),
            parent_id=record.get("parent_id"),
            payload=json.dumps(record),
        ))

    # ---- datasets ------------------------------------------------------
    def add_dataset(self, partition, symbol_scope, sha256, *, n_rows=None,
                    t_start=None, t_end=None, payload=None) -> str:
        return self._insert("datasets", dict(
            id="DS-" + uuid.uuid4().hex[:8], ts=_now(), partition=partition,
            symbol_scope=symbol_scope, n_rows=n_rows, sha256=sha256,
            t_start=t_start, t_end=t_end, payload=json.dumps(payload or {}),
        ))

    # ---- experiments ---------------------------------------------------
    def new_experiment(self, hyp_id, family, spec: dict, stage="discovery") -> str:
        eid = _exp_id()
        self._insert("experiments", dict(
            id=eid, ts=_now(), hyp_id=hyp_id, family=family,
            spec_sha=_sha(spec), spec=json.dumps(spec), stage=stage, status="open",
        ))
        return eid

    def close_experiment(self, exp_id, status="done"):
        # append-only: record status change as a metric, do not UPDATE the row
        self.log_metric(exp_id, "experiment_status", None, text_value=status)

    # ---- trials --------------------------------------------------------
    def log_trial(self, exp_id, config: dict, *, stage="discovery",
                  status="planned", counts_as_trial=True) -> str:
        tid = "T-" + uuid.uuid4().hex[:10]
        self._insert("trials", dict(
            id=tid, ts=_now(), exp_id=exp_id,
            family=self._exp_family(exp_id), config=json.dumps(config),
            config_sha=_sha(config), stage=stage, status=status,
            counts_as_trial=int(bool(counts_as_trial)),
        ))
        return tid

    def mark_trial(self, trial_id, status):
        # append-only: log a transition metric instead of UPDATE
        self.log_metric(self._trial_exp(trial_id), "trial_status", None,
                        text_value=status, context={"trial_id": trial_id})

    def log_screen_eval(self, exp_id, config: dict, metrics: dict) -> str:
        """SYSTEM_SPEC [FIX 1]: a cheap-screen evaluation. Logged for honesty
        (stage='screen', counts_as_trial=0) but NOT counted by trial_count() /
        DSR. Only pre-registered full evaluations are trials."""
        tid = "T-" + uuid.uuid4().hex[:10]
        self._insert("trials", dict(
            id=tid, ts=_now(), exp_id=exp_id, family=self._exp_family(exp_id),
            config=json.dumps(config), config_sha=_sha(config),
            stage="screen", status="done", counts_as_trial=0,
        ))
        for k, v in (metrics or {}).items():
            if isinstance(v, (int, float)) and v == v:
                self.log_metric(exp_id, f"screen_{k}", float(v), trial_id=tid)
        return tid

    def screen_count(self, *, family=None) -> int:
        q = "SELECT COUNT(*) FROM trials WHERE stage='screen'"
        a = []
        if family:
            q += " AND family=?"
            a.append(family)
        cur = self.conn.cursor()
        cur.execute(q, tuple(a))
        return int(cur.fetchone()[0])

    # ---- metrics -----------------------------------------------------
    def log_metric(self, exp_id, name, value, *, trial_id=None, text_value=None,
                   context=None) -> str:
        return self._insert("metrics", dict(
            id="M-" + uuid.uuid4().hex[:10], ts=_now(), exp_id=exp_id,
            trial_id=trial_id, name=name,
            value=None if value is None else float(value),
            text_value=text_value, context=json.dumps(context or {}),
        ))

    # ---- verdicts ----------------------------------------------------
    def add_verdict(self, subject_id, gate, status, by_role, evidence_exp_ids,
                    rationale, strongest_surviving_objection) -> str:
        if not evidence_exp_ids:
            raise ValueError("verdict requires >=1 evidence_exp_id (verdict.schema)")
        return self._insert("verdicts", dict(
            id="V-" + uuid.uuid4().hex[:8], ts=_now(), subject_id=subject_id,
            gate=gate, status=status, by_role=by_role,
            evidence_exp_ids=json.dumps(list(evidence_exp_ids)),
            rationale=rationale,
            strongest_surviving_objection=strongest_surviving_objection,
        ))

    # ---- lessons ----------------------------------------------------
    def add_lesson(self, root_cause, lesson, *, subject_id=None, family=None,
                   ladder_level=None) -> str:
        return self._insert("lessons", dict(
            id="L-" + uuid.uuid4().hex[:8], ts=_now(), subject_id=subject_id,
            family=family, root_cause=root_cause, lesson=lesson,
            ladder_level=ladder_level,
        ))

    def query_lessons(self, family=None):
        cur = self.conn.cursor()
        if family:
            cur.execute("SELECT root_cause, lesson, ladder_level FROM lessons "
                        "WHERE family=? OR family IS NULL ORDER BY ts", (family,))
        else:
            cur.execute("SELECT root_cause, lesson, ladder_level FROM lessons ORDER BY ts")
        return cur.fetchall()

    # ---- test looks (R2) -------------------------------------------------
    def record_test_look(self, family, exp_id, dataset_sha, approved_by, note=None) -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM test_looks WHERE family=?", (family,))
        if cur.fetchone()[0] >= 1:
            raise PermissionError(
                f"R2: family {family} has already consumed its single test look")
        return self._insert("test_looks", dict(
            id="TL-" + uuid.uuid4().hex[:8], ts=_now(), family=family,
            exp_id=exp_id, dataset_sha=dataset_sha, approved_by=approved_by, note=note,
        ))

    def test_looks_used(self, family) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM test_looks WHERE family=?", (family,))
        return cur.fetchone()[0]

    # ---- bounds ----------------------------------------------------
    def add_bound(self, family, objective, bound_stmt, evidence_exp_ids) -> str:
        return self._insert("bounds", dict(
            id="B-" + uuid.uuid4().hex[:8], ts=_now(), family=family,
            objective=objective, bound_stmt=bound_stmt,
            evidence_exp_ids=json.dumps(list(evidence_exp_ids)),
        ))

    # ---- universe ------------------------------------------------------
    def add_universe_row(self, as_of, symbol, status, reason=None) -> str:
        return self._insert("universe", dict(
            id="U-" + uuid.uuid4().hex[:10], ts=_now(), as_of=as_of,
            symbol=symbol, status=status, reason=reason,
        ))

    # ---- R3: machine-counted trial totals ------------------------------
    def trial_count(self, *, family=None, hyp_id=None, stage=None,
                    counts_only=True) -> int:
        """The single source of truth for multiplicity. dsr.py calls this."""
        q = "SELECT COUNT(*) FROM trials WHERE 1=1"
        args = []
        if counts_only:
            q += " AND counts_as_trial=1"
        if family:
            q += " AND family=?"
            args.append(family)
        if stage:
            q += " AND stage=?"
            args.append(stage)
        if hyp_id:
            q += " AND exp_id IN (SELECT id FROM experiments WHERE hyp_id=?)"
            args.append(hyp_id)
        cur = self.conn.cursor()
        cur.execute(q, tuple(args))
        return int(cur.fetchone()[0])

    # ---- helpers -----------------------------------------------------
    def _exp_family(self, exp_id) -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT family FROM experiments WHERE id=?", (exp_id,))
        row = cur.fetchone()
        return row[0] if row else "UNSET"

    def _trial_exp(self, trial_id) -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT exp_id FROM trials WHERE id=?", (trial_id,))
        row = cur.fetchone()
        return row[0] if row else None

    def experiment_ids(self, hyp_id=None):
        cur = self.conn.cursor()
        if hyp_id:
            cur.execute("SELECT id FROM experiments WHERE hyp_id=?", (hyp_id,))
        else:
            cur.execute("SELECT id FROM experiments")
        return [r[0] for r in cur.fetchall()]

    def metric_values(self, name, *, exp_id=None):
        cur = self.conn.cursor()
        if exp_id:
            cur.execute("SELECT value FROM metrics WHERE name=? AND exp_id=? "
                        "AND value IS NOT NULL", (name, exp_id))
        else:
            cur.execute("SELECT value FROM metrics WHERE name=? AND value IS NOT NULL",
                        (name,))
        return [r[0] for r in cur.fetchall()]

    def has_experiment(self, exp_id) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM experiments WHERE id=?", (exp_id,))
        return cur.fetchone() is not None

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    L = Ledger()
    print("ledger at", L.path)
    print("result tables:", ", ".join(RESULT_TABLES))
    print("trial_count(all) =", L.trial_count())

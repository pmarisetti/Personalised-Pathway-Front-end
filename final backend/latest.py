import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, g
from flask_cors import CORS

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "learning_sessions.db"
QUESTIONS_PATH = BASE_DIR / "questions.json"
MATERIALS_PATH = BASE_DIR / "materials.json"
FEEDBACK_PATH = BASE_DIR / "feedback_questions.json"

# Columns stored as JSON strings in DB that should be parsed on read
JSON_COLS = ("preferences", "answers_by_module", "results_json", "feedback_json")

# ------------------------------------------------------------------------------
# Load JSON configuration (questions & materials)
# ------------------------------------------------------------------------------
def load_json_or_500(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"{name} file not found at {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to parse {name}: {e}")

try:
    QUESTIONS = load_json_or_500(QUESTIONS_PATH, "questions.json")
    MATERIALS = load_json_or_500(MATERIALS_PATH, "materials.json")
    FEEDBACK = load_json_or_500(FEEDBACK_PATH, "feedback_questions.json")
except RuntimeError as e:
    QUESTIONS = None
    MATERIALS = None
    FEEDBACK = None
    _startup_error = str(e)
else:
    _startup_error = None

# ------------------------------------------------------------------------------
# Materials helpers
# ------------------------------------------------------------------------------
def get_phases():
    return MATERIALS["meta"]["phases"]

PHASE_DESCRIPTIONS = {
    "Absorb":  "Take in new ideas via articles, videos, lectures, and docs.",
    "Build":   "Practice with guided, structured work and exercises.",
    "Connect": "Discuss, peer-review, and collaborate to deepen understanding.",
    "Learn":   "Explore and investigate real-world problems and case studies.",
    "Do":      "Apply skills hands-on (projects, deployment, production).",
}

def get_modules_blob():
    return MATERIALS["modules"]  # dict: module -> level -> phase -> [items]

def get_stakeholder_supplements():
    return MATERIALS.get("stakeholder_resources", {})

# ------------------------------------------------------------------------------
# Preferences → Phase mapping
# ------------------------------------------------------------------------------
PREF_TO_PHASES = {
    "Step-by-step guided tutorials with clear instructions": {"primary": ["Absorb"], "secondary": ["Build"]},
    "Hands-on practice with real datasets and examples": {"primary": ["Do"], "secondary": ["Build"]},
    "Interactive problem-solving with immediate feedback": {"primary": ["Do"], "secondary": ["Connect"]},
    "Self-directed exploration with resources and references": {"primary": ["Learn"], "secondary": ["Absorb"]},
    "Visual presentations and diagrams": {"primary": ["Absorb"], "secondary": ["Connect"]},
    "Written explanations and documentation": {"primary": ["Absorb"], "secondary": ["Learn"]},
    "Video demonstrations and walkthroughs": {"primary": ["Absorb"], "secondary": ["Do"]},
    "Interactive exercises and simulations": {"primary": ["Do"], "secondary": ["Build"]},
    "Working through structured exercises with solutions": {"primary": ["Build"], "secondary": ["Absorb"]},
    "Applying methods to real healthcare case studies": {"primary": ["Learn"], "secondary": ["Do"]},
    "Collaborating with peers on group projects": {"primary": ["Connect"], "secondary": ["Build"]},
    "Independent research and experimentation": {"primary": ["Learn"], "secondary": ["Build"]},
}

def phase_weights_from_preferences(preferences):
    weights = {p: 0 for p in get_phases()}
    for pref in preferences or []:
        mp = PREF_TO_PHASES.get(pref, {})
        for p in mp.get("primary", []):
            if p in weights:
                weights[p] += 2
        for p in mp.get("secondary", []):
            if p in weights:
                weights[p] += 1
    return weights

# ------------------------------------------------------------------------------
# DB (fresh schema; delete learning_sessions.db before first run)
# ------------------------------------------------------------------------------
def get_db():
    db = getattr(g, "_db", None)
    if db is None:
        db = g._db = sqlite3.connect(str(DB_PATH))
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(_exc):
    db = getattr(g, "_db", None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    db.executescript(
        """
        PRAGMA journal_mode = WAL;

        CREATE TABLE IF NOT EXISTS sessions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          TEXT UNIQUE NOT NULL,         -- logical ID from FE
            stakeholder         TEXT NOT NULL,                -- student|academic|healthcare_professional|lay_representative
            preferences         TEXT NOT NULL,                -- JSON array
            answers_by_module   TEXT NOT NULL,                -- JSON object {module:[1..5]}
            results_json        TEXT NOT NULL,                -- JSON object (computed)
            feedback_rating     INTEGER,                      -- optional 1..5
            feedback_comments   TEXT,                         -- optional free text
            feedback_json       TEXT,                         -- optional structured feedback
            completed_at        TIMESTAMP NOT NULL,           -- ISO string
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_created_at    ON sessions(created_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_completed_at  ON sessions(completed_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_stakeholder   ON sessions(stakeholder);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_sid    ON sessions(session_id);
        """
    )
    db.commit()

with app.app_context():
    init_db()

# ------------------------------------------------------------------------------
# Level computation
# ------------------------------------------------------------------------------
def compute_level_auto(score, per_module_q_count):
    qmin, qmax = per_module_q_count * 1, per_module_q_count * 5
    t1 = qmin + 0.35 * (qmax - qmin)   # beginner upper bound
    t2 = qmin + 0.725 * (qmax - qmin)  # intermediate upper bound
    if score <= t1:
        return "beginner"
    if score <= t2:
        return "intermediate"
    return "expert"

def get_timeline_recommendation(level, stakeholder_key):
    stakeholder_key = (stakeholder_key or "").replace("-", "_").strip()
    table = {
        "beginner": {
            "student": "12-16 weeks (1 semester)",
            "academic": "8-12 weeks (intensive)",
            "healthcare_professional": "16-20 weeks (part-time)",
            "lay_representative": "20-24 weeks (self-paced)",
        },
        "intermediate": {
            "student": "8-12 weeks (accelerated)",
            "academic": "6-8 weeks (focused)",
            "healthcare_professional": "12-16 weeks (part-time)",
            "lay_representative": "16-20 weeks (self-paced)",
        },
        "expert": {
            "student": "4-8 weeks (intensive)",
            "academic": "4-6 weeks (research-focused)",
            "healthcare_professional": "8-12 weeks (specialized)",
            "lay_representative": "12-16 weeks (advanced topics)",
        },
    }
    return table.get(level, {}).get(stakeholder_key, "8-12 weeks")

def get_estimated_hours(level):
    return {
        "beginner": "120-150 hours",
        "intermediate": "80-120 hours",
        "expert": "60-80 hours",
    }.get(level, "100 hours")

# ------------------------------------------------------------------------------
# Resource collection per module/level/phase
# ------------------------------------------------------------------------------
def collect_resources_for_module(module_key, level, stakeholder_key, preferences, top_k=10):
    modules_blob = get_modules_blob()
    module_def = modules_blob.get(module_key, {})
    level_def = module_def.get(level, {})

    supplements = get_stakeholder_supplements().get(stakeholder_key, {}).get("supplementary", [])

    phase_weights = phase_weights_from_preferences(preferences)
    phases_order = get_phases()
    phase_index = {p: i for i, p in enumerate(phases_order)}

    blocks = []
    for phase in phases_order:
        phase_items = list(level_def.get(phase, []))
        if not phase_items:
            continue

        enriched = phase_items + supplements

        w = phase_weights.get(phase, 0)
        sorted_items = sorted(enriched, key=lambda it: (-w, (it.get("title") or "").lower()))

        curated = []
        for it in sorted_items[:top_k]:
            curated.append({
                "title": it.get("title"),
                "type": it.get("type"),
                "url": it.get("url"),
                "level": level,
                "meta": {k: v for k, v in it.items() if k not in ("title", "type", "url")}
            })

        if curated:
            blocks.append({
                "phase": phase,
                "description": PHASE_DESCRIPTIONS.get(phase, ""),
                "resources": curated
            })

    blocks.sort(key=lambda b: phase_index.get(b["phase"], 999))
    return blocks

# ------------------------------------------------------------------------------
# Pathways for all modules in one request
# ------------------------------------------------------------------------------
def generate_all_module_pathways(stakeholder_key, preferences, answers_by_module, per_module_q_count):
    out = {}
    for module_key, answers in answers_by_module.items():
        score = sum(int(x) for x in (answers or []))
        level = compute_level_auto(score, per_module_q_count)

        pathway_blocks = collect_resources_for_module(
            module_key=module_key,
            level=level,
            stakeholder_key=stakeholder_key,
            preferences=preferences
        )

        out[module_key] = {
            "module": module_key,
            "score": score,
            "questions_count": len(answers or []),
            "max_score": len(answers or []) * 5,
            "level": level,
            "timeline": get_timeline_recommendation(level, stakeholder_key),
            "estimated_hours": get_estimated_hours(level),
            "phases": [b["phase"] for b in pathway_blocks],
            "learning_pathway": pathway_blocks,
            "total_resources": sum(len(b["resources"]) for b in pathway_blocks),
        }
    return out

# ------------------------------------------------------------------------------
# Shared row helpers (JSON parse + filters)
# ------------------------------------------------------------------------------
def _parse_row_to_dict(row):
    d = dict(row)
    for k in JSON_COLS:
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except Exception:
                pass
    return d

def _build_where_clause(args):
    where = []
    params = []

    if args.get("session_id"):
        where.append("session_id = ?")
        params.append(args["session_id"])

    stake = args.get("stakeholder")
    if stake:
        if stake.endswith("*"):
            where.append("stakeholder LIKE ?")
            params.append(stake[:-1] + "%")
        else:
            where.append("stakeholder = ?")
            params.append(stake)

    if args.get("since"):
        where.append("created_at >= ?")
        params.append(args["since"])
    if args.get("until"):
        where.append("created_at <= ?")
        params.append(args["until"])

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    return clause, params

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.before_request
def guard_startup():
    if _startup_error:
        return jsonify({"error": _startup_error}), 500

@app.route("/")
def home():
    return jsonify({
        "name": "Adaptive Learning Pathway API",
        "version": "3.0",
        "endpoints": [
            "GET  /health",
            "GET  /modules",
            "GET  /phases",
            "GET  /questions",
            "GET  /feedback_questions",
            "POST /generate_pathways",
            "POST /complete_session",
            "GET  /export_session/<session_id>",
            "GET  /get_analytics",
            "GET  /admin/api/sessions",
            "GET  /admin/api/sessions/<session_id>",
            "DELETE /admin/api/sessions/<session_id>",
            "GET  /admin/api/analytics"
        ]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"
    })

@app.route("/modules", methods=["GET"])
def modules():
    return jsonify({"modules": list(get_modules_blob().keys())})

@app.route("/phases", methods=["GET"])
def phases():
    return jsonify({"phases": get_phases()})

@app.route("/questions", methods=["GET"])
def questions():
    stakeholder = request.args.get("stakeholder")
    base = {"version": QUESTIONS.get("version"), "preferences": QUESTIONS.get("preferences", [])}

    if not stakeholder:
        return jsonify({**base, **{k: v for k, v in QUESTIONS.items() if k != "version"}})

    modules = QUESTIONS.get("modules", {})
    if not modules:
        return jsonify({"error": "questions.json is missing 'modules'"}), 500

    sliced = {}
    for module_key, stakeholders_map in modules.items():
        if stakeholder in stakeholders_map:
            sliced[module_key] = stakeholders_map[stakeholder]
    if not sliced:
        return jsonify({"error": f"Unknown stakeholder '{stakeholder}'"}), 400

    return jsonify({**base, "modules": sliced})

@app.route("/generate_pathways", methods=["POST"])
def generate_pathways():
    data = request.get_json(force=True)
    missing = [k for k in ["stakeholder", "competency_answers_by_module"] if k not in data]
    if missing:
        return jsonify({"error": f"Missing required field(s): {', '.join(missing)}"}), 400

    stakeholder = data["stakeholder"]
    preferences = data.get("preferences", [])
    answers_by_module = data["competency_answers_by_module"]

    try:
        per_module_q_count = QUESTIONS["meta"]["scoring"]["per_module_question_count"]
    except Exception:
        first_mod = next(iter(answers_by_module.values()), [])
        per_module_q_count = len(first_mod) if first_mod else 4

    results = generate_all_module_pathways(
        stakeholder_key=stakeholder,
        preferences=preferences,
        answers_by_module=answers_by_module,
        per_module_q_count=per_module_q_count
    )

    session_id = data.get("session_id", f"session_{int(datetime.now().timestamp()*1000)}_{uuid.uuid4().hex[:8]}")

    return jsonify({
        "session_id": session_id,
        "stakeholder": stakeholder,
        "preferences": preferences,
        "results": results
    })

@app.route("/feedback_questions", methods=["GET"])
def feedback_questions():
    if FEEDBACK is None:
        return jsonify({"error": "feedback_questions.json not loaded"}), 500

    section_id = request.args.get("section_id")
    if not section_id:
        return jsonify(FEEDBACK)

    sections = FEEDBACK.get("sections", [])
    match = next((s for s in sections if s.get("id") == section_id), None)
    if not match:
        return jsonify({"error": f"No section found for id '{section_id}'"}), 404
    return jsonify(match)

@app.route("/complete_session", methods=["POST"])
def complete_session():
    """
    Body:
    {
      "stakeholder": "...",
      "preferences": [...],
      "competency_answers_by_module": {"statistics":[...], "machine_learning":[...]},
      "session_id": "optional",
      "feedback_rating": 1..5,
      "feedback_comments": "text",
      "feedback": {...}   # full structured feedback JSON
    }
    """
    data = request.get_json(force=True)
    missing = [k for k in ["stakeholder", "competency_answers_by_module"] if k not in data]
    if missing:
        return jsonify({"error": f"Missing required field(s): {', '.join(missing)}"}), 400

    stakeholder = data["stakeholder"]
    preferences = data.get("preferences", [])
    answers_by_module = data["competency_answers_by_module"]

    try:
        per_module_q_count = QUESTIONS["meta"]["scoring"]["per_module_question_count"]
    except Exception:
        first_mod = next(iter(answers_by_module.values()), [])
        per_module_q_count = len(first_mod) if first_mod else 4

    results = generate_all_module_pathways(
        stakeholder_key=stakeholder,
        preferences=preferences,
        answers_by_module=answers_by_module,
        per_module_q_count=per_module_q_count
    )

    session_id = data.get("session_id", f"session_{int(datetime.now().timestamp()*1000)}_{uuid.uuid4().hex[:8]}")
    feedback_rating = data.get("feedback_rating")
    feedback_comments = (data.get("feedback_comments") or "").strip()
    feedback_obj = data.get("feedback")
    feedback_json = json.dumps(feedback_obj) if feedback_obj is not None else None

    row = {
        "session_id": session_id,
        "stakeholder": stakeholder,
        "preferences": json.dumps(preferences),
        "answers_by_module": json.dumps(answers_by_module),
        "results_json": json.dumps(results),
        "feedback_rating": feedback_rating,
        "feedback_comments": feedback_comments,
        "feedback_json": feedback_json,
        "completed_at": datetime.now().isoformat(),
    }

    db = get_db()
    try:
        db.execute(
            """
            INSERT INTO sessions (
                session_id, stakeholder, preferences, answers_by_module,
                results_json, feedback_rating, feedback_comments, feedback_json, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["session_id"], row["stakeholder"], row["preferences"],
                row["answers_by_module"], row["results_json"],
                row["feedback_rating"], row["feedback_comments"], row["feedback_json"],
                row["completed_at"]
            )
        )
        db.commit()
    except sqlite3.IntegrityError:
        db.execute(
            """
            UPDATE sessions SET
                stakeholder = ?,
                preferences = ?,
                answers_by_module = ?,
                results_json = ?,
                feedback_rating = ?,
                feedback_comments = ?,
                feedback_json = ?,
                completed_at = ?
            WHERE session_id = ?
            """,
            (
                row["stakeholder"], row["preferences"], row["answers_by_module"],
                row["results_json"], row["feedback_rating"], row["feedback_comments"],
                row["feedback_json"], row["completed_at"], row["session_id"]
            )
        )
        db.commit()

    return jsonify({
        "success": True,
        "message": "Session saved",
        "session_id": session_id,
        "results": results,
        "feedback": feedback_obj
    })

@app.route("/export_session/<session_id>", methods=["GET"])
def export_session(session_id):
    db = get_db()
    row = db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row:
        return jsonify({"error": "Session not found"}), 404

    payload = dict(row)
    for k in JSON_COLS:
        if payload.get(k):
            try:
                payload[k] = json.loads(payload[k])
            except Exception:
                pass
    return jsonify(payload)

@app.route("/get_analytics", methods=["GET"])
def get_analytics():
    db = get_db()
    total = db.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]

    stake = db.execute(
        "SELECT stakeholder, COUNT(*) AS count FROM sessions GROUP BY stakeholder ORDER BY count DESC"
    ).fetchall()

    avg = db.execute(
        "SELECT AVG(feedback_rating) AS avg_rating FROM sessions WHERE feedback_rating IS NOT NULL"
    ).fetchone()["avg_rating"]

    rec = db.execute(
        "SELECT session_id, stakeholder, created_at, completed_at FROM sessions ORDER BY created_at DESC LIMIT 10"
    ).fetchall()

    return jsonify({
        "total_sessions": total,
        "stakeholder_distribution": [dict(r) for r in stake],
        "average_feedback_rating": round(avg, 2) if avg is not None else None,
        "recent_sessions": [dict(r) for r in rec]
    })

# ===============================
# Admin API (JSON-only endpoints)
# ===============================
@app.route("/admin/api/sessions", methods=["GET"])
def admin_api_sessions():
    db = get_db()

    limit = min(int(request.args.get("limit", 50)), 200)
    offset = max(int(request.args.get("offset", 0)), 0)
    order = request.args.get("order", "created_at")
    order = order if order in ("created_at", "completed_at") else "created_at"
    direction = request.args.get("dir", "desc").lower()
    direction = "ASC" if direction == "asc" else "DESC"
    raw = request.args.get("raw", "false").lower() == "true"

    where_clause, params = _build_where_clause(request.args)

    total = db.execute(f"SELECT COUNT(*) AS c FROM sessions{where_clause}", params).fetchone()["c"]

    rows = db.execute(
        f"""
        SELECT * FROM sessions
        {where_clause}
        ORDER BY {order} {direction}
        LIMIT ? OFFSET ?
        """,
        (*params, limit, offset)
    ).fetchall()

    results = [dict(r) for r in rows] if raw else [_parse_row_to_dict(r) for r in rows]

    return jsonify({
        "total": total,
        "limit": limit,
        "offset": offset,
        "order": order,
        "dir": "asc" if direction == "ASC" else "desc",
        "results": results
    })

@app.route("/admin/api/sessions/<session_id>", methods=["GET"])
def admin_api_session_detail(session_id):
    db = get_db()
    row = db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(_parse_row_to_dict(row))

@app.route("/admin/api/sessions/<session_id>", methods=["DELETE"])
def admin_api_session_delete(session_id):
    db = get_db()
    cur = db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    db.commit()
    if cur.rowcount == 0:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"success": True, "deleted_session_id": session_id})

@app.route("/admin/api/analytics", methods=["GET"])
def admin_api_analytics():
    db = get_db()
    rows = db.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
    parsed = [_parse_row_to_dict(r) for r in rows]

    from collections import Counter, defaultdict
    total = len(parsed)
    by_stakeholder = Counter([r.get("stakeholder") for r in parsed if r.get("stakeholder")])

    rating_vals = [r["feedback_rating"] for r in parsed if isinstance(r.get("feedback_rating"), (int, float))]
    rating_avg = round(sum(rating_vals)/len(rating_vals), 2) if rating_vals else None
    rating_hist = Counter(rating_vals)

    pref_counter = Counter()
    for r in parsed:
        for p in (r.get("preferences") or []):
            pref_counter[p] += 1

    level_dist = defaultdict(Counter)
    score_stats = defaultdict(list)
    for r in parsed:
        results = r.get("results_json") or {}
        if isinstance(results, dict):
            for m, info in results.items():
                if isinstance(info, dict):
                    lvl = info.get("level")
                    if lvl:
                        level_dist[m][lvl] += 1
                    sc = info.get("score")
                    if isinstance(sc, (int, float)):
                        score_stats[m].append(sc)
    avg_scores = {m: round(sum(v)/len(v), 2) for m, v in score_stats.items() if v}

    by_day = Counter()
    for r in parsed:
        created = (r.get("created_at") or "")[:10]
        if created:
            by_day[created] += 1

    return jsonify({
        "total_sessions": total,
        "stakeholder_distribution": dict(by_stakeholder),
        "feedback": {
            "average_rating": rating_avg,
            "ratings_histogram": dict(rating_hist),
            "top_preferences": pref_counter.most_common(10),
        },
        "modules": {
            "level_distribution": {m: dict(c) for m, c in level_dist.items()},
            "average_scores": avg_scores,
        },
        "completions_by_day": dict(sorted(by_day.items()))
    })

# ------------------------------------------------------------------------------
# Error handlers
# ------------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": f"Internal server error: {e}"}), 500

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if _startup_error:
        print("Startup error:", _startup_error)
    else:
        with app.app_context():
            init_db()
            print("DB ready at", DB_PATH)
            print("Phases:", get_phases())
            print("Modules:", list(get_modules_blob().keys()))
        app.run(host="0.0.0.0", port=5000, debug=False)

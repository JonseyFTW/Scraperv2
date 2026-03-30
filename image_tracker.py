#!/usr/bin/env python3
"""
Card Image Tracker — Web portal to track missing card images per set.
Run: python image_tracker.py [--port 5000] [--host 0.0.0.0]
"""
import argparse
import math
import os

import chromadb
from flask import Flask, render_template_string, request, jsonify, send_file, abort
import psycopg2
import psycopg2.extras
import requests as http_requests

import config
from config import DATABASE_URL

app = Flask(__name__)

# ── DB helpers ───────────────────────────────────────────────────────────────

def get_db():
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    conn.autocommit = True
    return conn


def query(sql, params=None, one=False):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql, params or ())
    rows = cur.fetchall()
    cur.close()
    conn.close()
    if one:
        return dict(rows[0]) if rows else None
    return [dict(r) for r in rows]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    # Overall stats
    stats = query("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'downloaded' THEN 1 ELSE 0 END) AS downloaded,
            SUM(CASE WHEN status = 'no_image' THEN 1 ELSE 0 END) AS no_image,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) AS processing,
            SUM(CASE WHEN status = 'image_found' THEN 1 ELSE 0 END) AS image_found,
            SUM(CASE WHEN status = 'downloading' THEN 1 ELSE 0 END) AS downloading
        FROM cards
    """, one=True)

    # Per-sport breakdown
    sports = query("""
        SELECT
            s.sport,
            COUNT(c.id) AS total,
            SUM(CASE WHEN c.status = 'downloaded' THEN 1 ELSE 0 END) AS downloaded,
            SUM(CASE WHEN c.status = 'no_image' THEN 1 ELSE 0 END) AS no_image,
            SUM(CASE WHEN c.status = 'error' THEN 1 ELSE 0 END) AS error,
            SUM(CASE WHEN c.status NOT IN ('downloaded', 'no_image', 'error') THEN 1 ELSE 0 END) AS in_progress
        FROM sets s
        JOIN cards c ON c.set_slug = s.slug
        GROUP BY s.sport
        ORDER BY s.sport
    """)

    # Pokemon stats
    pokemon = query("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'downloaded' THEN 1 ELSE 0 END) AS downloaded,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending
        FROM pokemon_cards
    """, one=True)

    return render_template_string(DASHBOARD_HTML, stats=stats, sports=sports, pokemon=pokemon)


@app.route("/sets")
def sets_list():
    sport = request.args.get("sport", "")
    status_filter = request.args.get("status", "")  # all, missing, complete, no_image
    search = request.args.get("q", "").strip()
    sort = request.args.get("sort", "missing_desc")
    page = int(request.args.get("page", 1))
    per_page = 50

    # Build query
    where = []
    params = []

    if sport:
        where.append("s.sport = %s")
        params.append(sport)
    if search:
        where.append("(s.name ILIKE %s OR s.slug ILIKE %s)")
        params.extend([f"%{search}%", f"%{search}%"])

    where_clause = "WHERE " + " AND ".join(where) if where else ""

    # Having clause for status filter
    having = ""
    if status_filter == "missing":
        having = "HAVING SUM(CASE WHEN c.status NOT IN ('downloaded') THEN 1 ELSE 0 END) > 0"
    elif status_filter == "complete":
        having = "HAVING SUM(CASE WHEN c.status NOT IN ('downloaded') THEN 1 ELSE 0 END) = 0"
    elif status_filter == "no_image":
        having = "HAVING SUM(CASE WHEN c.status = 'no_image' THEN 1 ELSE 0 END) > 0"

    # Sort
    sort_map = {
        "missing_desc": "missing DESC",
        "missing_asc": "missing ASC",
        "name_asc": "s.name ASC",
        "name_desc": "s.name DESC",
        "pct_asc": "pct ASC",
        "pct_desc": "pct DESC",
        "total_desc": "total DESC",
    }
    order = sort_map.get(sort, "missing DESC")

    # Count total for pagination
    count_sql = f"""
        SELECT COUNT(*) AS c FROM (
            SELECT s.slug
            FROM sets s
            JOIN cards c ON c.set_slug = s.slug
            {where_clause}
            GROUP BY s.slug, s.name, s.sport
            {having}
        ) sub
    """
    total_count = query(count_sql, params, one=True)["c"]
    total_pages = max(1, math.ceil(total_count / per_page))

    # Main query
    sql = f"""
        SELECT
            s.slug, s.name, s.sport,
            COUNT(c.id) AS total,
            SUM(CASE WHEN c.status = 'downloaded' THEN 1 ELSE 0 END) AS downloaded,
            SUM(CASE WHEN c.status = 'no_image' THEN 1 ELSE 0 END) AS no_image,
            SUM(CASE WHEN c.status = 'error' THEN 1 ELSE 0 END) AS error,
            SUM(CASE WHEN c.status NOT IN ('downloaded') THEN 1 ELSE 0 END) AS missing,
            ROUND(100.0 * SUM(CASE WHEN c.status = 'downloaded' THEN 1 ELSE 0 END) / NULLIF(COUNT(c.id), 0), 1) AS pct
        FROM sets s
        JOIN cards c ON c.set_slug = s.slug
        {where_clause}
        GROUP BY s.slug, s.name, s.sport
        {having}
        ORDER BY {order}
        LIMIT %s OFFSET %s
    """
    params.extend([per_page, (page - 1) * per_page])
    sets = query(sql, params)

    # Get sport list for filter dropdown
    all_sports = query("SELECT DISTINCT sport FROM sets ORDER BY sport")

    return render_template_string(
        SETS_HTML,
        sets=sets, sport=sport, status_filter=status_filter, search=search,
        sort=sort, page=page, total_pages=total_pages, total_count=total_count,
        all_sports=all_sports,
    )


@app.route("/set/<slug>")
def set_detail(slug):
    status_filter = request.args.get("status", "")
    page = int(request.args.get("page", 1))
    per_page = 100

    set_info = query("SELECT * FROM sets WHERE slug = %s", [slug], one=True)
    if not set_info:
        return "Set not found", 404

    # Status summary for this set
    summary = query("""
        SELECT status, COUNT(*) AS cnt
        FROM cards WHERE set_slug = %s
        GROUP BY status ORDER BY status
    """, [slug])

    # Cards query
    where = "WHERE c.set_slug = %s"
    params = [slug]
    if status_filter:
        where += " AND c.status = %s"
        params.append(status_filter)

    total = query(f"SELECT COUNT(*) AS c FROM cards c {where}", params, one=True)["c"]
    total_pages = max(1, math.ceil(total / per_page))

    cards = query(f"""
        SELECT c.* FROM cards c
        {where}
        ORDER BY c.product_name
        LIMIT %s OFFSET %s
    """, params + [per_page, (page - 1) * per_page])

    return render_template_string(
        SET_DETAIL_HTML,
        set_info=set_info, summary=summary, cards=cards,
        status_filter=status_filter, page=page, total_pages=total_pages, total=total,
    )


@app.route("/pokemon")
def pokemon_sets():
    search = request.args.get("q", "").strip()
    status_filter = request.args.get("status", "")
    page = int(request.args.get("page", 1))
    per_page = 50

    where = []
    params = []
    if search:
        where.append("(ps.name ILIKE %s OR ps.id ILIKE %s)")
        params.extend([f"%{search}%", f"%{search}%"])

    having = ""
    if status_filter == "missing":
        having = "HAVING SUM(CASE WHEN pc.status != 'downloaded' THEN 1 ELSE 0 END) > 0"
    elif status_filter == "complete":
        having = "HAVING SUM(CASE WHEN pc.status != 'downloaded' THEN 1 ELSE 0 END) = 0"
    elif status_filter == "error":
        having = "HAVING SUM(CASE WHEN pc.status = 'error' THEN 1 ELSE 0 END) > 0"

    where_clause = "WHERE " + " AND ".join(where) if where else ""

    count_sql = f"""
        SELECT COUNT(*) AS c FROM (
            SELECT ps.id
            FROM pokemon_sets ps
            JOIN pokemon_cards pc ON pc.set_id = ps.id
            {where_clause}
            GROUP BY ps.id
            {having}
        ) sub
    """
    total_count = query(count_sql, params, one=True)["c"]
    total_pages = max(1, math.ceil(total_count / per_page))

    sql = f"""
        SELECT
            ps.id, ps.name, ps.series,
            COUNT(pc.id) AS total,
            SUM(CASE WHEN pc.status = 'downloaded' THEN 1 ELSE 0 END) AS downloaded,
            SUM(CASE WHEN pc.status = 'error' THEN 1 ELSE 0 END) AS error,
            SUM(CASE WHEN pc.status = 'pending' THEN 1 ELSE 0 END) AS pending,
            ROUND(100.0 * SUM(CASE WHEN pc.status = 'downloaded' THEN 1 ELSE 0 END) / NULLIF(COUNT(pc.id), 0), 1) AS pct
        FROM pokemon_sets ps
        JOIN pokemon_cards pc ON pc.set_id = ps.id
        {where_clause}
        GROUP BY ps.id, ps.name, ps.series
        {having}
        ORDER BY pending DESC, ps.name
        LIMIT %s OFFSET %s
    """
    params.extend([per_page, (page - 1) * per_page])
    sets = query(sql, params)

    return render_template_string(
        POKEMON_HTML,
        sets=sets, search=search, status_filter=status_filter,
        page=page, total_pages=total_pages, total_count=total_count,
    )


@app.route("/pokemon/<set_id>")
def pokemon_detail(set_id):
    status_filter = request.args.get("status", "")
    page = int(request.args.get("page", 1))
    per_page = 100

    set_info = query("SELECT * FROM pokemon_sets WHERE id = %s", [set_id], one=True)
    if not set_info:
        return "Set not found", 404

    summary = query("""
        SELECT status, COUNT(*) AS cnt
        FROM pokemon_cards WHERE set_id = %s
        GROUP BY status ORDER BY status
    """, [set_id])

    where = "WHERE pc.set_id = %s"
    params = [set_id]
    if status_filter:
        where += " AND pc.status = %s"
        params.append(status_filter)

    total = query(f"SELECT COUNT(*) AS c FROM pokemon_cards pc {where}", params, one=True)["c"]
    total_pages = max(1, math.ceil(total / per_page))

    cards = query(f"""
        SELECT pc.* FROM pokemon_cards pc
        {where}
        ORDER BY pc.local_id
        LIMIT %s OFFSET %s
    """, params + [per_page, (page - 1) * per_page])

    return render_template_string(
        POKEMON_DETAIL_HTML,
        set_info=set_info, summary=summary, cards=cards,
        status_filter=status_filter, page=page, total_pages=total_pages, total=total,
    )


@app.route("/api/stats")
def api_stats():
    """JSON endpoint for programmatic access."""
    stats = query("""
        SELECT
            s.sport,
            COUNT(c.id) AS total,
            SUM(CASE WHEN c.status = 'downloaded' THEN 1 ELSE 0 END) AS downloaded,
            SUM(CASE WHEN c.status = 'no_image' THEN 1 ELSE 0 END) AS no_image,
            SUM(CASE WHEN c.status = 'error' THEN 1 ELSE 0 END) AS error,
            SUM(CASE WHEN c.status IN ('pending','processing','image_found','downloading') THEN 1 ELSE 0 END) AS in_progress
        FROM sets s
        JOIN cards c ON c.set_slug = s.slug
        GROUP BY s.sport ORDER BY s.sport
    """)
    return jsonify(stats)


@app.route("/embeddings")
def embeddings_page():
    """Show local ChromaDB vs RunPod embedding counts."""
    # --- Local ChromaDB ---
    local_collections = {}
    local_available = False
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        for col in client.list_collections():
            name = col.name if hasattr(col, 'name') else col
            c = client.get_collection(name)
            count = c.count()
            if count > 0:
                local_available = True
            local_collections[name] = count
    except Exception:
        pass  # ChromaDB not available on this machine — that's fine

    # --- DB card counts (for context) ---
    scp_downloaded = query(
        "SELECT COUNT(*) AS c FROM cards WHERE status='downloaded'", one=True
    )["c"]
    try:
        pokemon_downloaded = query(
            "SELECT COUNT(*) AS c FROM pokemon_cards WHERE status='downloaded'", one=True
        )["c"]
    except Exception:
        pokemon_downloaded = 0
    try:
        tcgplayer_downloaded = query(
            "SELECT COUNT(*) AS c FROM tcgplayer_cards WHERE status='downloaded'", one=True
        )["c"]
    except Exception:
        tcgplayer_downloaded = 0

    # --- RunPod health check ---
    runpod_data = None
    runpod_error = None
    runpod_collections = {}
    if config.RUNPOD_API_KEY and config.RUNPOD_ENDPOINT_ID:
        try:
            resp = http_requests.post(
                f"https://api.runpod.ai/v2/{config.RUNPOD_ENDPOINT_ID}/runsync",
                json={"input": {"action": "health"}},
                headers={
                    "Authorization": f"Bearer {config.RUNPOD_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("status") == "COMPLETED":
                runpod_data = result.get("output", {})
                runpod_collections = runpod_data.get("collections", {})
                # Fallback for old handler without collections field
                if not runpod_collections and runpod_data.get("embedding_count"):
                    runpod_collections = {"card_embeddings_dinov2": runpod_data["embedding_count"]}
            else:
                runpod_error = f"Endpoint status: {result.get('status')} (may be cold starting)"
        except Exception as e:
            runpod_error = str(e)
    else:
        runpod_error = "RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID not configured"

    return render_template_string(
        EMBEDDINGS_HTML,
        local=local_collections,
        local_available=local_available,
        runpod=runpod_data,
        runpod_error=runpod_error,
        runpod_collections=runpod_collections,
        scp_downloaded=scp_downloaded,
        pokemon_downloaded=pokemon_downloaded,
        tcgplayer_downloaded=tcgplayer_downloaded,
    )


@app.route("/embeddings/browse")
def embeddings_browse():
    """Browse embeddings with card images from a ChromaDB collection."""
    collection_name = request.args.get("collection", "card_embeddings_dinov2")
    search_query = request.args.get("q", "").strip()
    page = int(request.args.get("page", 1))
    per_page = 48

    try:
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        col = client.get_collection(collection_name)
        total = col.count()
    except Exception as e:
        return render_template_string(
            BROWSE_HTML,
            cards=[], collection=collection_name, search=search_query,
            page=1, total_pages=1, total=0, error=str(e),
            collections=[],
        )

    # List available collections for the dropdown
    all_collections = []
    for c in client.list_collections():
        name = c.name if hasattr(c, 'name') else c
        all_collections.append(name)

    # Search or paginate
    cards = []
    if search_query:
        # Filter by metadata name/title match
        results = col.get(
            include=["metadatas"],
            limit=total,  # get all to filter
        )
        matched = []
        q_lower = search_query.lower()
        for cid, meta in zip(results["ids"], results["metadatas"]):
            searchable = " ".join([
                str(meta.get("product_name", "")),
                str(meta.get("name", "")),
                str(meta.get("full_title", "")),
                str(meta.get("set_slug", "")),
                str(meta.get("set_name", "")),
            ]).lower()
            if q_lower in searchable:
                matched.append({"id": cid, "meta": meta})

        total = len(matched)
        total_pages = max(1, math.ceil(total / per_page))
        page = min(page, total_pages)
        start = (page - 1) * per_page
        cards = matched[start:start + per_page]
    else:
        total_pages = max(1, math.ceil(total / per_page))
        page = min(page, total_pages)
        offset = (page - 1) * per_page

        results = col.get(
            include=["metadatas"],
            limit=per_page,
            offset=offset,
        )
        for cid, meta in zip(results["ids"], results["metadatas"]):
            cards.append({"id": cid, "meta": meta})

    return render_template_string(
        BROWSE_HTML,
        cards=cards, collection=collection_name, search=search_query,
        page=page, total_pages=total_pages, total=total, error=None,
        collections=all_collections,
    )


@app.route("/card-image")
def serve_card_image():
    """Proxy route to serve card images from local filesystem paths stored in ChromaDB."""
    path = request.args.get("path", "")
    if not path:
        abort(404)

    # Security: only serve from known image directories
    allowed_prefixes = [
        config.IMAGE_DIR,
        config.POKEMON_IMAGE_DIR,
        config.TCGPLAYER_IMAGE_DIR,
        config.DATA_DIR,
    ]
    # Also allow LINUX_DATA_PREFIX paths (translate them)
    if config.LINUX_DATA_PREFIX and path.startswith(config.LINUX_DATA_PREFIX):
        path = os.path.join(config.DATA_DIR, path[len(config.LINUX_DATA_PREFIX):].lstrip("/\\"))

    real_path = os.path.realpath(path)
    if not any(real_path.startswith(os.path.realpath(p)) for p in allowed_prefixes if p):
        abort(403)

    if not os.path.isfile(real_path):
        abort(404)

    return send_file(real_path)


# ── HTML Templates ───────────────────────────────────────────────────────────

BASE_CSS = """
<style>
  :root {
    --bg: #0f172a; --surface: #1e293b; --surface2: #334155;
    --text: #f1f5f9; --text2: #94a3b8; --accent: #38bdf8;
    --green: #4ade80; --red: #f87171; --yellow: #fbbf24; --orange: #fb923c;
    --blue: #60a5fa;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.5; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  .container { max-width: 1400px; margin: 0 auto; padding: 1rem 1.5rem; }
  nav { background: var(--surface); border-bottom: 1px solid var(--surface2); padding: 0.75rem 1.5rem;
        display: flex; align-items: center; gap: 2rem; position: sticky; top: 0; z-index: 100; }
  nav .brand { font-size: 1.25rem; font-weight: 700; color: var(--accent); }
  nav a { color: var(--text2); font-size: 0.9rem; }
  nav a:hover, nav a.active { color: var(--text); text-decoration: none; }

  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }
  .stat-card { background: var(--surface); border-radius: 12px; padding: 1.25rem; border: 1px solid var(--surface2); }
  .stat-card .label { font-size: 0.8rem; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; }
  .stat-card .value { font-size: 2rem; font-weight: 700; margin-top: 0.25rem; }
  .stat-card .value.green { color: var(--green); }
  .stat-card .value.red { color: var(--red); }
  .stat-card .value.yellow { color: var(--yellow); }
  .stat-card .value.blue { color: var(--blue); }
  .stat-card .value.orange { color: var(--orange); }

  h1 { font-size: 1.75rem; margin-bottom: 0.5rem; }
  h2 { font-size: 1.25rem; margin: 1.5rem 0 0.75rem; color: var(--text2); }

  .filters { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1rem 0; align-items: center; }
  .filters select, .filters input {
    background: var(--surface); color: var(--text); border: 1px solid var(--surface2);
    border-radius: 8px; padding: 0.5rem 0.75rem; font-size: 0.9rem; }
  .filters input { min-width: 250px; }
  .btn { background: var(--accent); color: var(--bg); border: none; border-radius: 8px;
         padding: 0.5rem 1rem; cursor: pointer; font-size: 0.9rem; font-weight: 600; }
  .btn:hover { opacity: 0.9; }
  .btn-sm { padding: 0.3rem 0.6rem; font-size: 0.8rem; }

  table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
  th { background: var(--surface2); color: var(--text2); font-size: 0.8rem; text-transform: uppercase;
       letter-spacing: 0.05em; padding: 0.75rem 1rem; text-align: left; position: sticky; top: 48px; }
  td { padding: 0.65rem 1rem; border-bottom: 1px solid var(--surface2); font-size: 0.9rem; }
  tr:hover td { background: rgba(56, 189, 248, 0.05); }
  th a { color: var(--text2); }
  th a:hover { color: var(--text); text-decoration: none; }

  .bar-bg { background: var(--surface2); border-radius: 4px; height: 8px; overflow: hidden; min-width: 120px; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
  .bar-fill.green { background: var(--green); }
  .bar-fill.yellow { background: var(--yellow); }
  .bar-fill.red { background: var(--red); }

  .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 99px; font-size: 0.75rem; font-weight: 600; }
  .badge-green { background: rgba(74,222,128,0.15); color: var(--green); }
  .badge-red { background: rgba(248,113,113,0.15); color: var(--red); }
  .badge-yellow { background: rgba(251,191,36,0.15); color: var(--yellow); }
  .badge-blue { background: rgba(96,165,250,0.15); color: var(--blue); }
  .badge-orange { background: rgba(251,146,60,0.15); color: var(--orange); }
  .badge-gray { background: rgba(148,163,184,0.15); color: var(--text2); }

  .pagination { display: flex; gap: 0.5rem; align-items: center; margin: 1.5rem 0; justify-content: center; }
  .pagination a, .pagination span {
    padding: 0.4rem 0.75rem; border-radius: 6px; font-size: 0.85rem; }
  .pagination a { background: var(--surface); border: 1px solid var(--surface2); }
  .pagination a:hover { background: var(--surface2); text-decoration: none; }
  .pagination .current { background: var(--accent); color: var(--bg); font-weight: 600; }

  .sport-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin: 1.5rem 0; }
  .sport-card { background: var(--surface); border-radius: 12px; padding: 1.25rem; border: 1px solid var(--surface2);
                transition: border-color 0.2s; cursor: pointer; }
  .sport-card:hover { border-color: var(--accent); text-decoration: none; }
  .sport-card h3 { text-transform: capitalize; margin-bottom: 0.5rem; }
  .sport-card .nums { display: flex; gap: 1rem; font-size: 0.85rem; color: var(--text2); margin-top: 0.5rem; }

  .empty { text-align: center; padding: 3rem; color: var(--text2); }
  .truncate { max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  @media (max-width: 768px) {
    .filters { flex-direction: column; }
    .filters input { min-width: 100%; }
    table { font-size: 0.8rem; }
    td, th { padding: 0.5rem; }
  }
</style>
"""

NAV = """
<nav>
  <span class="brand">Card Image Tracker</span>
  <a href="/" {% if request.path == '/' %}class="active" style="color:var(--text)"{% endif %}>Dashboard</a>
  <a href="/sets" {% if '/sets' in request.path %}class="active" style="color:var(--text)"{% endif %}>Sports Cards</a>
  <a href="/pokemon" {% if '/pokemon' in request.path %}class="active" style="color:var(--text)"{% endif %}>Pokemon</a>
  <a href="/embeddings" {% if '/embeddings' in request.path %}class="active" style="color:var(--text)"{% endif %}>Embeddings</a>
</nav>
"""

PAGINATION_MACRO = """
{% macro pagination(page, total_pages, base_url) %}
{% if total_pages > 1 %}
<div class="pagination">
  {% if page > 1 %}<a href="{{ base_url }}&page={{ page-1 }}">&laquo; Prev</a>{% endif %}
  {% for p in range(1, total_pages+1) %}
    {% if p == page %}
      <span class="current">{{ p }}</span>
    {% elif p <= 3 or p > total_pages-3 or (p >= page-2 and p <= page+2) %}
      <a href="{{ base_url }}&page={{ p }}">{{ p }}</a>
    {% elif p == 4 or p == total_pages-3 %}
      <span style="color:var(--text2)">...</span>
    {% endif %}
  {% endfor %}
  {% if page < total_pages %}<a href="{{ base_url }}&page={{ page+1 }}">Next &raquo;</a>{% endif %}
</div>
{% endif %}
{% endmacro %}
"""

DASHBOARD_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Card Image Tracker</title>""" + BASE_CSS + """</head><body>""" + NAV + """
<div class="container">
  <h1>Dashboard</h1>
  <p style="color:var(--text2)">Overview of card image collection progress</p>

  <h2>Sports Cards</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="label">Total Cards</div><div class="value">{{ "{:,}".format(stats.total) }}</div></div>
    <div class="stat-card"><div class="label">Downloaded</div><div class="value green">{{ "{:,}".format(stats.downloaded) }}</div></div>
    <div class="stat-card"><div class="label">No Image</div><div class="value orange">{{ "{:,}".format(stats.no_image) }}</div></div>
    <div class="stat-card"><div class="label">Errors</div><div class="value red">{{ "{:,}".format(stats.error) }}</div></div>
    <div class="stat-card"><div class="label">Pending</div><div class="value yellow">{{ "{:,}".format(stats.pending) }}</div></div>
    <div class="stat-card"><div class="label">In Progress</div><div class="value blue">{{ "{:,}".format(stats.processing + stats.image_found + stats.downloading) }}</div></div>
    <div class="stat-card"><div class="label">Completion</div>
      <div class="value {% if stats.total > 0 %}{% if stats.downloaded * 100 // stats.total > 80 %}green{% elif stats.downloaded * 100 // stats.total > 50 %}yellow{% else %}red{% endif %}{% endif %}">
        {{ "%.1f"|format(stats.downloaded * 100 / stats.total) if stats.total > 0 else 0 }}%
      </div>
    </div>
    <div class="stat-card"><div class="label">Still Need</div>
      <div class="value red">{{ "{:,}".format(stats.total - stats.downloaded) }}</div>
    </div>
  </div>

  <h2>By Sport</h2>
  <div class="sport-cards">
    {% for sp in sports %}
    <a href="/sets?sport={{ sp.sport }}" class="sport-card" style="text-decoration:none;color:var(--text)">
      <h3>{{ sp.sport }}</h3>
      <div class="bar-bg"><div class="bar-fill {% if sp.total > 0 %}{% if sp.downloaded * 100 // sp.total > 80 %}green{% elif sp.downloaded * 100 // sp.total > 50 %}yellow{% else %}red{% endif %}{% endif %}" style="width:{{ (sp.downloaded * 100 / sp.total)|round(1) if sp.total > 0 else 0 }}%"></div></div>
      <div class="nums">
        <span style="color:var(--green)">{{ "{:,}".format(sp.downloaded) }} done</span>
        <span style="color:var(--orange)">{{ "{:,}".format(sp.no_image) }} no img</span>
        <span style="color:var(--red)">{{ "{:,}".format(sp.error) }} err</span>
        <span>{{ "{:,}".format(sp.total) }} total</span>
      </div>
    </a>
    {% endfor %}
  </div>

  {% if pokemon and pokemon.total > 0 %}
  <h2>Pokemon TCG</h2>
  <a href="/pokemon" style="text-decoration:none">
  <div class="stats-grid">
    <div class="stat-card"><div class="label">Total</div><div class="value">{{ "{:,}".format(pokemon.total) }}</div></div>
    <div class="stat-card"><div class="label">Downloaded</div><div class="value green">{{ "{:,}".format(pokemon.downloaded) }}</div></div>
    <div class="stat-card"><div class="label">Errors</div><div class="value red">{{ "{:,}".format(pokemon.error) }}</div></div>
    <div class="stat-card"><div class="label">Pending</div><div class="value yellow">{{ "{:,}".format(pokemon.pending) }}</div></div>
  </div>
  </a>
  {% endif %}
</div>
</body></html>"""


SETS_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sports Card Sets — Image Tracker</title>""" + BASE_CSS + """</head><body>""" + NAV + PAGINATION_MACRO + """
<div class="container">
  <h1>Sports Card Sets</h1>
  <p style="color:var(--text2)">{{ "{:,}".format(total_count) }} sets found</p>

  <form class="filters" method="get" action="/sets">
    <select name="sport" onchange="this.form.submit()">
      <option value="">All Sports</option>
      {% for s in all_sports %}<option value="{{ s.sport }}" {% if s.sport == sport %}selected{% endif %}>{{ s.sport|capitalize }}</option>{% endfor %}
    </select>
    <select name="status" onchange="this.form.submit()">
      <option value="" {% if not status_filter %}selected{% endif %}>All Statuses</option>
      <option value="missing" {% if status_filter == 'missing' %}selected{% endif %}>Has Missing</option>
      <option value="complete" {% if status_filter == 'complete' %}selected{% endif %}>Complete</option>
      <option value="no_image" {% if status_filter == 'no_image' %}selected{% endif %}>Has No-Image</option>
    </select>
    <select name="sort" onchange="this.form.submit()">
      <option value="missing_desc" {% if sort == 'missing_desc' %}selected{% endif %}>Most Missing</option>
      <option value="missing_asc" {% if sort == 'missing_asc' %}selected{% endif %}>Least Missing</option>
      <option value="pct_asc" {% if sort == 'pct_asc' %}selected{% endif %}>Lowest %</option>
      <option value="pct_desc" {% if sort == 'pct_desc' %}selected{% endif %}>Highest %</option>
      <option value="total_desc" {% if sort == 'total_desc' %}selected{% endif %}>Most Cards</option>
      <option value="name_asc" {% if sort == 'name_asc' %}selected{% endif %}>Name A-Z</option>
    </select>
    <input type="text" name="q" value="{{ search }}" placeholder="Search sets...">
    <button type="submit" class="btn">Search</button>
    {% if search or sport or status_filter %}<a href="/sets" style="font-size:0.85rem">Clear</a>{% endif %}
  </form>

  {% if sets %}
  <table>
    <thead><tr>
      <th>Set Name</th><th>Sport</th><th>Total</th><th>Downloaded</th><th>No Image</th><th>Error</th><th>Missing</th><th>Progress</th>
    </tr></thead>
    <tbody>
    {% for s in sets %}
    <tr>
      <td><a href="/set/{{ s.slug }}">{{ s.name or s.slug }}</a></td>
      <td><span class="badge badge-blue">{{ s.sport }}</span></td>
      <td>{{ "{:,}".format(s.total) }}</td>
      <td style="color:var(--green)">{{ "{:,}".format(s.downloaded) }}</td>
      <td>{% if s.no_image > 0 %}<span style="color:var(--orange)">{{ "{:,}".format(s.no_image) }}</span>{% else %}-{% endif %}</td>
      <td>{% if s.error > 0 %}<span style="color:var(--red)">{{ "{:,}".format(s.error) }}</span>{% else %}-{% endif %}</td>
      <td>{% if s.missing > 0 %}<span style="color:var(--yellow)">{{ "{:,}".format(s.missing) }}</span>{% else %}<span class="badge badge-green">complete</span>{% endif %}</td>
      <td>
        <div style="display:flex;align-items:center;gap:0.5rem">
          <div class="bar-bg"><div class="bar-fill {% if s.pct and s.pct > 80 %}green{% elif s.pct and s.pct > 50 %}yellow{% else %}red{% endif %}" style="width:{{ s.pct or 0 }}%"></div></div>
          <span style="font-size:0.8rem;color:var(--text2);min-width:3rem">{{ s.pct or 0 }}%</span>
        </div>
      </td>
    </tr>
    {% endfor %}
    </tbody>
  </table>

  {{ pagination(page, total_pages, '/sets?sport=' ~ sport ~ '&status=' ~ status_filter ~ '&sort=' ~ sort ~ '&q=' ~ search) }}
  {% else %}
  <div class="empty">No sets found matching your filters.</div>
  {% endif %}
</div>
</body></html>"""


SET_DETAIL_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ set_info.name or set_info.slug }} — Image Tracker</title>""" + BASE_CSS + """</head><body>""" + NAV + PAGINATION_MACRO + """
<div class="container">
  <p><a href="/sets?sport={{ set_info.sport }}">&larr; Back to {{ set_info.sport|capitalize }} sets</a></p>
  <h1>{{ set_info.name or set_info.slug }}</h1>
  <p style="color:var(--text2)">{{ set_info.sport|capitalize }} &mdash; {{ "{:,}".format(total) }} cards shown</p>

  <div class="stats-grid" style="margin:1rem 0">
    {% for s in summary %}
    <a href="/set/{{ set_info.slug }}?status={{ s.status }}" style="text-decoration:none;color:var(--text)">
    <div class="stat-card" {% if status_filter == s.status %}style="border-color:var(--accent)"{% endif %}>
      <div class="label">{{ s.status }}</div>
      <div class="value {% if s.status == 'downloaded' %}green{% elif s.status == 'no_image' %}orange{% elif s.status == 'error' %}red{% elif s.status == 'pending' %}yellow{% else %}blue{% endif %}">{{ "{:,}".format(s.cnt) }}</div>
    </div></a>
    {% endfor %}
    <a href="/set/{{ set_info.slug }}" style="text-decoration:none;color:var(--text)">
    <div class="stat-card" {% if not status_filter %}style="border-color:var(--accent)"{% endif %}>
      <div class="label">All</div>
      <div class="value">{{ "{:,}".format(summary|sum(attribute='cnt')) }}</div>
    </div></a>
  </div>

  {% if cards %}
  <table>
    <thead><tr>
      <th>Card Name</th><th>Status</th><th>Image URL</th><th>Error</th>
    </tr></thead>
    <tbody>
    {% for c in cards %}
    <tr>
      <td class="truncate" title="{{ c.product_name }}">{{ c.product_name }}</td>
      <td>
        <span class="badge {% if c.status == 'downloaded' %}badge-green{% elif c.status == 'no_image' %}badge-orange{% elif c.status == 'error' %}badge-red{% elif c.status == 'pending' %}badge-yellow{% else %}badge-blue{% endif %}">{{ c.status }}</span>
      </td>
      <td class="truncate" style="max-width:200px">
        {% if c.image_url %}<a href="{{ c.image_url }}" target="_blank" title="{{ c.image_url }}">view</a>{% else %}-{% endif %}
      </td>
      <td class="truncate" style="color:var(--red);max-width:200px" title="{{ c.error_msg or '' }}">{{ c.error_msg or '-' }}</td>
    </tr>
    {% endfor %}
    </tbody>
  </table>

  {{ pagination(page, total_pages, '/set/' ~ set_info.slug ~ '?status=' ~ status_filter) }}
  {% else %}
  <div class="empty">No cards match this filter.</div>
  {% endif %}
</div>
</body></html>"""


POKEMON_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pokemon Sets — Image Tracker</title>""" + BASE_CSS + """</head><body>""" + NAV + PAGINATION_MACRO + """
<div class="container">
  <h1>Pokemon TCG Sets</h1>
  <p style="color:var(--text2)">{{ "{:,}".format(total_count) }} sets found</p>

  <form class="filters" method="get" action="/pokemon">
    <select name="status" onchange="this.form.submit()">
      <option value="" {% if not status_filter %}selected{% endif %}>All</option>
      <option value="missing" {% if status_filter == 'missing' %}selected{% endif %}>Has Missing</option>
      <option value="complete" {% if status_filter == 'complete' %}selected{% endif %}>Complete</option>
      <option value="error" {% if status_filter == 'error' %}selected{% endif %}>Has Errors</option>
    </select>
    <input type="text" name="q" value="{{ search }}" placeholder="Search sets...">
    <button type="submit" class="btn">Search</button>
    {% if search or status_filter %}<a href="/pokemon" style="font-size:0.85rem">Clear</a>{% endif %}
  </form>

  {% if sets %}
  <table>
    <thead><tr>
      <th>Set Name</th><th>Series</th><th>Total</th><th>Downloaded</th><th>Error</th><th>Pending</th><th>Progress</th>
    </tr></thead>
    <tbody>
    {% for s in sets %}
    <tr>
      <td><a href="/pokemon/{{ s.id }}">{{ s.name }}</a></td>
      <td style="color:var(--text2)">{{ s.series or '-' }}</td>
      <td>{{ "{:,}".format(s.total) }}</td>
      <td style="color:var(--green)">{{ "{:,}".format(s.downloaded) }}</td>
      <td>{% if s.error > 0 %}<span style="color:var(--red)">{{ "{:,}".format(s.error) }}</span>{% else %}-{% endif %}</td>
      <td>{% if s.pending > 0 %}<span style="color:var(--yellow)">{{ "{:,}".format(s.pending) }}</span>{% else %}-{% endif %}</td>
      <td>
        <div style="display:flex;align-items:center;gap:0.5rem">
          <div class="bar-bg"><div class="bar-fill {% if s.pct and s.pct > 80 %}green{% elif s.pct and s.pct > 50 %}yellow{% else %}red{% endif %}" style="width:{{ s.pct or 0 }}%"></div></div>
          <span style="font-size:0.8rem;color:var(--text2);min-width:3rem">{{ s.pct or 0 }}%</span>
        </div>
      </td>
    </tr>
    {% endfor %}
    </tbody>
  </table>

  {{ pagination(page, total_pages, '/pokemon?status=' ~ status_filter ~ '&q=' ~ search) }}
  {% else %}
  <div class="empty">No sets found matching your filters.</div>
  {% endif %}
</div>
</body></html>"""


POKEMON_DETAIL_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ set_info.name }} — Image Tracker</title>""" + BASE_CSS + """</head><body>""" + NAV + PAGINATION_MACRO + """
<div class="container">
  <p><a href="/pokemon">&larr; Back to Pokemon sets</a></p>
  <h1>{{ set_info.name }}</h1>
  <p style="color:var(--text2)">{{ set_info.series or '' }} &mdash; {{ "{:,}".format(total) }} cards shown</p>

  <div class="stats-grid" style="margin:1rem 0">
    {% for s in summary %}
    <a href="/pokemon/{{ set_info.id }}?status={{ s.status }}" style="text-decoration:none;color:var(--text)">
    <div class="stat-card" {% if status_filter == s.status %}style="border-color:var(--accent)"{% endif %}>
      <div class="label">{{ s.status }}</div>
      <div class="value {% if s.status == 'downloaded' %}green{% elif s.status == 'error' %}red{% else %}yellow{% endif %}">{{ "{:,}".format(s.cnt) }}</div>
    </div></a>
    {% endfor %}
    <a href="/pokemon/{{ set_info.id }}" style="text-decoration:none;color:var(--text)">
    <div class="stat-card" {% if not status_filter %}style="border-color:var(--accent)"{% endif %}>
      <div class="label">All</div>
      <div class="value">{{ "{:,}".format(summary|sum(attribute='cnt')) }}</div>
    </div></a>
  </div>

  {% if cards %}
  <table>
    <thead><tr>
      <th>Card</th><th>Name</th><th>Category</th><th>Status</th><th>Error</th>
    </tr></thead>
    <tbody>
    {% for c in cards %}
    <tr>
      <td>{{ c.local_id or c.id }}</td>
      <td>{{ c.name }}</td>
      <td style="color:var(--text2)">{{ c.category or '-' }}</td>
      <td>
        <span class="badge {% if c.status == 'downloaded' %}badge-green{% elif c.status == 'error' %}badge-red{% else %}badge-yellow{% endif %}">{{ c.status }}</span>
      </td>
      <td class="truncate" style="color:var(--red);max-width:200px" title="{{ c.error_msg or '' }}">{{ c.error_msg or '-' }}</td>
    </tr>
    {% endfor %}
    </tbody>
  </table>

  {{ pagination(page, total_pages, '/pokemon/' ~ set_info.id ~ '?status=' ~ status_filter) }}
  {% else %}
  <div class="empty">No cards match this filter.</div>
  {% endif %}
</div>
</body></html>"""


EMBEDDINGS_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Embeddings — Image Tracker</title>""" + BASE_CSS + """</head><body>""" + NAV + """
<div class="container">
  <h1>Embeddings</h1>
  <p style="color:var(--text2)">ChromaDB embedding status — local and RunPod</p>

  <h2>Downloaded Cards (available for embedding)</h2>
  <div class="stats-grid">
    <div class="stat-card">
      <div class="label">Sports Cards</div>
      <div class="value green">{{ "{:,}".format(scp_downloaded) }}</div>
    </div>
    <div class="stat-card">
      <div class="label">Pokemon (TCGdex)</div>
      <div class="value green">{{ "{:,}".format(pokemon_downloaded) }}</div>
    </div>
    <div class="stat-card">
      <div class="label">Pokemon (TCGPlayer)</div>
      <div class="value green">{{ "{:,}".format(tcgplayer_downloaded) }}</div>
    </div>
  </div>

  {% if local_available %}
  <h2>Local ChromaDB Collections</h2>
  <div class="stats-grid">
    {% for name, count in local.items() %}
    <a href="/embeddings/browse?collection={{ name }}" style="text-decoration:none;color:var(--text)">
    <div class="stat-card" style="cursor:pointer">
      <div class="label">{{ name }}</div>
      <div class="value blue">{{ "{:,}".format(count) }}</div>
      <div style="font-size:0.75rem;color:var(--accent);margin-top:0.25rem">Browse &rarr;</div>
    </div>
    </a>
    {% endfor %}
  </div>
  {% endif %}

  <h2>RunPod Endpoint</h2>
  {% if runpod %}
  <div class="stats-grid">
    <div class="stat-card">
      <div class="label">Status</div>
      <div class="value green" style="font-size:1.5rem">{{ runpod.status or 'ok' }}</div>
    </div>
    <div class="stat-card">
      <div class="label">Device</div>
      <div class="value" style="font-size:1.25rem;color:var(--text2)">{{ runpod.device or 'unknown' }}</div>
    </div>
    <div class="stat-card">
      <div class="label">Model</div>
      <div class="value" style="font-size:0.9rem;color:var(--text2)">{{ runpod.model or 'unknown' }}</div>
    </div>
  </div>

  <h2>RunPod Collections</h2>
  <div class="stats-grid">
    {% for name, count in runpod_collections.items() %}
    <div class="stat-card">
      <div class="label">{{ name }}</div>
      <div class="value blue">{{ "{:,}".format(count) }}</div>
    </div>
    {% endfor %}
  </div>

  <h2>Sync Status</h2>
  <table>
    <thead><tr><th>Collection</th><th>Local</th><th>RunPod</th><th>Difference</th><th>Status</th></tr></thead>
    <tbody>
      {% set all_names = [] %}
      {% for name in runpod_collections %}{% if all_names.append(name) %}{% endif %}{% endfor %}
      {% for name in local %}{% if name not in all_names and all_names.append(name) %}{% endif %}{% endfor %}
      {% for name in all_names %}
      {% set local_count = local.get(name, 0) %}
      {% set remote_count = runpod_collections.get(name, 0) %}
      {% set has_local = name in local and local[name] > 0 %}
      {% set has_remote = name in runpod_collections %}
      <tr>
        <td>{{ name }}</td>
        <td>{% if has_local %}{{ "{:,}".format(local_count) }}{% else %}<span style="color:var(--text2)">—</span>{% endif %}</td>
        <td>{% if has_remote %}{{ "{:,}".format(remote_count) }}{% else %}<span style="color:var(--text2)">—</span>{% endif %}</td>
        <td>
          {% if has_local and has_remote %}
            {% set diff = local_count - remote_count %}
            {% if diff == 0 %}—
            {% elif diff > 0 %}<span style="color:var(--yellow)">+{{ "{:,}".format(diff) }} local</span>
            {% else %}<span style="color:var(--yellow)">+{{ "{:,}".format(diff|abs) }} remote</span>{% endif %}
          {% elif not has_local and has_remote %}
            <span style="color:var(--text2)">local N/A</span>
          {% else %}
            <span style="color:var(--text2)">—</span>
          {% endif %}
        </td>
        <td>
          {% if has_local and has_remote %}
            {% if local_count == remote_count %}<span class="badge badge-green">In Sync</span>
            {% else %}<span class="badge badge-red">Out of Sync</span>{% endif %}
          {% elif not has_local and has_remote %}
            <span style="color:var(--text2);font-size:0.8rem">Local ChromaDB not on this machine</span>
          {% else %}
            <span style="color:var(--text2);font-size:0.8rem">Not on RunPod</span>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  {% elif runpod_error %}
  <div class="stat-card" style="border-color:var(--yellow)">
    <div class="label">RunPod Unavailable</div>
    <div class="value yellow" style="font-size:1rem;word-break:break-all">{{ runpod_error }}</div>
  </div>
  <p style="margin-top:1rem;color:var(--text2)">The endpoint may be sleeping. Try refreshing in 30-60 seconds.</p>
  {% endif %}

  <h2 style="margin-top:2rem">Coverage</h2>
  {% set scp_emb = runpod_collections.get('card_embeddings_dinov2', 0) if runpod_collections else local.get('card_embeddings_dinov2', 0) %}
  {% set poke_emb = runpod_collections.get('pokemon_embeddings_dinov2', 0) if runpod_collections else local.get('pokemon_embeddings_dinov2', 0) %}
  {% set scp_pct = (100 * scp_emb / scp_downloaded) if scp_downloaded > 0 else 0 %}
  {% set poke_total = pokemon_downloaded + tcgplayer_downloaded %}
  {% set poke_pct = (100 * poke_emb / poke_total) if poke_total > 0 else 0 %}
  <table>
    <thead><tr><th>Source</th><th>Downloaded</th><th>Embedded</th><th>Coverage</th></tr></thead>
    <tbody>
      <tr>
        <td>Sports Cards</td>
        <td>{{ "{:,}".format(scp_downloaded) }}</td>
        <td>{{ "{:,}".format(scp_emb) }}</td>
        <td>
          <div style="display:flex;align-items:center;gap:0.5rem">
            <div class="bar-bg" style="min-width:200px"><div class="bar-fill {% if scp_pct > 80 %}green{% elif scp_pct > 50 %}yellow{% else %}red{% endif %}" style="width:{{ scp_pct }}%"></div></div>
            <span style="font-size:0.8rem;color:var(--text2)">{{ "%.1f"|format(scp_pct) }}%</span>
          </div>
        </td>
      </tr>
      <tr>
        <td>Pokemon (all sources)</td>
        <td>{{ "{:,}".format(poke_total) }}</td>
        <td>{{ "{:,}".format(poke_emb) }}</td>
        <td>
          <div style="display:flex;align-items:center;gap:0.5rem">
            <div class="bar-bg" style="min-width:200px"><div class="bar-fill {% if poke_pct > 80 %}green{% elif poke_pct > 50 %}yellow{% else %}red{% endif %}" style="width:{{ poke_pct }}%"></div></div>
            <span style="font-size:0.8rem;color:var(--text2)">{{ "%.1f"|format(poke_pct) }}%</span>
          </div>
        </td>
      </tr>
    </tbody>
  </table>
</div>
</body></html>"""



BROWSE_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Browse Embeddings — Image Tracker</title>""" + BASE_CSS + """
<style>
  .card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0; }
  .card-item { background: var(--surface); border: 1px solid var(--surface2); border-radius: 12px; overflow: hidden; transition: border-color 0.2s; }
  .card-item:hover { border-color: var(--accent); }
  .card-img { width: 100%; aspect-ratio: 3/4; object-fit: contain; background: #0a0f1a; display: block; }
  .card-img-placeholder { width: 100%; aspect-ratio: 3/4; background: var(--surface2); display: flex; align-items: center;
    justify-content: center; color: var(--text2); font-size: 0.8rem; }
  .card-info { padding: 0.75rem; }
  .card-info .name { font-size: 0.85rem; font-weight: 600; line-height: 1.3; margin-bottom: 0.25rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .card-info .meta { font-size: 0.75rem; color: var(--text2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .card-info .id { font-size: 0.7rem; color: var(--surface2); font-family: monospace; margin-top: 0.25rem; }
</style>
</head><body>""" + NAV + PAGINATION_MACRO + """
<div class="container">
  <h1>Browse Embeddings</h1>
  <p style="color:var(--text2)">Viewing images stored in ChromaDB — {{ "{:,}".format(total) }} embeddings</p>

  {% if error %}
  <div class="stat-card" style="border-color:var(--red);margin:1rem 0">
    <div class="label">Error</div>
    <div class="value red" style="font-size:1rem">{{ error }}</div>
  </div>
  {% else %}

  <div class="filters">
    <form method="get" action="/embeddings/browse" style="display:flex;gap:0.75rem;align-items:center;flex-wrap:wrap">
      <select name="collection">
        {% for c in collections %}
        <option value="{{ c }}" {% if c == collection %}selected{% endif %}>{{ c }}</option>
        {% endfor %}
      </select>
      <input type="text" name="q" value="{{ search }}" placeholder="Search by card name or set...">
      <button type="submit" class="btn">Search</button>
      {% if search %}<a href="/embeddings/browse?collection={{ collection }}" class="btn" style="background:var(--surface2);color:var(--text)">Clear</a>{% endif %}
    </form>
  </div>

  {% if cards %}
  <div class="card-grid">
    {% for card in cards %}
    <div class="card-item">
      {% set img_path = card.meta.get('image_path', '') %}
      {% if img_path %}
      <img class="card-img" src="/card-image?path={{ img_path | urlencode }}" alt="{{ card.meta.get('product_name') or card.meta.get('name', '') }}" loading="lazy"
           onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
      <div class="card-img-placeholder" style="display:none">Image not found</div>
      {% else %}
      <div class="card-img-placeholder">No image path</div>
      {% endif %}
      <div class="card-info">
        <div class="name" title="{{ card.meta.get('product_name') or card.meta.get('full_title') or card.meta.get('name', 'Unknown') }}">
          {{ card.meta.get('product_name') or card.meta.get('full_title') or card.meta.get('name', 'Unknown') }}
        </div>
        <div class="meta">{{ card.meta.get('set_slug') or card.meta.get('set_name', '') }}</div>
        {% if card.meta.get('loose_price') and card.meta.loose_price > 0 %}
        <div class="meta" style="color:var(--green)">${{ "%.2f"|format(card.meta.loose_price) }}</div>
        {% endif %}
        {% if card.meta.get('source') %}
        <div class="meta">Source: {{ card.meta.source }}</div>
        {% endif %}
        <div class="id">{{ card.id }}</div>
      </div>
    </div>
    {% endfor %}
  </div>

  {{ pagination(page, total_pages, '/embeddings/browse?collection=' ~ collection ~ '&q=' ~ search) }}
  {% else %}
  <div class="empty">No embeddings found{% if search %} matching "{{ search }}"{% endif %}.</div>
  {% endif %}
  {% endif %}

  <p style="margin-top:1.5rem"><a href="/embeddings">&larr; Back to Embeddings overview</a></p>
</div>
</body></html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Card Image Tracker Web Portal")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print(f"\n  Card Image Tracker running at http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)

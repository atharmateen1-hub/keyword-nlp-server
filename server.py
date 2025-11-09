import json
import os
import subprocess
from flask import Flask, request, jsonify

app = Flask(__name__)


# -------------------------------------------------
# Health check (GET /)
# -------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    """
    Simple health check so you can open the Render URL
    in the browser and see that the server is alive.
    """
    return jsonify({
        "status": "ok",
        "message": "Keyword NLP server is running successfully on Render"
    })


# -------------------------------------------------
# Info for GET /analyze (for debugging in browser)
# -------------------------------------------------
@app.route("/analyze", methods=["GET"])
def analyze_get():
    """
    If you open /analyze in a browser, you get a simple
    message instead of an HTML error.
    The WP plugin uses POST /analyze.
    """
    return jsonify({
        "status": "ok",
        "message": "Use POST /analyze with JSON payload from WordPress."
    })


# -------------------------------------------------
# Helper: run nlp_keywords.py safely
# -------------------------------------------------
def run_nlp_script(payload: dict):
    """
    Call nlp_keywords.py as a subprocess.

    - Sends `payload` as JSON on stdin.
    - Waits up to 45 seconds.
    - Returns (result_dict, error_message)
      where one of them is None.
    """
    try:
        process = subprocess.Popen(
            ["python", "nlp_keywords.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        out, err = process.communicate(json.dumps(payload), timeout=45)

        if process.returncode != 0:
            # Script exited with error code
            return None, f"nlp_keywords.py exited with code {process.returncode}: {err}"

        if not out:
            return None, "nlp_keywords.py returned no output."

        try:
            parsed = json.loads(out)
        except Exception as e:
            return None, f"Failed to parse JSON from nlp_keywords.py: {e}"

        if not isinstance(parsed, dict):
            return None, "nlp_keywords.py output is not a JSON object."

        return parsed, None

    except subprocess.TimeoutExpired:
        process.kill()
        return None, "nlp_keywords.py timed out after 45 seconds."
    except Exception as e:
        return None, f"Error running nlp_keywords.py: {e}"


# -------------------------------------------------
# Main endpoint: POST /analyze
# -------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze_post():
    """
    Main endpoint used by WordPress.

    Expected JSON payload (from the WP plugin):
      {
        "focus_keyword": "...",
        "competitor_urls": [...],
        "your_keywords": [...],
        "keyword_rows": [...],
        "content_selector": "..."
      }

    We try to call nlp_keywords.py with this payload.
    If anything fails, we return a simple fallback
    structure so WordPress always gets valid JSON.
    """
    # Safely parse JSON body
    data = request.get_json(force=True, silent=True) or {}

    focus_keyword = data.get("focus_keyword", "") or ""
    competitor_urls = data.get("competitor_urls", []) or []
    your_keywords = data.get("your_keywords", []) or []
    keyword_rows = data.get("keyword_rows", []) or []
    content_selector = data.get("content_selector", "") or ""

    # -----------------------------
    # 1) Try to get real NLP result
    # -----------------------------
    nlp_result, nlp_error = run_nlp_script(data)

    # -----------------------------
    # 2) If NLP fails, build fallback
    # -----------------------------
    if not nlp_result:
        keywords = []

        # Very simple example: one primary keyword per URL using focus keyword
        for idx, url in enumerate(competitor_urls, start=1):
            keywords.append({
                "phrase": focus_keyword or f"keyword-{idx}",
                "main_type": "primary",
                "cluster_id": 1,
                "total_count": 1,
                "doc_count": 1,
                "semantic_score": 1.0,
                "volume": None,
                "kd": None,
                "traffic": None,
                "opportunity_score": 1.0,
                "is_gap": False,
                "is_semantic": True,
                "is_sli": False,
                "is_nlp": True,
            })

        nlp_result = {
            "focus_keyword": focus_keyword,
            "summary": {
                "total_keywords": len(keywords),
                "semantic_count": len(keywords),
                "sli_count": 0,
                "nlp_count": len(keywords),
                "gap_count": 0,
                "map_count": 1,
            },
            "keywords": keywords,
            "keyword_map": [
                {
                    "cluster_id": 1,
                    "main_keyword": focus_keyword or "Main Topic",
                    "body_keywords": [focus_keyword] if focus_keyword else [],
                }
            ],
            "debug": {
                "nlp_error": nlp_error,
                "received": {
                    "focus_keyword": focus_keyword,
                    "competitor_urls_count": len(competitor_urls),
                    "your_keywords_count": len(your_keywords),
                    "keyword_rows_count": len(keyword_rows),
                    "content_selector": content_selector,
                },
            },
        }
    else:
        # If NLP succeeded, make sure we still attach debug info
        debug_info = nlp_result.get("debug", {})
        debug_info.setdefault("nlp_error", nlp_error)
        debug_info.setdefault("received", {
            "focus_keyword": focus_keyword,
            "competitor_urls_count": len(competitor_urls),
            "your_keywords_count": len(your_keywords),
            "keyword_rows_count": len(keyword_rows),
            "content_selector": content_selector,
        })
        nlp_result["debug"] = debug_info

    # Always return JSON
    return jsonify(nlp_result)


# -------------------------------------------------
# Entry point for local dev / Render start command
# -------------------------------------------------
if __name__ == "__main__":
    # Render typically sets PORT environment variable.
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)

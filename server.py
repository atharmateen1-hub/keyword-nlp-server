from flask import Flask, request, jsonify
import subprocess
import json

app = Flask(__name__)


# -------------------------------------------------
# 1. Health check route (for browser / Render test)
# -------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    """
    Simple check so you can open the Render URL in the browser
    and see that the server is alive.
    """
    return jsonify({
        "status": "ok",
        "message": "Keyword NLP server is running successfully on Render"
    })


# -------------------------------------------------
# 2. NLP analyze route (WordPress will call this)
# -------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Expects JSON from WordPress, passes it to nlp_keywords.py,
    reads the result, and returns it as JSON.
    """
    # Get JSON body from the request (force=True allows no header)
    data = request.get_json(force=True)

    # Call the Python NLP script as a subprocess
    process = subprocess.Popen(
        ["python", "nlp_keywords.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Send the JSON data to nlp_keywords.py via stdin
    out, err = process.communicate(json.dumps(data))

    if process.returncode != 0:
        # Something went wrong in nlp_keywords.py
        return jsonify({
            "error": "nlp_keywords.py failed",
            "details": err
        }), 500

    # Parse the JSON string coming back from nlp_keywords.py
    try:
        result = json.loads(out)
    except json.JSONDecodeError:
        return jsonify({
            "error": "Failed to decode JSON from nlp_keywords.py",
            "raw_output": out
        }), 500

    return jsonify(result)


# -------------------------------------------------
# 3. Local dev entrypoint (not used by Render)
# -------------------------------------------------
if __name__ == "__main__":
    # This is only for running locally: python server.py
    app.run(host="0.0.0.0", port=5000, debug=True)

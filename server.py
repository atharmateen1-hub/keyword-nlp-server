from flask import Flask, request, jsonify
import subprocess, json

app = Flask(__name__)

# Health check endpoint for Render / Browser test
@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "message": "Keyword NLP server is running successfully on Render"
    })

@app.post("/analyze")
def analyze():
    data = request.get_json()
    process = subprocess.Popen(
        ["python", "nlp_keywords.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )
    out, _ = process.communicate(json.dumps(data))
    return jsonify(json.loads(out))



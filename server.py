from flask import Flask, request, jsonify
import subprocess, json

app = Flask(__name__)

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

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

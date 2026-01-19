from flask import Flask, jsonify
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    print("🔥 TEST HEALTH WORKS!")
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
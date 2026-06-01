from flask import Flask, request, jsonify
from worker import predict_task

app = Flask(__name__)

# ✅ health check
@app.route("/", methods=["GET"])
def home():
    return "API is running", 200

# ✅ endpoint used by Node.js
@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.json

        result_id = data.get("result_id")
        image_path = data.get("image_path")

        if not result_id or not image_path:
            return jsonify({"error": "Missing data"}), 400

        # send task to Redis → Celery
        predict_task.delay(result_id, image_path)

        return jsonify({"status": "task sent"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

import os
import sys
from celery import Celery
from pymongo import MongoClient
from bson.objectid import ObjectId
from model import predict_image

# =========================
# 🔥 ENV VARIABLES
# =========================
REDIS_URL = os.environ.get("CELERY_BROKER_URL")
MONGO_URI = os.environ.get("MONGO_URI")

if not REDIS_URL:
    raise ValueError("❌ CELERY_BROKER_URL is not set")

if not MONGO_URI:
    raise ValueError("❌ MONGO_URI is not set")

# =========================
# 🔥 CELERY CONFIG (Upstash TLS)
# =========================
app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# ⚠️ Required for Upstash (TLS Redis)
app.conf.broker_use_ssl = {
    "ssl_cert_reqs": None
}
app.conf.redis_backend_use_ssl = {
    "ssl_cert_reqs": None
}

# Windows fix (safe for Render too)
if sys.platform == "win32":
    app.conf.update(
        worker_pool="solo",
        worker_prefetch_multiplier=1
    )

# =========================
# 🔥 MONGODB CONNECTION
# =========================
try:
    client = MongoClient(MONGO_URI)
    db = client.get_database("dairy-sonogram")
    sonogram_collection = db["sonogramresults"]
    print("✅ MongoDB Connected")
except Exception as e:
    print("❌ MongoDB connection failed:", e)
    raise e

# =========================
# 🔥 CELERY TASK
# =========================
@app.task(name="tasks.predict")
def predict_sonogram_task(sonogram_id, image_path):
    print(f"📥 Received Task | ID: {sonogram_id}")

    try:
        # =========================
        # Update status → PROCESSING
        # =========================
        sonogram_collection.update_one(
            {"_id": ObjectId(sonogram_id)},
            {"$set": {"status": "PROCESSING"}}
        )

        print("🔄 Running ML model...")

        # =========================
        # 🔥 ML MODEL INFERENCE
        # =========================
        classification, confidence, predicted_yield = predict_image(image_path)

        print(f"✅ Prediction Done: {classification}, {confidence}, {predicted_yield}")

        # =========================
        # Save result to DB
        # =========================
        sonogram_collection.update_one(
            {"_id": ObjectId(sonogram_id)},
            {
                "$set": {
                    "status": "COMPLETED",
                    "classification": classification,
                    "confidence": float(confidence),
                    "predictedYield": float(predicted_yield)
                }
            }
        )

        return {
            "status": "success",
            "classification": classification,
            "confidence": float(confidence),
            "predictedYield": float(predicted_yield)
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

        # =========================
        # Update status → FAILED
        # =========================
        sonogram_collection.update_one(
            {"_id": ObjectId(sonogram_id)},
            {"$set": {"status": "FAILED"}}
        )

        return {
            "status": "failed",
            "error": str(e)
        }

# =========================
# 🔥 DEBUG START LOG
# =========================
print("🚀 Celery Worker Started...")

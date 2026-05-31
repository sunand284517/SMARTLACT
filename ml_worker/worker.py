import os
import sys
import ssl
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
# 🔥 CELERY SETUP & SSL CRASH FIX
# =========================
# Passing the SSL settings directly into the constructor blocks the URL parser error
app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE}
)

# Alternative fallback configuration properties
app.conf.update(
    broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    result_backend_transport_options={"ssl_cert_reqs": ssl.CERT_NONE}
)

# ✅ Windows / safe mode local runner option
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
    db = client["dairy-sonogram"]
    collection = db["sonogramresults"]
    print("✅ MongoDB Connected")
except Exception as e:
    print("❌ MongoDB connection failed:", e)
    raise e

# =========================
# 🔥 CELERY TASK
# =========================
@app.task(name="predict_task")
def predict_task(sonogram_id, image_path):
    print(f"📥 Task received | ID: {sonogram_id}")

    try:
        # =========================
        # Update status → PROCESSING
        # =========================
        collection.update_one(
            {"_id": ObjectId(sonogram_id)},
            {"$set": {"status": "PROCESSING"}}
        )

        print("🔄 Running ML model...")

        # =========================
        # 🔥 ML MODEL INFERENCE
        # =========================
        classification, confidence, predicted_yield = predict_image(image_path)

        print(f"✅ Prediction: {classification}, {confidence}, {predicted_yield}")

        # =========================
        # Save result to DB
        # =========================
        collection.update_one(
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
        collection.update_one(
            {"_id": ObjectId(sonogram_id)},
            {"$set": {"status": "FAILED"}}
        )

        return {
            "status": "failed",
            "error": str(e)
        }

# =========================
# 🔥 START LOG
# =========================
print("🚀 Celery Worker Started...")

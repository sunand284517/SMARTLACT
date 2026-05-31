import os
import sys
import ssl
from celery import Celery
from pymongo import MongoClient
from bson.objectid import ObjectId

from model import predict_image, load_model


# =========================
# ENV VARIABLES
# =========================
REDIS_URL = os.environ.get("CELERY_BROKER_URL")
MONGO_URI = os.environ.get("MONGO_URI")

if not REDIS_URL:
    raise ValueError("❌ CELERY_BROKER_URL is not set")

if not MONGO_URI:
    raise ValueError("❌ MONGO_URI is not set")


# =========================
# CELERY SETUP
# =========================
app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE}
)

app.conf.update(
    broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    result_backend_transport_options={"ssl_cert_reqs": ssl.CERT_NONE}
)

if sys.platform == "win32":
    app.conf.update(
        worker_pool="solo",
        worker_prefetch_multiplier=1
    )


# =========================
# MONGODB CONNECTION (FIXED)
# =========================
try:
    client = MongoClient(
        MONGO_URI,
        connectTimeoutMS=5000,
        serverSelectionTimeoutMS=5000
    )

    db = client.get_default_database()

    # FIX: NO boolean check on Database object
    if db is None:
        db = client["dairy-sonogram"]

    collection = db["sonogramresults"]

    print(f"✅ MongoDB Connected successfully to database: {db.name}")

except Exception as e:
    print("❌ MongoDB connection failed:", e)
    raise e


# =========================
# MODEL WARMUP (FIXED)
# =========================
try:
    print("🔥 Warming up ML model...")
    load_model()
    print("✅ Model loaded and ready")
except Exception as e:
    print("⚠️ Model warmup failed:", e)


# =========================
# SAFE OBJECTID
# =========================
def safe_objectid(id_str):
    try:
        return ObjectId(id_str)
    except Exception:
        return None


# =========================
# CELERY TASK
# =========================
@app.task(name="predict_task")
def predict_task(sonogram_id, image_path):
    print(f"📥 Task received | Record ID: {sonogram_id}")
    print(f"🖼️ Image URL: {image_path}")

    try:
        obj_id = safe_objectid(sonogram_id)
        if obj_id is None:
            raise ValueError("Invalid MongoDB ObjectId")

        # Update status → PROCESSING
        collection.update_one(
            {"_id": obj_id},
            {"$set": {"status": "PROCESSING"}}
        )

        print("🔄 Running ML inference...")

        # =========================
        # ML INFERENCE (FIXED)
        # =========================
        result = predict_image(image_path)

        if not isinstance(result, dict):
            raise ValueError("Model returned invalid response")

        if result.get("status") != "success":
            raise Exception(result.get("error", "Unknown prediction error"))

        classification = result["classification"]
        confidence = result["confidence"]

        print(f"✅ Prediction: {classification} | Conf: {confidence:.2f}")

        # =========================
        # SAVE RESULT
        # =========================
        collection.update_one(
            {"_id": obj_id},
            {
                "$set": {
                    "status": "COMPLETED",
                    "classification": classification,
                    "confidence": float(confidence)
                }
            }
        )

        print("💾 Result saved to MongoDB")

        return {
            "status": "success",
            "classification": classification,
            "confidence": float(confidence)
        }

    except Exception as e:
        print(f"❌ EXECUTION ERROR: {str(e)}")

        try:
            obj_id = safe_objectid(sonogram_id)
            if obj_id:
                collection.update_one(
                    {"_id": obj_id},
                    {
                        "$set": {
                            "status": "FAILED",
                            "errorReason": str(e)
                        }
                    }
                )
        except Exception as mongo_err:
            print(f"❌ Mongo update failed: {mongo_err}")

        return {
            "status": "failed",
            "error": str(e)
        }


# =========================
# START LOG
# =========================
print("🚀 Celery Worker Environment Initialized")

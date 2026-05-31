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
# MONGODB (FIXED SAFE)
# =========================
client = MongoClient(
    MONGO_URI,
    connectTimeoutMS=5000,
    serverSelectionTimeoutMS=5000
)

db = client.get_database() if client.get_database() else client["dairy-sonogram"]
collection = db["sonogramresults"]

print(f"✅ MongoDB Connected: {db.name}")


# =========================
# MODEL WARMUP (SAFE)
# =========================
try:
    print("🔥 Warming up ML model...")
    load_model()
    print("✅ Model ready")
except Exception as e:
    print("⚠️ Model warmup failed (non-fatal):", e)


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

    print(f"📥 Task | {sonogram_id}")
    print(f"🖼️ Image | {image_path}")

    try:
        obj_id = safe_objectid(sonogram_id)
        if not obj_id:
            raise ValueError("Invalid ObjectId")

        collection.update_one(
            {"_id": obj_id},
            {"$set": {"status": "PROCESSING"}}
        )

        print("🔄 Running inference...")

        result = predict_image(image_path)

        # =========================
        # HARD VALIDATION (IMPORTANT)
        # =========================
        if not isinstance(result, dict):
            raise ValueError("Model returned invalid output")

        if result.get("status") == "failed":
            raise ValueError(result.get("error", "Prediction failed"))

        classification = result.get("classification")
        confidence = float(result.get("confidence", 0.0))
        yield_litres = float(result.get("yield_litres", 0.0))

        print(f"✅ {classification} | Conf={confidence:.3f}")

        collection.update_one(
            {"_id": obj_id},
            {
                "$set": {
                    "status": "COMPLETED",
                    "classification": classification,
                    "confidence": confidence,
                    "yield_litres": yield_litres
                }
            }
        )

        return {
            "status": "success",
            "classification": classification,
            "confidence": confidence,
            "yield_litres": yield_litres
        }

    except Exception as e:

        print(f"❌ ERROR: {str(e)}")

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

        return {
            "status": "failed",
            "error": str(e)
        }


print("🚀 Celery Worker Ready")

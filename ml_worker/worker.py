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
    app.conf.update(worker_pool="solo", worker_prefetch_multiplier=1)


# =========================
# MONGODB (🔥 FIXED HERE)
# =========================
client = MongoClient(
    MONGO_URI,
    connectTimeoutMS=5000,
    serverSelectionTimeoutMS=5000
)

# ✅ ALWAYS use direct DB name (SAFE)
db = client["dairy-sonogram"]
collection = db["sonogramresults"]

print(f"✅ MongoDB Connected: {db.name}")


# =========================
# MODEL WARMUP
# =========================
try:
    print("🔥 Warming up ML model...")
    load_model()
    print("✅ Model ready")
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

    try:
        obj_id = safe_objectid(sonogram_id)
        if not obj_id:
            raise ValueError("Invalid ObjectId")

        collection.update_one(
            {"_id": obj_id},
            {"$set": {"status": "PROCESSING"}}
        )

        result = predict_image(image_path)

        if result.get("status") == "failed":
            raise ValueError(result.get("error"))

        collection.update_one(
            {"_id": obj_id},
            {
                "$set": {
                    "status": "COMPLETED",
                    "classification": result["classification"],
                    "confidence": float(result["confidence"]),
                    "yield_litres": float(result["yield_litres"])
                }
            }
        )

        return {"status": "success", **result}

    except Exception as e:

        obj_id = safe_objectid(sonogram_id)
        if obj_id:
            collection.update_one(
                {"_id": obj_id},
                {"$set": {"status": "FAILED", "errorReason": str(e)}}
            )

        return {"status": "failed", "error": str(e)}


print("🚀 Celery Worker Ready")

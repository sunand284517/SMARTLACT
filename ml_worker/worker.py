import os
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
    backend=REDIS_URL
)

# ✅ Required for Upstash (TLS)
app.conf.update(
    broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True
)


# =========================
# MONGODB
# =========================
client = MongoClient(MONGO_URI)
db = client["dairy-sonogram"]
collection = db["sonogramresults"]

print(f"✅ MongoDB Connected: {db.name}")


# =========================
# MODEL LOAD
# =========================
print("🔥 Loading ML model...")
load_model()
print("✅ Model ready")


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

    print(f"📥 Received task: {sonogram_id}")

    obj_id = safe_objectid(sonogram_id)
    if not obj_id:
        return {"status": "failed", "error": "Invalid ObjectId"}

    try:
        # ======================
        # 1. SET PROCESSING
        # ======================
        collection.update_one(
            {"_id": obj_id},
            {"$set": {"status": "processing"}}  # ✅ lowercase FIX
        )

        # ======================
        # 2. RUN MODEL
        # ======================
        result = predict_image(image_path)

        if result.get("status") == "failed":
            raise Exception(result.get("error"))

        # ======================
        # 3. SAVE RESULT
        # ======================
        collection.update_one(
            {"_id": obj_id},
            {
                "$set": {
                    "status": "completed",  # ✅ lowercase FIX
                    "classification": result["classification"],
                    "confidence": float(result["confidence"]),
                    "yield_litres": float(result["yield_litres"])
                }
            }
        )

        print(f"✅ Completed: {sonogram_id}")

        return {"status": "success"}

    except Exception as e:
        print(f"❌ Failed: {str(e)}")

        collection.update_one(
            {"_id": obj_id},
            {"$set": {"status": "failed", "errorReason": str(e)}}
        )

        return {"status": "failed", "error": str(e)}


print("🚀 Celery Worker Ready")

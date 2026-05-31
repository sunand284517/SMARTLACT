import os
import ssl
from celery import Celery
from pymongo import MongoClient
from bson.objectid import ObjectId

from model import predict_image

# =========================
# ENV
# =========================
REDIS_URL = os.environ.get("CELERY_BROKER_URL")
MONGO_URI = os.environ.get("MONGO_URI")

if not REDIS_URL:
    raise ValueError("CELERY_BROKER_URL missing")

if not MONGO_URI:
    raise ValueError("MONGO_URI missing")


# =========================
# CELERY
# =========================
app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE}
)


# =========================
# MONGO FIX (IMPORTANT)
# =========================
client = MongoClient(MONGO_URI)
db = client["dairy-sonogram"]
collection = db["sonogramresults"]

print("✅ MongoDB Connected")


# =========================
# TASK
# =========================
@app.task(name="predict_task")
def predict_task(sonogram_id, image_url):

    print("Task received:", sonogram_id)

    try:
        collection.update_one(
            {"_id": ObjectId(sonogram_id)},
            {"$set": {"status": "PROCESSING"}}
        )

        print("Running inference...")

        result = predict_image(image_url)

        print("Prediction:", result)

        collection.update_one(
            {"_id": ObjectId(sonogram_id)},
            {"$set": {
                "status": "COMPLETED",
                "classification": result["class"],
                "confidence": result["confidence"],
                "milk_yield": result["milk_yield_liters"]
            }}
        )

        return {"status": "success", "result": result}

    except Exception as e:
        print("ERROR:", str(e))

        collection.update_one(
            {"_id": ObjectId(sonogram_id)},
            {"$set": {"status": "FAILED", "error": str(e)}}
        )

        return {"status": "failed", "error": str(e)}

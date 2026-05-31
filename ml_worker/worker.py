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
    # ✅ FIX: Forcing explicit backend authentication parsing rules across the client cluster
    client = MongoClient(MONGO_URI, connectTimeoutMS=5000, serverSelectionTimeoutMS=5000)
    
    # Safely extract database name from URI if provided, otherwise fallback cleanly to your cluster name
    db_name = MongoClient(MONGO_URI).get_default_database()
    db = db_name if db_name is not None else client["dairy-sonogram"]
    
    collection = db["sonogramresults"]
    print(f"✅ MongoDB Connected successfully to database: {db.name}")
except Exception as e:
    print("❌ MongoDB connection failed:", e)
    raise e

# =========================
# 🔥 CELERY TASK
# =========================
@app.task(name="predict_task")
def predict_task(sonogram_id, image_path):
    print(f"📥 Task received | Record ID: {sonogram_id}")
    print(f"🖼️ Input Cloud asset pathway link: {image_path}")

    try:
        # =========================
        # Update status → PROCESSING
        # =========================
        result = collection.update_one(
            {"_id": ObjectId(sonogram_id)},
            {"$set": {"status": "PROCESSING"}}
        )
        
        if result.matched_count == 0:
            raise ValueError(f"Target record ID missing or dropped from cluster collection: {sonogram_id}")

        print("🔄 Running ML pipeline prediction layers...")

        # =========================
        # 🔥 ML MODEL INFERENCE
        # =========================
        # Pulls the network stream, downloads image bytes, and processes down to values
        classification, confidence, predicted_yield = predict_image(image_path)

        print(f"✅ Prediction generated: {classification} | Conf: {confidence:.2f} | Yield: {predicted_yield:.2f}")

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
        print(f"💾 Record states cleanly written to Mongo database collection.")

        return {
            "status": "success",
            "classification": classification,
            "confidence": float(confidence),
            "predictedYield": float(predicted_yield)
        }

    except Exception as e:
        print(f"❌ EXECUTION FAULT ENCOUNTERED: {str(e)}")

        # =========================
        # Update status → FAILED
        # =========================
        try:
            collection.update_one(
                {"_id": ObjectId(sonogram_id)},
                {"$set": {"status": "FAILED", "errorReason": str(e)}}
            )
        except Exception as mongo_err:
            print(f"❌ Failed to report fallback state back to Mongo: {mongo_err}")

        return {
            "status": "failed",
            "error": str(e)
        }

# =========================
# 🔥 START LOG
# =========================
print("🚀 Celery Worker Environment Core Stack Initialized...")

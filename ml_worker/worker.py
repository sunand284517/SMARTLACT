import os
from pymongo import MongoClient
from bson.objectid import ObjectId

from model import predict_image, load_model
from celery_app import celery 

MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is missing from Railway settings!")

client = MongoClient(MONGO_URI)

# =========================================================================
# 🎯 HARD TARGET: FORCE PRODUCTION DATABASE
# Points explicitly to 'smartlact' to override any default system fallbacks
# =========================================================================
db = client["smartlact"]
collection = db["sonogramresults"]

print(f"✅ Celery Connected to Production Database: [{db.name}] | Collection: [{collection.name}]")

_model = None

def get_model():
    global _model
    if _model is None:
        print("🔥 Loading ML model weights into RAM...")
        _model = load_model()
        print("✅ Model loaded successfully.")
    return _model


def safe_objectid(id_str):
    try:
        return ObjectId(id_str)
    except:
        return None


@celery.task(name="predict_task") 
def predict_task(sonogram_id, image_path):
    print(f"\n📥 Received task for processing ID: {sonogram_id}")

    obj_id = safe_objectid(sonogram_id)
    if not obj_id:
        return {"status": "failed", "error": "Invalid ObjectId structure"}

    try:
        # Move document status into processing state
        collection.update_one(
            {"_id": obj_id},
            {"$set": {"status": "processing"}}
        )

        model = get_model()
        result = predict_image(image_path)

        if not result or result.get("status") == "failed":
            raise Exception(result.get("error", "Model inference returned a failure status"))

        classification = result.get("classification", "Unknown")
        confidence = float(result.get("confidence", 0.0))
        yield_litres = float(result.get("yield_litres", 0.0))

        # =========================================================================
        # 📊 VISIBLE LOG BLOCKS IN RAILWAY
        # =========================================================================
        print("\n🚀 ================= ML INFERENCE EXECUTION =================")
        print(f"📋 CLASSIFICATION STAGE : {classification}")
        print(f"📈 CONFIDENCE LEVEL      : {confidence * 100:.2f}%")
        print(f"🥛 PREDICTED MILK YIELD  : {yield_litres} Litres")
        print("============================================================\n")

        # Update database document with completed metrics fields
        db_update = collection.update_one(
            {"_id": obj_id},
            {
                "$set": {
                    "status": "completed",
                    "classification": classification,
                    "confidence": confidence,
                    "yield_litres": yield_litres
                }
            }
        )
        
        print(f"✅ DB Synchronized -> Matched: {db_update.matched_count} | Modified: {db_update.modified_count}")
        return {"status": "success"}

    except Exception as e:
        print(f"❌ Worker Process Failure: {str(e)}")
        collection.update_one(
            {"_id": obj_id},
            {
                "$set": {
                    "status": "failed",
                    "errorReason": str(e)
                }
            }
        )
        return {"status": "failed", "error": str(e)}

import os
from pymongo import MongoClient
from bson.objectid import ObjectId

from model import predict_image, load_model
from celery_app import celery 

MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not set")

client = MongoClient(MONGO_URI)
db = client["dairy-sonogram"]
collection = db["sonogramresults"]

print(f"✅ MongoDB Connected: {db.name}")

_model = None

def get_model():
    global _model
    if _model is None:
        print("🔥 Loading ML model...")
        _model = load_model()
    return _model


def safe_objectid(id_str):
    try:
        return ObjectId(id_str)
    except:
        return None


@celery.task(name="predict_task") 
def predict_task(sonogram_id, image_path):

    print(f"📥 Received task: {sonogram_id}")

    obj_id = safe_objectid(sonogram_id)
    if not obj_id:
        return {"status": "failed", "error": "Invalid ObjectId"}

    try:
        collection.update_one(
            {"_id": obj_id},
            {"$set": {"status": "processing"}}
        )

        model = get_model()
        
        # NOTE: If your predict_image function inside model.py accepts the model instance, 
        # change this line to: result = predict_image(image_path, model)
        result = predict_image(image_path)

        if result.get("status") == "failed":
            raise Exception(result.get("error"))

        collection.update_one(
            {"_id": obj_id},
            {
                "$set": {
                    "status": "completed",
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
            {
                "$set": {
                    "status": "failed",
                    "errorReason": str(e)
                }
            }
        )

        return {"status": "failed", "error": str(e)}

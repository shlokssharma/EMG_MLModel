import time
import json
import joblib
import asyncio
import numpy as np

from pymongo import MongoClient

import websockets


# ==========================================================
# CONFIGURATION
# ==========================================================

MONGO_URI = "mongodb://localhost:27017"

DATABASE_NAME = "Stroke_Rehabilitation"

COLLECTION_NAME = "Patient_Data"

MODEL_NAME = "rehab_model.pkl"

WEBSOCKET_URI = "ws://localhost:8000/ws"

POLLING_TIME = 2


# ==========================================================
# CONNECT TO DATABASE
# ==========================================================

def connect_database():

    client = MongoClient(MONGO_URI)

    db = client[DATABASE_NAME]

    collection = db[COLLECTION_NAME]

    return collection


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def calculate_mav(raw_signal):

    raw_signal = np.array(raw_signal)

    return np.mean(np.abs(raw_signal))


def calculate_rms_mean(rms):

    rms = np.array(rms)

    return np.mean(rms)


def calculate_rms_std(rms):

    rms = np.array(rms)

    return np.std(rms)


def calculate_mfi(rms):

    rms = np.array(rms)

    rms_mean = np.mean(rms)

    rms_std = np.std(rms)

    return rms_std / (rms_mean + 1e-6)


def calculate_mqs(rms):

    rms = np.array(rms)

    rms_std = np.std(rms)

    score = 100 - (rms_std * 100)

    score = max(0, min(score, 100))

    return score


# ==========================================================
# GENERATE FEATURES
# ==========================================================

def generate_features(raw_signal, rms):

    mav = calculate_mav(raw_signal)

    rms_mean = calculate_rms_mean(rms)

    rms_std = calculate_rms_std(rms)

    mfi = calculate_mfi(rms)

    mqs = calculate_mqs(rms)

    return np.array([[

        mav,

        rms_mean,

        rms_std,

        mfi,

        mqs

    ]])


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

model = joblib.load(MODEL_NAME)


# ==========================================================
# GET LATEST RECORD
# ==========================================================

def get_latest_record(collection):

    latest = collection.find_one(

        sort=[("_id", -1)]

    )

    return latest


# ==========================================================
# PREDICT RECOVERY SCORE
# ==========================================================

def predict_score(record):

    raw_signal = record["raw_emg"]

    rms = record["rms"]

    features = generate_features(

        raw_signal,

        rms

    )

    prediction = model.predict(features)[0]

    return prediction

# ==========================================================
# DETERMINE RECOVERY STAGE
# ==========================================================

def determine_stage(score):

    if score < 40:

        return "Early Recovery"

    elif score < 70:

        return "Intermediate Recovery"

    else:

        return "Advanced Recovery"


# ==========================================================
# AI RECOMMENDATION
# ==========================================================

def generate_ai_recommendation(score, mfi, mqs):

    if mfi > 0.35:

        return "High muscle fatigue detected. Reduce exercise intensity and allow adequate recovery."

    elif mqs < 60:

        return "Movement quality is below the desired threshold. Emphasize controlled and guided exercises."

    elif score >= 70:

        return "Excellent rehabilitation progress. Continue with the current therapy protocol."

    else:

        return "Steady rehabilitation progress observed. Maintain the current rehabilitation program."


# ==========================================================
# SEND DATA TO DASHBOARD
# ==========================================================

async def send_to_dashboard(data):

    try:

        async with websockets.connect(WEBSOCKET_URI) as websocket:

            await websocket.send(json.dumps(data))

    except Exception as e:

        print(f"Dashboard Error : {e}")


# ==========================================================
# UPDATE DATABASE
# ==========================================================

def update_database(collection, record_id, score, stage, recommendation):

    collection.update_one(

        {"_id": record_id},

        {

            "$set": {

                "predicted_progress_score": float(score),

                "recovery_stage": stage,

                "ai_recommendation": recommendation

            }

        }

    )


# ==========================================================
# MAIN PREDICTION LOOP
# ==========================================================

async def prediction_loop():

    collection = connect_database()

    print("=" * 60)

    print("Stroke Rehabilitation Prediction Started")

    print("=" * 60)

    last_id = None

    while True:

        record = get_latest_record(collection)

        if record is None:

            time.sleep(POLLING_TIME)

            continue

        if record["_id"] == last_id:

            time.sleep(POLLING_TIME)

            continue

        last_id = record["_id"]

        raw_signal = record["raw_emg"]

        rms = record["rms"]

        features = generate_features(raw_signal, rms)

        prediction = model.predict(features)[0]

        score = float(prediction)

        stage = determine_stage(score)

        mfi = calculate_mfi(rms)

        mqs = calculate_mqs(rms)

        recommendation = generate_ai_recommendation(

            score,

            mfi,

            mqs

        )

        update_database(

            collection,

            record["_id"],

            score,

            stage,

            recommendation

        )

        dashboard_data = {

            "predicted_progress_score": score,

            "recovery_stage": stage,

            "muscle_fatigue_index": round(mfi, 3),

            "movement_quality_score": round(mqs, 2),

            "ai_recommendation": recommendation

        }

        await send_to_dashboard(dashboard_data)

        print()

        print("=" * 50)

        print("Prediction Completed")

        print("=" * 50)

        print(f"Predicted Score  : {score:.2f}")

        print(f"Recovery Stage   : {stage}")

        print(f"MFI              : {mfi:.3f}")

        print(f"MQS              : {mqs:.2f}")

        print(f"AI Insight       : {recommendation}")

        print()

        time.sleep(POLLING_TIME)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    asyncio.run(prediction_loop())
	
import os
import pickle
import pandas as pd
from app.ml.train import train_model

# 📁 Paths
MODEL_PATH = "app/ml/model.pkl"
COMMODITY_ENCODER_PATH = "app/ml/commodity_encoder.pkl"
GROUP_ENCODER_PATH = "app/ml/group_encoder.pkl"

# 🚀 Auto-train model if not found
if not os.path.exists(MODEL_PATH):
    print("⚠️ Model not found. Training now...")
    train_model()

# 📦 Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# 📦 Load encoders
with open(COMMODITY_ENCODER_PATH, "rb") as f:
    commodity_encoder = pickle.load(f)

with open(GROUP_ENCODER_PATH, "rb") as f:
    group_encoder = pickle.load(f)


def predict_price(
    day: int,
    month: int,
    year: int,
    price_t1: float,
    price_t2: float,
    price_t3: float,
    msp: float,
    arrival: float,
    commodity: str,
    group: str
) -> float:
    """
    Predict agricultural commodity price using lag features.

    Parameters:
    - price_t1: price at t-1 (yesterday)
    - price_t2: price at t-2
    - price_t3: price at t-3
    """

    try:
        # 🔤 Encode categorical features
        commodity_encoded = commodity_encoder.transform([commodity])[0]
        group_encoded = group_encoder.transform([group])[0]

        # 📊 Create input dataframe (MUST match training features)
        input_data = pd.DataFrame([{
            "day": day,
            "month": month,
            "year": year,
            "price_t-1": price_t1,
            "price_t-2": price_t2,
            "price_t-3": price_t3,
            "msp": msp,
            "arrival": arrival,
            "commodity_encoded": commodity_encoded,
            "group_encoded": group_encoded
        }])

        # 🔮 Predict price
        prediction = model.predict(input_data)

        return max(0.0, float(prediction[0]))

    except ValueError as e:
        raise ValueError(
            f"Prediction error: {str(e)}. "
            f"Check commodity/group and input features."
        )


# 🧪 Test run
if __name__ == "__main__":
    try:
        sample_prediction = predict_price(
            day=2,
            month=4,
            year=2025,
            price_t1=2670.55,
            price_t2=2650.00,
            price_t3=2630.00,
            msp=2775,
            arrival=190145.33,
            commodity="Bajra",
            group="Cereals"
        )

        print(f"✅ Predicted Price: {sample_prediction:.2f}")

    except Exception as e:
        print(f"❌ Error: {e}")
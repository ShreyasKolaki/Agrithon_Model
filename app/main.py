from fastapi import FastAPI
from datetime import datetime
import pandas as pd

from app.schemas import PredictionResponse
from app.ml.predict import predict_price
from app.utils.agent1 import agent1_chat
# 📂 Load dataset
DATA_PATH = "data/agri_market_dataset_1year.csv"
df = pd.read_csv(DATA_PATH)

# 🧹 Clean columns
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "Commodity": "commodity",
    "Commodity Group": "group",
    "MSP (Rs./Quintal)": "msp",
    "Arrival (Metric Tonnes)": "arrival",
    "Price": "price"
})

# 🕒 Convert date for proper sorting
df["date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["commodity", "date"])

app = FastAPI()


# 🏠 Health check
@app.get("/")
def home():
    return {"message": "Agri Price Prediction API is running 🚀"}


# 🔮 Predict price (UPDATED)
@app.get("/predict", response_model=PredictionResponse)
def get_prediction(commodity: str):
    now = datetime.now()

    # 📊 Filter commodity
    filtered = df[df["commodity"].str.lower() == commodity.lower()]

    if len(filtered) < 3:
        return {
            "predicted_price": 0.0,
            "suggestion": f"Not enough data for '{commodity}'"
        }

    # 🔁 Get last 3 prices (lag features)
    price_t1 = float(filtered.iloc[-1]["price"])
    price_t2 = float(filtered.iloc[-2]["price"])
    price_t3 = float(filtered.iloc[-3]["price"])

    # 📊 Get latest MSP & arrival
    latest = filtered.iloc[-1]
    msp = float(latest["msp"])
    arrival = float(latest["arrival"])

    # 🔤 Auto-get group (no user input needed)
    group = latest["group"]

    # 🔮 Predict
    predicted_price = predict_price(
        day=now.day,
        month=now.month,
        year=now.year,
        price_t1=price_t1,
        price_t2=price_t2,
        price_t3=price_t3,
        msp=msp,
        arrival=arrival,
        commodity=commodity,
        group=group
    )

    # 💡 Suggestion logic (compare with latest price)
    current_price = price_t1

    if predicted_price > current_price:
        suggestion = "HOLD"
    elif predicted_price < current_price:
        suggestion = "SELL"
    else:
        suggestion = "STABLE"

    return {
        "predicted_price": predicted_price,
        "suggestion": suggestion
    }

@app.get("/chatbot")
def chatbot(query: str):
    result = agent1_chat(query)
    return result
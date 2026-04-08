import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/agri_market_dataset_1year.csv"


def train_model():
    # 📥 Load dataset
    df = pd.read_csv(DATA_PATH)

    if len(df) < 50:
        print("❌ Not enough data to train model")
        return

    # 🧹 Clean column names
    df.columns = df.columns.str.strip()

    # 🔄 Rename columns
    df = df.rename(columns={
        "Date": "date",
        "Price": "price",
        "MSP (Rs./Quintal)": "msp",
        "Arrival (Metric Tonnes)": "arrival",
        "Commodity": "commodity",
        "Commodity Group": "group"
    })

    # 🕒 Convert date
    df["date"] = pd.to_datetime(df["date"])

    # 📅 Extract features
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # 🔤 Encode categorical features
    commodity_encoder = LabelEncoder()
    group_encoder = LabelEncoder()

    df["commodity_encoded"] = commodity_encoder.fit_transform(df["commodity"])
    df["group_encoded"] = group_encoder.fit_transform(df["group"])

    # 🔁 Sort by commodity + date (VERY IMPORTANT)
    df = df.sort_values(["commodity", "date"])

    # 🔁 Create lag features (IMPORTANT 🔥)
    df["price_t-1"] = df.groupby("commodity")["price"].shift(1)
    df["price_t-2"] = df.groupby("commodity")["price"].shift(2)
    df["price_t-3"] = df.groupby("commodity")["price"].shift(3)

    # Remove NaN rows (caused by lagging)
    df = df.dropna()

    # 🎯 Features & target
    X = df[
    [
        "day",
        "month",
        "year",
        "price_t-1",
        "price_t-2",
        "price_t-3",
        "msp",
        "arrival",
        "commodity_encoded",
        "group_encoded",
    ]
]
    y = df["price"]

    # 📊 Train-test split (IMPORTANT: no shuffle for time-series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 🤖 Train model
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    # 🔮 Predictions
    y_pred = model.predict(X_test)

    # 📊 Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 MODEL PERFORMANCE")
    print(f"MAE  (avg error): ₹{mae:.2f}")
    print(f"RMSE (penalty):   ₹{rmse:.2f}")
    print(f"R² Score:         {r2:.4f}")

    # 💾 Save model + encoders
    with open("app/ml/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("app/ml/commodity_encoder.pkl", "wb") as f:
        pickle.dump(commodity_encoder, f)

    with open("app/ml/group_encoder.pkl", "wb") as f:
        pickle.dump(group_encoder, f)

    print("\n✅ Model trained and saved successfully!")


if __name__ == "__main__":
    train_model()
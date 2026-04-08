from pydantic import BaseModel


# 🔮 Prediction request schema
class PredictionInput(BaseModel):
    commodity: str
    group: str
    prev_price: float


# 📤 Prediction response schema
class PredictionResponse(BaseModel):
    predicted_price: float
    suggestion: str
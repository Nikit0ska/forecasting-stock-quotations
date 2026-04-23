from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime

class ForecastResponse(BaseModel):
    train: List[float]
    val: List[float]
    test: List[float]
    forecast: List[float]
    metrics: Dict[str, float]
    dates: List[str]
    # counter: Dict[str, int]

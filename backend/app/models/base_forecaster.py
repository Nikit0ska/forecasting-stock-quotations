from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class BaseForecaster(ABC):
    @abstractmethod
    def forecast(
        self,
        full_series: pd.Series,
        train_end: pd.Timestamp,
        val_end: pd.Timestamp,
        steps: int,
        **kwargs
    ) -> Dict[str, Any]:
        pass
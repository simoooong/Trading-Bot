from abc import ABC, abstractmethod
from typing import List, Dict, Any

class DatabaseInterface(ABC):
    @abstractmethod
    def get_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch stock data from the database for a given symbol between two dates."""
        pass

    @abstractmethod
    def store_data(self, symbol: str, data: List[Dict[str, Any]]) -> None:
        """Store stock data for a given symbol."""
        pass

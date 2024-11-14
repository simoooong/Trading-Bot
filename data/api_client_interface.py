from abc import ABC, abstractmethod

class ApiClientInterface(ABC):
    @abstractmethod
    def fetch_data_from_api(self, symbol: str, year: int,  month: int):
        """Fetch time series data for a specified financial symbol over a given period."""
        pass
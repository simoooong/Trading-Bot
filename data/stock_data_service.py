from data.database_interface import DatabaseInterface
from data.api_handler import ApiClient
from typing import List, Dict, Any
from calendar import monthrange

class StockDataService:
    def __init__(self, db: DatabaseInterface, api_client: ApiClient):
        self.db = db
        self.api_client = api_client

    def get_monthly_data(self, symbol: str, year: int, month: int) -> List[Dict[str, Any]]:
        start_date = f"{year}-{str(month).zfill(2)}-01"
        end_date = f"{year}-{str(month).zfill(2)}-{monthrange(year, month)[1]}"

        # Try fetching data from the database first
        data = self.db.get_data(symbol, start_date, end_date)
        if data:
            print("Data retrieved from database.")  
            return data
        
        # If no data in the database, fetch it from the API
        data = self.api_client.fetch_data_from_api(symbol, year, month)
        self.db.store_data(symbol, data)
        
        return data

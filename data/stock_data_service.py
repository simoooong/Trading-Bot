from data.database_interface import DatabaseInterface
from data.api_client_interface import ApiClientInterface
from typing import List, Dict, Any
from calendar import monthrange

class StockDataService:
    def __init__(self, db: DatabaseInterface, api_client: ApiClientInterface):
        self.db = db
        self.api_client = api_client

    def get_monthly_data(self, symbol: str, year: int, month: int) -> List[Dict[str, Any]]:
        start_date = f"{year}-{str(month).zfill(2)}-01"
        end_date = f"{year}-{str(month).zfill(2)}-{monthrange(year, month)[1]}"

        # Try fetching data from the database first
        data = self.db.get_data(symbol, start_date, end_date)
        if data:
            #print("Data retrieved from database.")  
            return data
        
        # If no data in the database, fetch it from the API
        data = self.api_client.fetch_data_from_api(symbol, year, month)
        self.db.store_data(symbol, data)
        
        return data
    
    def get_data(self, symbol, start_date, end_date):

        print_start = f"{start_date[0]}-{str(start_date[1]).zfill(2)}-01"
        print_end = f"{end_date[0]}-{str(end_date[1]).zfill(2)}-{monthrange(end_date[0], end_date[1])[1]}"
        print(print_start, print_end)

        data = []

        current_year, current_month = start_date
        end_year, end_month = end_date

        while (current_year, current_month) <= (end_year, end_month):
            monthly_data = self.get_monthly_data(symbol, current_year, current_month)

            if monthly_data is not None:
                data.append(monthly_data)
                
            if current_month == 12:
                current_month = 1
                current_year += 1
            else:
                current_month += 1

        return data
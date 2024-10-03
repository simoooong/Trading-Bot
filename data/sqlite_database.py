import sqlite3
from datetime import datetime
from typing import List, Dict, Any
from .database_interface import DatabaseInterface

class SQLiteDatabase(DatabaseInterface):
    def __init__(self, db_file: str):
        try:
            self.conn = sqlite3.connect(db_file)
            self._create_table()
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")

    def _create_table(self):
        """Create the stock_data table if it doesn't exist."""
        try:
            with self.conn:
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS stock_data (
                        symbol TEXT NOT NULL,
                        date DATE NOT NULL,
                        time TIME NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        PRIMARY KEY (symbol, date, time)
                    )
                ''')
        except sqlite3.Error as e:
            print(f"Error creating table: {e}")

    def get_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch data for a given symbol between the start and end dates."""
        try:
            print(start_date, end_date)
            cursor = self.conn.cursor()
            query = '''
                SELECT * FROM stock_data
                WHERE symbol = ? AND date BETWEEN ? AND ?
            '''
            cursor.execute(query, (symbol, start_date, end_date))
            rows = cursor.fetchall()

            # Convert the rows to a list of dictionaries
            return [
                {
                    "symbol": row[0],
                    "date": row[1],
                    "time": row[2],
                    "open": row[3],
                    "high": row[4],
                    "low": row[5],
                    "close": row[6],
                    "volume": row[7]
                }
                for row in rows
            ]
        except sqlite3.Error as e:
            print(f"Error fetching data: {e}")
            return []

    def store_data(self, symbol: str, data: List[Dict[str, Any]]) -> None:
        """Store a list of stock data in the database."""
        try:
            cursor = self.conn.cursor()
            with self.conn:
                cursor.executemany('''
                    INSERT OR IGNORE INTO stock_data (symbol, date, time, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', [
                    (
                        symbol,
                        item['date'],
                        item['time'],
                        item['open'],
                        item['high'],
                        item['low'],
                        item['close'],
                        item['volume']
                    )
                    for item in data
                ])
        except sqlite3.Error as e:
            print(f"Error storing data: {e}")

    def __del__(self):
        """Close the database connection when the object is destroyed."""
        if self.conn:
            self.conn.close()

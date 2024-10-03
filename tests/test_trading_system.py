import unittest
from unittest.mock import MagicMock
from trading_system import TradingSystem
from data.stock_data_service import StockDataService
from portfolio import Portfolio
from strategy import BasicStrategy

class TestTradingSystem(unittest.TestCase):

    def setUp(self):
        initial_balance = 100000
        self.portfolio = Portfolio(risk_tolerance=1, initial_balance=initial_balance)
        strategy = BasicStrategy(self.portfolio)
        data_service = MagicMock()
        self.trading_system = TradingSystem(self.portfolio, strategy, data_service)

        self.trading_stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C']

        data_service.get_monthly_data.side_effect = lambda symbol, year, month: {
            # take_profit at 2% before drop
            'STOCK_A': [
                {'symbol': 'STOCK_A', 'date': '2024-08-01', 'time': '16:00:00', 'open': 100, 'high': 100, 'low': 100, 'close': 100, 'volume': 1000},  # 2% rise
                {'symbol': 'STOCK_A', 'date': '2024-08-01', 'time': '17:00:00', 'open': 102, 'high': 102, 'low': 102, 'close': 102, 'volume': 1000},  # 2% rise
                {'symbol': 'STOCK_A', 'date': '2024-08-01', 'time': '18:00:00', 'open': 95, 'high': 95, 'low': 95, 'close': 95, 'volume': 1000},  # 2% rise
            ],
            # stopp_loss at 1% before rise
            'STOCK_B': [
                {'symbol': 'STOCK_B', 'date': '2024-08-02', 'time': '16:00:00', 'open': 100, 'high': 100, 'low': 100, 'close': 100, 'volume': 1000},
                {'symbol': 'STOCK_B', 'date': '2024-08-02', 'time': '17:00:00', 'open': 99, 'high': 99, 'low': 99, 'close': 99, 'volume': 1000},
                {'symbol': 'STOCK_B', 'date': '2024-08-02', 'time': '18:00:00', 'open': 105, 'high': 105, 'low': 105, 'close': 105, 'volume': 1000},
            ],
            # no trigger just sell at the end of the day
            'STOCK_C': [
                {'symbol': 'STOCK_C', 'date': '2024-08-03', 'time': '16:00:00', 'open': 100, 'high': 100, 'low': 100, 'close': 100, 'volume': 1000},
                {'symbol': 'STOCK_C', 'date': '2024-08-03', 'time': '17:00:00', 'open': 101, 'high': 101, 'low': 101, 'close': 101, 'volume': 1000},
            ]
        }[symbol]

    def test_trading_simulation_with_mocked_data(self):
        self.trading_system.run_trading_simulation(self.trading_stocks, trading_start=(2024, 8), trading_end=(2024, 8))

        print(self.get)

        assert self.portfolio.get_positions_symbol('STOCK_A') is None
        assert self.portfolio.get_positions_symbol('STOCK_B') is None
        assert self.portfolio.get_positions_symbol('STOCK_C') is None

        self.assertEqual(self.portfolio.get_balance(), 101989)

        expceted_trade_history = [
            {'symbol': 'STOCK_A', 'date': '2024-08-01-16:00:00', 'quantity': 1000, 'price': '100.0000', 'type': 'long_entry'},
            {'symbol': 'STOCK_A', 'date': '2024-08-01-17:00:00', 'quantity': 1000, 'price': '102.0000', 'type': 'long_exit', 'profit_loss': '2000.0000'},
            {'symbol': 'STOCK_A', 'date': '2024-08-01-18:00:00', 'quantity': 1073, 'price': '95.0000', 'type': 'long_entry'},
            {'symbol': 'STOCK_A', 'date': '2024-08-01-18:00:00', 'quantity': 1073, 'price': '95.0000', 'type': 'long_exit', 'profit_loss': '0.0000'},
            {'symbol': 'STOCK_B', 'date': '2024-08-02-16:00:00', 'quantity': 1020, 'price': '100.0000', 'type': 'long_entry'},
            {'symbol': 'STOCK_B', 'date': '2024-08-02-17:00:00', 'quantity': 1020, 'price': '99.0000', 'type': 'long_exit', 'profit_loss': '-1020.0000'},
            {'symbol': 'STOCK_B', 'date': '2024-08-02-18:00:00', 'quantity': 961, 'price': '105.0000', 'type': 'long_entry'},
            {'symbol': 'STOCK_B', 'date': '2024-08-02-18:00:00', 'quantity': 961, 'price': '105.0000', 'type': 'long_exit', 'profit_loss': '0.0000'},
            {'symbol': 'STOCK_C', 'date': '2024-08-03-16:00:00', 'quantity': 1009, 'price': '100.0000', 'type': 'long_entry'},
            {'symbol': 'STOCK_C', 'date': '2024-08-03-17:00:00', 'quantity': 1009, 'price': '101.0000', 'type': 'long_exit', 'profit_loss': '1009.0000'},
        ]

        self.assertEqual(self.portfolio.get_trade_history(), expceted_trade_history)

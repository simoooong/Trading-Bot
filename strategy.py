from portfolio import Portfolio

class BasicStrategy:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

    def should_enter_trade(self, symbol, price_data):
        # ToDo: define trading strategy
        if self.portfolio.get_positions_symbol(symbol) is not None:
            return False

        return True

    def enter_trade_long(self, symbol, entry_price, date):
        quantity = int(self.portfolio.get_balance() / entry_price)
        entry_price = entry_price
        stop_loss = entry_price * 0.99
        take_profit = entry_price * 1.02

        self.portfolio.trade_long(symbol, date, quantity, entry_price, stop_loss, take_profit)

    def exit_trade_long(self, symbol, current_price, date):
        self.portfolio.exit_long(symbol, date, current_price)

    def check_for_exit(self, symbol, current_price):
        positions = self.portfolio.get_positions()
        
        if symbol not in positions:
            return False
    
        position = positions[symbol]
        
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
    
        if current_price <= stop_loss or current_price >= take_profit:
            return True
    
        return False
    
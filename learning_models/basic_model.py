from learning_models.strategy_interface import TradingStrategy

class BasicModel(TradingStrategy):
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
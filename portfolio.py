class Portfolio:
    def __init__(self, risk_tolerance, initial_balance = 100000):
        self.risk_tolerance = risk_tolerance
        self.balance = initial_balance
        self.active_positions = {}
        self.trade_history = []
    
    def check_order(self, type, quantity, price, stop_loss, take_profit, lever=1):
        cost = quantity * price
        
        if cost > self.balance:
            raise ValueError("Not enough balance to place this long order.")
        
        if type == 'long':
            pot_loss = lever * quantity * (price - stop_loss)

            if stop_loss >= price or take_profit <= price:
                raise ValueError("Invalid stop loss or take profit levels.")
            if pot_loss >= self.balance * self.risk_tolerance:
                raise ValueError("Potential loss exceeds risk tolerance.")
        elif type == 'short':
            pot_loss = lever * quantity * (stop_loss - price)

            if stop_loss <= price or take_profit >= price:
                raise ValueError("Invalid stop loss or take profit levels for short.")
            if pot_loss >= self.balance * self.risk_tolerance:
                raise ValueError("Potential loss exceeds risk tolerance.")

        return True
    
    def trade_long(self, symbol, date, quantity, price, stop_loss, take_profit, lever=1):
        self.check_order('long', quantity, price, stop_loss, take_profit, lever)
        
        if symbol in self.active_positions:
            raise ValueError(f"Position already exists for {symbol}. Close it before placing a new one.")
        
        cost = quantity * price
        self.balance -= cost
        self.active_positions[symbol] = {
            'type': 'long',
            'quantity': quantity,
            'price': price, 
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'lever': lever}
        
        self.trade_history.append({
            'symbol': symbol,
            'date': date,
            'quantity': quantity,
            'price': price,
            'type': 'long_entry'
        })

    def exit_long(self, symbol, date, price):
        if symbol not in self.active_positions or self.active_positions[symbol]['type'] != 'long':
            raise ValueError(f"No active long position found for {symbol}.")

        position = self.active_positions[symbol]

        quantity = position['quantity']
        entry_price = position['price']
        profit_loss = (price - entry_price) * quantity * position['lever']
        self.balance += quantity * price

        del self.active_positions[symbol]
        self.trade_history.append({
            'symbol': symbol,
            'date': date,
            'quantity': quantity,
            'price': price,
            'type': 'long_exit',
            'profit_loss': profit_loss
        })

    def close_all_positions(self, symbol, closing_market_price, date):
        if symbol in self.active_positions:
            self.exit_long(symbol, date, closing_market_price)
   
    def get_risk_tolerance(self):
        return self.risk_tolerance

    def get_balance(self):
        return self.balance
    
    def get_positions(self):
        return self.active_positions
    
    def get_positions_symbol(self, symbol):
        if symbol in self.active_positions:
            return self.active_positions[symbol]
        else:
            return None
    
    def get_trade_history(self):
        return self.trade_history

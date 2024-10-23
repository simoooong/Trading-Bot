import matplotlib.pyplot as plt
import pandas as pd

def generate_portfolio_data(stock_data, trade_history, initial_balance=100000):
    portfolio_data = []
    #trade_history_sorted = sorted(trade_history, key=lambda x: x['date'])
    
    current_balance = initial_balance
    trade_index = 0

    for i, stock_row in stock_data.iterrows():
        stock_date = stock_row['date']
        stock_time = stock_row['time']
        stock_datetime = f"{stock_date}-{stock_time}"

        while trade_index < len(trade_history) and trade_history[trade_index]['date'] == stock_datetime:
            trade = trade_history[trade_index]
            if trade['type'] == 'long_exit':
                profit_loss = float(trade['profit_loss'])
                current_balance += profit_loss

            trade_index += 1
            
    
        portfolio_data.append(current_balance)

    return portfolio_data

def visualize_performance(stock_data, trade_history, initial_balance):
    sorted_stock_data = stock_data.sort_values(by=['date', 'time'])
    symbol = stock_data['symbol'].iloc[0]
    print(sorted_stock_data)

    sorted_stock_data['portfolio_data'] = generate_portfolio_data(sorted_stock_data, trade_history, initial_balance)

    sorted_stock_data['relative_stock'] = sorted_stock_data['close'] / sorted_stock_data['close'].iloc[0]
    sorted_stock_data['relative portfolio'] = sorted_stock_data['portfolio_data'] / initial_balance
    
    plt.figure(figsize=(14, 7))

    plt.plot(sorted_stock_data['date'][::10], sorted_stock_data['relative_stock'][::10], label=f"{symbol}", color='black', linewidth=2)
    #plt.plot(sorted_stock_data['date'], sorted_stock_data['relative_stock'], label='Stock Price', color='black', linewidth=2)
    plt.plot(sorted_stock_data['date'][::10], sorted_stock_data['relative portfolio'][::10], label='Trading Bot', color='red', linewidth=2)

    plt.title(f"{symbol} vs Trading Bot over Time", fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Relative Performance', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)
    
    plt.show()
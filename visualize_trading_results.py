import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os

def generate_portfolio_data(stock_data, trade_history, initial_balance=100000):
    portfolio_data = []
    #trade_history_sorted = sorted(trade_history, key=lambda x: x['date'])
    
    current_balance = initial_balance
    trade_index = 0

    while trade_index < len(trade_history):
        stock_date = stock_data['date'].iloc[0]
        stock_time = stock_data['time'].iloc[0]
        stock_datetime = f"{stock_date}-{stock_time}"
        if trade_history[trade_index]['date'] >= stock_datetime:
            break
        
        trade_index += 1

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

    plt.plot(sorted_stock_data['date'][::10], sorted_stock_data['relative_stock'], label=f"{symbol}", color='black', linewidth=2)
    #plt.plot(sorted_stock_data['date'], sorted_stock_data['relative_stock'], label='Stock Price', color='black', linewidth=2)
    plt.plot(sorted_stock_data['date'], sorted_stock_data['relative portfolio'], label='Trading Bot', color='red', linewidth=2)

    plt.title(f"{symbol} vs Trading Bot over Time", fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Relative Performance', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)
    
    plt.show()

def visualize_multiple_performance(
    stock_data, trade_history_logs, initial_balance, save,
    save_dir="figures", model_name="Logistic Regression", hyper_opt=False, trained_on_all=False
):
    # Sort the stock data for consistent plotting
    sorted_stock_data = stock_data.sort_values(by=['date', 'time'])
    symbol = stock_data['symbol'].iloc[0]
    sorted_stock_data['relative_stock'] = sorted_stock_data['close'] / sorted_stock_data['close'].iloc[0]
    performance = (sorted_stock_data['relative_stock'].iloc[-1] - 1) * 100

    colors = list(mcolors.TABLEAU_COLORS.values())

    # Create the plot
    plt.figure(figsize=(14, 7))
    plt.plot(
        sorted_stock_data['date'][::10], 
        sorted_stock_data['relative_stock'][::10], 
        label=f"{symbol} ({performance:.2f} %)",
        color='black',
        linewidth=2
    )

    # Add portfolio performance lines for each trade history log
    for i, (trade_model_name, trade_history) in enumerate(trade_history_logs.items()):
        if trade_history is None:
            continue
        sorted_stock_data['portfolio_data'] = generate_portfolio_data(sorted_stock_data, trade_history, initial_balance)
        sorted_stock_data['relative_portfolio'] = sorted_stock_data['portfolio_data'] / initial_balance
        performance = (sorted_stock_data['relative_portfolio'].iloc[-1] - 1) * 100
        plt.plot(
            sorted_stock_data['date'][::10],
            sorted_stock_data['relative_portfolio'][::10],
            label=f"{trade_model_name} ({performance:.2f} %)",
            color=colors[i % len(colors)],
            linewidth=2
        )

    # Customize title with hyperparameter optimization and model name
    hyper_opt_text = "With Hyperparameter Optimization" if hyper_opt else "Without Hyperparameter Optimization"
    plt.title(f"{symbol} vs Trading Bots over Time\n{model_name} - {hyper_opt_text}", fontsize=16)

    # Customize labels, legend, and grid
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Relative Performance', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save or display the plot
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create the directory if it doesn't exist

        # Generate a descriptive filename
        hyper_opt_suffix = "hyperopt" if hyper_opt else "no_hyperopt"
        trained_suffix = "trained_all" if trained_on_all else "trained_simple"
        save_file_path = os.path.join(
            save_dir, f"{symbol}_{model_name.replace(' ', '_').lower()}_{hyper_opt_suffix}_{trained_suffix}.pdf"
        )

        plt.savefig(save_file_path, format='pdf', dpi=300)
        print(f"Plot saved at: {save_file_path}")
    else:
        plt.show()

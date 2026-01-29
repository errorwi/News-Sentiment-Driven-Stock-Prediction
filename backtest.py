def backtest(df, predictions):
    df = df.copy()
    df['Signal'] = predictions
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']

    strategy = df['Strategy_Return'].sum()
    buy_hold = df['Return'].sum()

    return strategy, buy_hold

import pandas as pd
from sentiment import compute_daily_sentiment
from models import train_models
from backtest import backtest

# Load data
stock = pd.read_csv("data/stock.csv", index_col=0)
print(stock.columns)
stock.index = pd.to_datetime(stock.index, errors='coerce').date

# Stock cleanup
stock = stock.dropna(subset=['Return', 'Direction'])

news = pd.read_csv(
    "data/news.csv",
    parse_dates=['date']
)

# Sentiment aggregation
sentiment = compute_daily_sentiment(news)
sentiment.index = pd.to_datetime(sentiment.index).date


# Merge stock + sentiment
df = stock.merge(
    sentiment,
    left_index=True,
    right_index=True,
    how="left"
)

df = df[~df.index.isna()]
df[['vader', 'finbert']] = df[['vader', 'finbert']].fillna(0.0)



print("Merged shape:", df.shape)
print(df.head())

# Predict next day movement
df['Target'] = df['Direction'].shift(-1)

# Drop NaNs only where they matter
df = df.dropna(subset=['Target', 'vader', 'finbert'])


# Features & labels
X = df[['vader', 'finbert']]
y = df['Target']

# Time-series split (NO shuffle)
split = int(0.8 * len(df))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train models
results = train_models(X_train, y_train, X_test, y_test)

# Backtest
for name, (model, acc) in results.items():
    preds = model.predict(X)
    strat, bh = backtest(df, preds)

    print(f"{name}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Excess Return: {(strat - bh):.4f}\n")

import pandas as pd
from sentiment import compute_daily_sentiment
from models import (
    train_models,
    create_sequences,
    build_lstm_model,
    train_lstm_model,
    predict_and_evaluate,
)
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


# -------------------------
# LSTM example (chronological, window=5)
# -------------------------
try:
    # Prepare features: sentiment + return (optional: add volume if available)
    df_lstm = df.copy()

    feature_cols = ['vader', 'Return']
    # If Volume exists, include it
    if 'Volume' in df_lstm.columns:
        feature_cols.append('Volume')

    df_lstm = df_lstm[feature_cols + ['Return']].dropna()

    window_size = 5
    X_seq, y_seq = create_sequences(df_lstm, feature_cols=feature_cols, target_col='Return', window_size=window_size)

    if len(X_seq) > 0:
        n = len(X_seq)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        X_train_seq = X_seq[:train_end]
        y_train_seq = y_seq[:train_end]
        X_val_seq = X_seq[train_end:val_end]
        y_val_seq = y_seq[train_end:val_end]
        X_test_seq = X_seq[val_end:]
        y_test_seq = y_seq[val_end:]

        input_shape = (window_size, X_seq.shape[2])
        lstm = build_lstm_model(input_shape=input_shape, learning_rate=0.001)

        lstm, history = train_lstm_model(
            lstm, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            epochs=50, batch_size=16, patience=5
        )

        eval_res = predict_and_evaluate(lstm, X_test_seq, y_test_seq)
        print("LSTM results on test set:")
        print(f"Accuracy: {eval_res['accuracy']:.4f}")
        print(f"AUC: {eval_res['auc']}")
    else:
        print("Not enough data to build LSTM sequences.")
except Exception as e:
    print("LSTM integration failed:", e)

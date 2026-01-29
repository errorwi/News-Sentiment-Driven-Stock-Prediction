import yfinance as yf
import pandas as pd
from GoogleNews import GoogleNews

def collect_stock_data(ticker, start, end):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )
    df = df[['Close']]
    df['Return'] = df['Close'].pct_change()
    df['Direction'] = (df['Return'] > 0).astype(int)
    return df.dropna()

def collect_news(query, start, end):
    googlenews = GoogleNews(lang='en', start=start, end=end)
    googlenews.search(query)
    results = googlenews.result()

    news = []
    for r in results:
        news.append({
            'date': r['date'],
            'title': r['title']
        })

    df = pd.DataFrame(news)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.dropna()

if __name__ == "__main__":
    stock = collect_stock_data("AAPL", "2024-01-01", "2025-01-01")
    news = collect_news("AAPL stock", "01/01/2024", "01/01/2025")

    stock.to_csv("data/stock.csv")
    news.to_csv("data/news.csv", index=False)

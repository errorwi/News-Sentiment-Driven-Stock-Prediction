import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    use_safetensors=True
)


vader = SentimentIntensityAnalyzer()
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def vader_score(text):
    return vader.polarity_scores(text)['compound']

def finbert_score(text):
    out = finbert(text)[0]
    return out['score'] if out['label'] == 'positive' else -out['score']

def compute_daily_sentiment(news_df):
    news_df['vader'] = news_df['title'].apply(vader_score)
    news_df['finbert'] = news_df['title'].apply(finbert_score)

    news_df['timestamp'] = pd.to_datetime(news_df['date'], utc=True, errors='coerce')
    news_df['date'] = news_df['timestamp'].dt.date

    daily = news_df.groupby('date').mean()

    daily.index = pd.to_datetime(daily.index)
    return daily

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os

# Downloading VADER lexicon
nltk.download('vader_lexicon')

# Loading feedback dataset
csv_path = os.path.join('data', 'googleplay_feedback.csv')
df = pd.read_csv(csv_path)

# Initializing VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Applying sentiment analysis
def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    return pd.Series([scores['neg'], scores['neu'], scores['pos'], scores['compound']])

df[['neg', 'neu', 'pos', 'compound']] = df['feedback_text'].apply(analyze_sentiment)

# Adding sentiment label (positive/negative/neutral)
def get_label(compound):
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_label'] = df['compound'].apply(get_label)

# Showing results
print(df[['department', 'feedback_text', 'sentiment_label', 'compound']])


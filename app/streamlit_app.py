import pandas as pd
import streamlit as st
import altair as alt
import os
from collections import Counter
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Setting wide layout
st.set_page_config(page_title="InsightAI – Feedback Sentiment Analysis", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .block-container {
        padding: 2rem 4rem;
    }
    .big-title {
        font-size: 2rem !important;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.3rem !important;
        font-weight: 600;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Loading data
csv_path = os.path.join('data', 'googleplay_feedback.csv')
@st.cache_data
def load_data():
    return pd.read_csv(csv_path)

df = load_data()

# NLTK Sentiment Analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
sid = SentimentIntensityAnalyzer()

# Sentiment scoring
def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    return pd.Series([scores['neg'], scores['neu'], scores['pos'], scores['compound']])

df[['neg', 'neu', 'pos', 'compound']] = df['feedback_text'].apply(analyze_sentiment)

def get_label(compound):
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_label'] = df['compound'].apply(get_label)

# ─────────────────────────────────────────────
# Title and Description
st.markdown("<div class='big-title'>📈 InsightAI – Feedback Sentiment Analysis Dashboard</div>", unsafe_allow_html=True)
st.markdown("Explore Google Play Store reviews with sentiment analysis, keyword analytics, and department-level insights.")

# ─────────────────────────────────────────────
# Sidebar Filters
st.sidebar.title("Filters")
departments = df['department'].unique().tolist()
selected_department = st.sidebar.multiselect("Select Apps", departments, default=departments)

filtered_df = df[df['department'].isin(selected_department)]

row_limit = st.sidebar.slider("Max Rows", min_value=1000, max_value=len(filtered_df), value=5000, step=1000)
filtered_df = filtered_df.head(row_limit)

# ─────────────────────────────────────────────
# Insight Summary Cards
st.markdown("### 🔍 Insight Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Reviews", len(filtered_df))

with col2:
    pos_pct = (filtered_df['sentiment_label'] == 'Positive').mean() * 100
    st.metric("Positive Feedback %", f"{pos_pct:.1f}%")

with col3:
    top_app = filtered_df['department'].value_counts().idxmax()
    st.metric("Most Reviewed App", top_app)

with col4:
    stop_words = set(stopwords.words('english'))
    all_words = ' '.join(filtered_df['feedback_text'].dropna()).lower().split()
    all_words = [w for w in all_words if w not in stop_words and len(w) > 2]
    top_kw = Counter(all_words).most_common(1)[0][0] if all_words else "N/A"
    st.metric("Top Keyword", top_kw)

st.markdown("---")

# ─────────────────────────────────────────────
# Sentiment Distribution
st.markdown("### 📌 Sentiment Distribution")
sentiment_counts = filtered_df['sentiment_label'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
chart = alt.Chart(sentiment_counts).mark_bar().encode(
    x='Sentiment',
    y='Count',
    color='Sentiment'
)
st.altair_chart(chart, use_container_width=True)

st.markdown("### 📊 Sentiment Proportion (Pie Chart)")
pie_chart = alt.Chart(sentiment_counts).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field="Count", type="quantitative"),
    color=alt.Color(field="Sentiment", type="nominal"),
    tooltip=['Sentiment', 'Count']
)
st.altair_chart(pie_chart, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# Keyword Frequency Analysis

st.subheader("Top Keywords in Feedback")

stop_words = set(stopwords.words('english'))

def extract_keywords(text_series):
    all_words = []
    for text in text_series.dropna():
        words = re.findall(r'\b\w+\b', text.lower())
        filtered = [word for word in words if word not in stop_words and len(word) > 2]
        all_words.extend(filtered)
    return Counter(all_words).most_common(15)

keywords = extract_keywords(filtered_df['feedback_text'])

# Convert to DataFrame for chart
keywords_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])

# Plot bar chart
keyword_chart = alt.Chart(keywords_df).mark_bar().encode(
    x='Frequency:Q',
    y=alt.Y('Keyword:N', sort='-x'),
    tooltip=['Keyword', 'Frequency']
)

st.altair_chart(keyword_chart, use_container_width=True)

# ─────────────────────────────────────────────
# Sentiment Trend Over Time
st.markdown("### 📈 Sentiment Trend Over Time")
filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'], errors='coerce')
trend_df = filtered_df.dropna(subset=['timestamp'])
trend_daily = trend_df.groupby(trend_df['timestamp'].dt.date)['compound'].mean().reset_index()
trend_daily.columns = ['Date', 'Avg Compound Score']
trend_chart = alt.Chart(trend_daily).mark_line().encode(
    x='Date:T',
    y='Avg Compound Score:Q',
    tooltip=['Date', 'Avg Compound Score']
).properties(width=800)
st.altair_chart(trend_chart, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# Top 5 Apps with Most Negative Feedback
st.markdown("### 🔥 Top 5 Apps by Negative Feedback")
neg_apps = filtered_df[filtered_df['sentiment_label'] == 'Negative']
neg_counts = neg_apps['department'].value_counts().nlargest(5).reset_index()
neg_counts.columns = ['App', 'Negative Count']
neg_chart = alt.Chart(neg_counts).mark_bar().encode(
    x='Negative Count:Q',
    y=alt.Y('App:N', sort='-x'),
    tooltip=['App', 'Negative Count']
).properties(width=800)
st.altair_chart(neg_chart, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# Word Cloud
st.markdown("### ☁️ Word Cloud")
combined_text = ' '.join(filtered_df['feedback_text'].dropna().tolist())
wordcloud = WordCloud(
    background_color='black',
    width=800,
    height=400,
    stopwords=stop_words
).generate(combined_text)
fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

st.markdown("---")

# ─────────────────────────────────────────────
# Heatmap of Sentiment by App
st.markdown("### 🌡️ Heatmap – Average Sentiment by App")
heat_df = filtered_df.groupby('department')['compound'].mean().reset_index()
heat_df = heat_df.sort_values('compound', ascending=False).head(20)
heatmap = alt.Chart(heat_df).mark_rect().encode(
    x=alt.X('compound:Q', axis=alt.Axis(title='Avg Compound Score')),
    y=alt.Y('department:N', sort='-x'),
    color='compound:Q',
    tooltip=['department', 'compound']
).properties(width=800, height=400)
st.altair_chart(heatmap, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# Full Data Table
st.markdown("### 🧾 Feedback Details")
st.dataframe(filtered_df[['department', 'feedback_text', 'sentiment_label', 'compound']])

st.markdown("---")

# ─────────────────────────────────────────────
# Downloading Filtered Data
st.markdown("### 📤 Download Filtered Data")
st.download_button(
    label="Download CSV",
    data=filtered_df.to_csv(index=False),
    file_name='filtered_feedback.csv',
    mime='text/csv'
)

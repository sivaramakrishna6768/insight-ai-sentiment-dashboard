
# InsightAI – Feedback Sentiment Analysis Dashboard

📊 InsightAI is a sentiment analysis and keyword analytics dashboard built with Python, Pandas, Streamlit, and VADER NLP.  
It processes user reviews from the Google Play Store and delivers actionable insights through dynamic visualizations.

---

## 🔍 Features

- 🧠 **Sentiment Analysis** using VADER (Positive / Negative / Neutral)
- 📊 **Keyword Frequency Visualization**
- 📈 **Sentiment Trend Over Time**
- ☁️ **Word Cloud of Review Keywords**
- 🌡️ **Heatmap: Average Sentiment by App**
- 📉 **Top 5 Apps by Negative Feedback**
- 🧾 **Feedback Table with Filtering**
- 📤 **Export Filtered Data as CSV**
- 🎛️ **Interactive Streamlit Dashboard**

---

## 📂 Folder Structure

```
INSIGHT-AI/
├── app/
│   └── streamlit_app.py       # Main Streamlit app
├── data/
│   └── googleplay_feedback.csv  # Feedback dataset
├── src/
│   └── sentiment_analyzer.py  # Standalone sentiment processor
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/insightai-sentiment-dashboard.git
cd insightai-sentiment-dashboard
```

### 2. Set up virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---

## 📦 Dataset Source

Google Play Store User Reviews Dataset  
[https://www.kaggle.com/datasets/lava18/google-play-store-apps](https://www.kaggle.com/datasets/lava18/google-play-store-apps)

> A subset of this dataset (37,000+ rows) was used to build InsightAI. For large-scale analysis, download the full version from Kaggle.

---

## 🧠 Skills Demonstrated

- Natural Language Processing (VADER, NLTK)
- Data Cleaning and Transformation (Pandas)
- Dashboard Design (Streamlit, Altair)
- Insight Extraction and Visualization
- Real-world Dataset Integration

---

## 🙌 Author

Siva Ramakrishna Palaparthy

---

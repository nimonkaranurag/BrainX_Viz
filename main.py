import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import json
from datetime import datetime

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Helper function to safely parse JSON-like strings
def safe_eval(x):
    if pd.isna(x):
        return None
    try:
        return literal_eval(x) if isinstance(x, str) else x
    except:
        return None

# Read the CSV file
df = pd.read_csv('./CTO_dataset.csv')

print("\nDataset Information:")
print("-" * 50)
print(f"Total number of trials: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

# Process the articles data
for i in range(10):
    if str(i) in df.columns:
        df[f'article_{i}'] = df[str(i)].apply(safe_eval)

def extract_article_features(row):
    features = {
        'sources': [],
        'dates': [],
        'sentiment_probs': [],
        'sentiments': []
    }
    
    for i in range(10):
        col = f'article_{i}'
        if col in row and row[col] is not None and isinstance(row[col], dict):
            article = row[col]
            features['sources'].append(article.get('source', ''))
            features['dates'].append(article.get('date', ''))
            features['sentiment_probs'].append(article.get('sentiment_prob', 0))
            features['sentiments'].append(article.get('sentiment', ''))
    
    return pd.Series({
        'n_articles': len(features['sources']),
        'unique_sources': len(set(features['sources'])),
        'avg_sentiment_prob': np.mean(features['sentiment_probs']) if features['sentiment_probs'] else 0,
        'sentiment_std': np.std(features['sentiment_probs']) if len(features['sentiment_probs']) > 1 else 0,
        'sources': features['sources'],
        'dates': features['dates']
    })

# Extract features
article_features = df.apply(extract_article_features, axis=1)
df = pd.concat([df, article_features], axis=1)

print("\nMissing values summary:")
print(df.isnull().sum())

# Create a figure for visualizations
plt.figure(figsize=(20, 25))

# 1. Article Count Distribution
plt.subplot(3, 3, 1)
sns.histplot(data=df, x='n_articles', bins=20)
plt.title('Distribution of Articles per Trial')
plt.xlabel('Number of Articles')
plt.ylabel('Count')

# 2. Source Distribution
plt.subplot(3, 3, 2)
all_sources = [source for sources in df['sources'] if isinstance(sources, list) for source in sources if source]
if all_sources:
    source_counts = pd.Series(all_sources).value_counts().head(10)  # Top 10 sources
    source_counts.plot(kind='bar')
    plt.title('Top 10 News Sources')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Source')
    plt.ylabel('Count')

# 3. Sentiment Probability Distribution
plt.subplot(3, 3, 3)
sns.histplot(data=df, x='avg_sentiment_prob', bins=30)
plt.title('Distribution of Average Sentiment Probabilities')
plt.xlabel('Average Sentiment Probability')
plt.ylabel('Count')

# 4. Articles per Source
plt.subplot(3, 3, 4)
sns.boxplot(data=df, y='unique_sources')
plt.title('Distribution of Unique Sources per Trial')
plt.ylabel('Number of Unique Sources')

# 5. Sentiment Probability vs Number of Articles
plt.subplot(3, 3, 5)
plt.scatter(df['n_articles'], df['avg_sentiment_prob'], alpha=0.5)
plt.title('Sentiment Probability vs Number of Articles')
plt.xlabel('Number of Articles')
plt.ylabel('Average Sentiment Probability')

# 6. Sentiment Standard Deviation
plt.subplot(3, 3, 6)
sns.histplot(data=df, x='sentiment_std', bins=30)
plt.title('Distribution of Sentiment Standard Deviation')
plt.xlabel('Sentiment Standard Deviation')
plt.ylabel('Count')

# 7. Time Series of Articles
plt.subplot(3, 3, 7)
all_dates = [date for dates in df['dates'] if isinstance(dates, list) for date in dates if date]
if all_dates:
    try:
        # Convert dates to datetime and create monthly counts
        dates_series = pd.to_datetime(pd.Series(all_dates))
        date_counts = dates_series.dt.to_period('M').value_counts().sort_index()
        date_counts.plot(kind='line')
        plt.title('Articles Published Over Time (Monthly)')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
    except:
        plt.title('Error Processing Dates')

# 8. Mode Distribution
plt.subplot(3, 3, 8)
df['mode'].value_counts().plot(kind='bar')
plt.title('Distribution of Sentiment Modes')
plt.xlabel('Sentiment Mode')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# 9. Success Rate Analysis
plt.subplot(3, 3, 9)
success_by_sentiment = df.groupby('mode')['lf'].mean()
success_by_sentiment.plot(kind='bar')
plt.title('Success Rate by Sentiment Mode')
plt.xlabel('Sentiment Mode')
plt.ylabel('Success Rate')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()

# Statistical Analysis
print("\nEnhanced Statistical Analysis:")
print("-" * 50)
print(f"Average articles per trial: {df['n_articles'].mean():.2f}")
print(f"Average unique sources per trial: {df['unique_sources'].mean():.2f}")
print(f"Average sentiment probability: {df['avg_sentiment_prob'].mean():.2f}")
print(f"Average sentiment standard deviation: {df['sentiment_std'].mean():.2f}")

# Success Rate Analysis
print("\nSuccess Rate Analysis:")
print("-" * 50)
success_rate = df['lf'].mean() * 100
print(f"Overall success rate: {success_rate:.2f}%")
print("\nSuccess rate by sentiment mode:")
print(df.groupby('mode')['lf'].agg(['mean', 'count', 'std']).round(3))

# Source Analysis
if all_sources:
    print("\nSource Analysis:")
    print("-" * 50)
    print("Top 10 most common sources:")
    print(pd.Series(all_sources).value_counts().head(10))

# Save the figure
plt.savefig('clinical_trials_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete! Visualizations have been saved to 'clinical_trials_analysis.png'")
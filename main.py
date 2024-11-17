import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy import stats

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

# Process the articles data
for i in range(10):
    if str(i) in df.columns:
        df[f'article_{i}'] = df[str(i)].apply(safe_eval)

def extract_article_features(row):
    features = {
        'sources': [],
        'dates': [],
        'sentiment_probs': [],
        'sentiments': [],
        'source_sentiments': []
    }
    
    for i in range(10):
        col = f'article_{i}'
        if col in row and row[col] is not None and isinstance(row[col], dict):
            article = row[col]
            source = article.get('source', '')
            sentiment = article.get('sentiment', '')
            features['sources'].append(source)
            features['dates'].append(article.get('date', ''))
            features['sentiment_probs'].append(article.get('sentiment_prob', 0))
            features['sentiments'].append(sentiment)
            features['source_sentiments'].append((source, sentiment))
    
    # Calculate source-specific sentiment statistics
    source_sentiment_dict = {}
    for source, sentiment in features['source_sentiments']:
        if source:
            if source not in source_sentiment_dict:
                source_sentiment_dict[source] = []
            source_sentiment_dict[source].append(sentiment)
    
    return pd.Series({
        'n_articles': len(features['sources']),
        'unique_sources': len(set(features['sources'])),
        'avg_sentiment_prob': np.mean(features['sentiment_probs']) if features['sentiment_probs'] else 0,
        'sentiment_std': np.std(features['sentiment_probs']) if len(features['sentiment_probs']) > 1 else 0,
        'sources': features['sources'],
        'dates': features['dates'],
        'source_sentiments': source_sentiment_dict
    })

# Extract features
article_features = df.apply(extract_article_features, axis=1)
df = pd.concat([df, article_features], axis=1)

# Create correlation matrix for numerical features
numerical_features = ['n_articles', 'unique_sources', 'avg_sentiment_prob', 'sentiment_std']
correlation_matrix = df[numerical_features].corr()

# Create a figure for visualizations with improved layout
plt.figure(figsize=(25, 30))

# 1. Correlation Heatmap
plt.subplot(4, 3, 1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')

all_sources = [source for sources in df['sources'] if isinstance(sources, list) for source in sources if source]
source_counts = pd.Series(all_sources).value_counts()

# Helper function to prepare hierarchical data
def prepare_source_data(source_counts, min_percentage=1):
    source_data = []
    
    def categorize_source(source):
        if '.com' in source.lower():
            return "Web Portals"
        elif any(journal in source.lower() for journal in ['nature', 'lancet', 'nejm', 'journal']):
            return "Scientific Journals"
        elif any(news in source.lower() for news in ['news', 'times', 'press']):
            return "News Media"
        return "Other Sources"
    
    total_count = sum(source_counts)
    categorized_sources = {}
    
    # Group sources by category
    for source, count in source_counts.items():
        category = categorize_source(source)
        if category not in categorized_sources:
            categorized_sources[category] = []
        categorized_sources[category].append((source, count))
    
    # Process each category
    for category, sources in categorized_sources.items():
        category_total = sum(count for _, count in sources)
        sorted_sources = sorted(sources, key=lambda x: x[1], reverse=True)
        
        # Take top 3 sources
        top_sources = sorted_sources[:3]
        other_sources = sorted_sources[3:]
        
        # Add top sources
        for source, count in top_sources:
            source_data.append({
                'id': source,
                'parent': category,
                'value': count,
                'percentage': (count / category_total) * 100
            })
        
        # Group remaining sources
        if other_sources:
            other_count = sum(count for _, count in other_sources)
            if other_count / category_total * 100 >= min_percentage:
                source_data.append({
                    'id': f'Other {category}',
                    'parent': category,
                    'value': other_count,
                    'percentage': (other_count / category_total) * 100
                })
        
        # Add category
        source_data.append({
            'id': category,
            'parent': 'All Sources',
            'value': category_total,
            'percentage': (category_total / total_count) * 100
        })
    
    # Add root
    source_data.append({
        'id': 'All Sources',
        'parent': '',
        'value': total_count,
        'percentage': 100
    })
    
    return pd.DataFrame(source_data)

# Prepare the data
df_hierarchy = prepare_source_data(source_counts)

# Define a professional blue color palette with more contrasting shades
blue_palette = [
    '#2171b5',  # Main section blue
    '#4292c6',  # First subsection
    '#6baed6',  # Second subsection
    '#9ecae1',  # Third subsection
    '#c6dbef'   # Others subsection
]

# Create enhanced treemap with fixed properties
fig = px.treemap(
    df_hierarchy,
    ids='id',
    names='id',
    parents='parent',
    values='value',
    color='parent',
    color_discrete_sequence=blue_palette,
    custom_data=['percentage'],
    branchvalues='total'  # Moved here from invalid property
)

# Update layout
fig.update_layout(
    title={
        'text': 'Distribution of News Article Sources<br>Used for Labeling Clinical Trials by NCT_ID',
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 28,
            'family': 'Helvetica Neue, Arial',
            'color': '#2171b5'
        }
    },
    width=1200,
    height=800,
    margin=dict(t=100, l=25, r=25, b=25),
    font=dict(
        family="Helvetica Neue, Arial",
        color='#2171b5'
    ),
    uniformtext=dict(minsize=11, mode='hide'),
    paper_bgcolor='#f8f9fa',
    plot_bgcolor='#f8f9fa',
    colorway=blue_palette  # Proper way to set color scheme
)

# Update traces for equal sections and better text display
fig.update_traces(
    hovertemplate="""
    <b style='font-size: 14px'>%{label}</b><br>
    Articles: %{value}<br>
    Distribution: %{customdata[0]:.1f}%
    <extra></extra>
    """,
    textfont=dict(
        family="Helvetica Neue, Arial",
        size=13,
        color='white'
    ),
    marker=dict(
        cornerradius=2,
        line=dict(width=1, color='white')
    ),
    textposition="middle center",
    texttemplate="<span style='font-style: italic'>%{label}</span><br>%{customdata[0]:.1f}%",
    tiling=dict(
        packing='squarify',
        pad=3,
        squarifyratio=1
    ),
    root_color="#f8f9fa"
)

# Save the refined visualization
fig.write_html('source_treemap.html', include_plotlyjs='cdn')


# 3. Combined Sentiment Analysis Plot
plt.subplot(4, 3, 2)
sns.kdeplot(data=df, x='avg_sentiment_prob', hue='mode', common_norm=False)
plt.title('Sentiment Distribution by Mode')
plt.xlabel('Average Sentiment Probability')
plt.ylabel('Density')

# 4. Improved Time Series Analysis
plt.subplot(4, 3, 3)
all_dates = [date for dates in df['dates'] if isinstance(dates, list) for date in dates if date]
if all_dates:
    try:
        # Filter out relative dates and only keep dates in standard format
        valid_dates = []
        for date_str in all_dates:
            try:
                # Try parsing as standard date format
                date = pd.to_datetime(date_str)
                valid_dates.append(date)
            except:
                continue
        
        if valid_dates:
            dates_series = pd.Series(valid_dates)
            date_counts = dates_series.dt.to_period('M').value_counts().sort_index()
            
            # Add trend line
            x = np.arange(len(date_counts))
            z = np.polyfit(x, date_counts.values, 1)
            p = np.poly1d(z)
            
            date_counts.plot(kind='line', alpha=0.7)
            plt.plot(x, p(x), "r--", alpha=0.8, label='Trend')
            plt.title('Articles Published Over Time (Monthly) with Trend')
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Date')
            plt.ylabel('Number of Articles')
            plt.legend()
    except Exception as e:
        print(f"Error in time series analysis: {e}")
        plt.title('Error Processing Dates')

# 5. Sentiment Analysis by Top Sources
plt.subplot(4, 3, 4)
source_sentiment_data = []
for row in df['source_sentiments']:
    if isinstance(row, dict):
        for source, sentiments in row.items():
            for sentiment in sentiments:
                source_sentiment_data.append({'source': source, 'sentiment': sentiment})

if source_sentiment_data:
    source_sentiment_df = pd.DataFrame(source_sentiment_data)
    top_sources = source_sentiment_df['source'].value_counts().head(10).index
    
    pivot_table = pd.crosstab(
        source_sentiment_df[source_sentiment_df['source'].isin(top_sources)]['source'],
        source_sentiment_df[source_sentiment_df['source'].isin(top_sources)]['sentiment'],
        normalize='index'
    )
    
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Sentiment Distribution by Top Sources')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Source')

# 6. Success Rate Analysis with Confidence Intervals
plt.subplot(4, 3, 5)
success_stats = df.groupby('mode')['lf'].agg(['mean', 'count', 'std']).reset_index()
success_stats['sem'] = success_stats['std'] / np.sqrt(success_stats['count'])
success_stats['ci'] = success_stats['sem'] * 1.96  # 95% confidence interval

plt.bar(success_stats['mode'], success_stats['mean'], 
        yerr=success_stats['ci'], 
        capsize=5)
plt.title('Success Rate by Mode with 95% Confidence Intervals')
plt.xlabel('Mode')
plt.ylabel('Success Rate')
plt.xticks(rotation=45, ha='right')

# Add statistical summary
print("\nEnhanced Statistical Analysis:")
print("-" * 50)
print("\nCorrelation Analysis:")
print(correlation_matrix.round(3))

print("\nSource Sentiment Analysis:")
print("-" * 50)
for source in source_counts.head(10).index:
    source_data = source_sentiment_df[source_sentiment_df['source'] == source]
    print(f"\n{source}:")
    print(source_data['sentiment'].value_counts(normalize=True).round(3))

# Save the figure
plt.tight_layout()
plt.savefig('enhanced_clinical_trials_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete! Visualizations have been saved to:")
print("1. enhanced_clinical_trials_analysis.png")
print("2. source_treemap.html (interactive treemap)")
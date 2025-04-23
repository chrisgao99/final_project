import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import f_oneway, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Load data
with open('daily_google_search_counts.json', 'r') as f:
    search_data = json.load(f)
with open('daily_watch_counts.json', 'r') as f:
    youtube_data = json.load(f)

# Create date range
start_date = datetime(2025, 1, 3)
end_date = datetime(2025, 4, 17)
dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
date_strs = [d.strftime('%Y-%m-%d') for d in dates]

# Initialize DataFrame
df = pd.DataFrame({
    'date': date_strs,
    'search_count': [0] * len(dates),
    'youtube_count': [0] * len(dates),
    'day_type': ['Other'] * len(dates)
})

# Populate counts
search_dict = dict(zip(search_data['date'], search_data['number_of_search']))
youtube_dict = dict(zip(youtube_data['date'], youtube_data.get('number_of_watch', youtube_data.get('number_of_search', []))))
for i, date in enumerate(df['date']):
    df.loc[i, 'search_count'] = search_dict.get(date, 0)
    df.loc[i, 'youtube_count'] = youtube_dict.get(date, 0)

# Label day types (0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday)
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.weekday
df.loc[df['weekday'] == 1, 'day_type'] = 'Tuesday'
df.loc[df['weekday'] == 2, 'day_type'] = 'Wednesday'
df.loc[df['weekday'] == 3, 'day_type'] = 'Thursday'

# Normalize counts
epsilon = 1e-6
search_range = df['search_count'].max() - df['search_count'].min()
youtube_range = df['youtube_count'].max() - df['youtube_count'].min()
df['search_count_norm'] = (df['search_count'] - df['search_count'].min()) / (search_range if search_range > 0 else epsilon)
df['youtube_count_norm'] = (df['youtube_count'] - df['youtube_count'].min()) / (youtube_range if youtube_range > 0 else epsilon)

# Add lagged features
df['search_count_norm_lag1'] = df['search_count_norm'].shift(1).fillna(0)
df['youtube_count_norm_lag1'] = df['youtube_count_norm'].shift(1).fillna(0)

# Statistical comparison (ANOVA and t-tests)
groups = ['Tuesday', 'Wednesday', 'Thursday', 'Other']
search_groups = [df[df['day_type'] == g]['search_count'] for g in groups]
youtube_groups = [df[df['day_type'] == g]['youtube_count'] for g in groups]

f_search, p_search_anova = f_oneway(*search_groups)
f_youtube, p_youtube_anova = f_oneway(*youtube_groups)

print("ANOVA Results:")
print(f"Search Count: F = {f_search:.3f}, p = {p_search_anova:.3f}")
print(f"YouTube Count: F = {f_youtube:.3f}, p = {p_youtube_anova:.3f}")

# Pairwise t-tests (Tuesday vs Other, Wednesday vs Other, Thursday vs Other)
other_search = df[df['day_type'] == 'Other']['search_count']
other_youtube = df[df['day_type'] == 'Other']['youtube_count']
print("\nT-Test Results (vs Other):")
for g in ['Tuesday', 'Wednesday', 'Thursday']:
    g_search = df[df['day_type'] == g]['search_count']
    g_youtube = df[df['day_type'] == g]['youtube_count']
    t_search, p_search = ttest_ind(g_search, other_search, equal_var=False)
    t_youtube, p_youtube = ttest_ind(g_youtube, other_youtube, equal_var=False)
    print(f"{g} Search: t = {t_search:.3f}, p = {p_search:.3f} (Mean = {g_search.mean():.2f}, Other Mean = {other_search.mean():.2f})")
    print(f"{g} YouTube: t = {t_youtube:.3f}, p = {p_youtube:.3f} (Mean = {g_youtube.mean():.2f}, Other Mean = {other_youtube.mean():.2f})")

# Visualization 1: Box plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='day_type', y='search_count_norm', data=df, order=groups)
plt.title('Normalized Search Counts by Day Type')
plt.xlabel('Day Type')
plt.subplot(1, 2, 2)
sns.boxplot(x='day_type', y='youtube_count_norm', data=df, order=groups)
plt.title('Normalized YouTube Counts by Day Type')
plt.xlabel('Day Type')
plt.tight_layout()
plt.savefig('meeting_day_box_plots.png')
plt.close()

# Visualization 2: Time series with markers
plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['search_count_norm'], label='Normalized Search Count', color='blue')
plt.plot(df['date'], df['youtube_count_norm'], label='Normalized YouTube Count', color='green')
for g, color, marker in [('Tuesday', 'orange', 'o'), ('Wednesday', 'red', 'x'), ('Thursday', 'purple', '^')]:
    plt.scatter(df[df['day_type'] == g]['date'], df[df['day_type'] == g]['search_count_norm'], 
                color=color, marker=marker, label=g, s=100)
plt.xlabel('Date')
plt.ylabel('Normalized Count')
plt.title('Search and YouTube Counts with Meeting-Related Days')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('meeting_day_time_series.png')
plt.close()

# Machine Learning Model
X = df[['search_count_norm', 'youtube_count_norm', 'search_count_norm_lag1', 'youtube_count_norm_lag1']]
y = df['day_type']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Test model
y_pred = model.predict(X_test)
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred, zero_division=0))

# Plot feature importance
importances = model.feature_importances_
features = ['Search Count', 'YouTube Count', 'Search Count (Lag-1)', 'YouTube Count (Lag-1)']
plt.figure(figsize=(8, 5))
plt.bar(features, importances, color='teal')
plt.title('Feature Importance for Predicting Day Type')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("Plots saved: 'meeting_day_box_plots.png', 'meeting_day_time_series.png', 'feature_importance.png'")
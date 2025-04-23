import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import ttest_ind
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
    'is_wednesday': [0] * len(dates)
})

# Populate counts
search_dict = dict(zip(search_data['date'], search_data['number_of_search']))
youtube_dict = dict(zip(youtube_data['date'], youtube_data.get('number_of_watch', youtube_data.get('number_of_search', []))))
for i, date in enumerate(df['date']):
    df.loc[i, 'search_count'] = search_dict.get(date, 0)
    df.loc[i, 'youtube_count'] = youtube_dict.get(date, 0)

# Label Wednesdays (weekday() == 2 for Wednesday)
df['date'] = pd.to_datetime(df['date'])
df['is_wednesday'] = (df['date'].dt.weekday == 2).astype(int)

# Normalize counts
epsilon = 1e-6
search_range = df['search_count'].max() - df['search_count'].min()
youtube_range = df['youtube_count'].max() - df['youtube_count'].min()
df['search_count_norm'] = (df['search_count'] - df['search_count'].min()) / (search_range if search_range > 0 else epsilon)
df['youtube_count_norm'] = (df['youtube_count'] - df['youtube_count'].min()) / (youtube_range if youtube_range > 0 else epsilon)

# Statistical comparison (t-test)
wed_search = df[df['is_wednesday'] == 1]['search_count']
non_wed_search = df[df['is_wednesday'] == 0]['search_count']
wed_youtube = df[df['is_wednesday'] == 1]['youtube_count']
non_wed_youtube = df[df['is_wednesday'] == 0]['youtube_count']

t_search, p_search = ttest_ind(wed_search, non_wed_search, equal_var=False)
t_youtube, p_youtube = ttest_ind(wed_youtube, non_wed_youtube, equal_var=False)

print("T-Test Results:")
print(f"Search Count: t = {t_search:.3f}, p = {p_search:.3f} (Wed mean = {wed_search.mean():.2f}, Non-Wed mean = {non_wed_search.mean():.2f})")
print(f"YouTube Count: t = {t_youtube:.3f}, p = {p_youtube:.3f} (Wed mean = {wed_youtube.mean():.2f}, Non-Wed mean = {non_wed_youtube.mean():.2f})")

# Visualization 1: Box plots
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='is_wednesday', y='search_count_norm', data=df)
plt.xticks([0, 1], ['Non-Wednesday', 'Wednesday'])
plt.title('Normalized Search Counts by Day Type')
plt.subplot(1, 2, 2)
sns.boxplot(x='is_wednesday', y='youtube_count_norm', data=df)
plt.xticks([0, 1], ['Non-Wednesday', 'Wednesday'])
plt.title('Normalized YouTube Counts by Day Type')
plt.tight_layout()
plt.savefig('wednesday_box_plots.png')
plt.close()

# Visualization 2: Time series with Wednesday markers
plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['search_count_norm'], label='Normalized Search Count', color='blue')
plt.plot(df['date'], df['youtube_count_norm'], label='Normalized YouTube Count', color='green')
plt.scatter(df[df['is_wednesday'] == 1]['date'], df[df['is_wednesday'] == 1]['search_count_norm'], 
            color='red', marker='x', label='Wednesday Meetings', s=100)
plt.xlabel('Date')
plt.ylabel('Normalized Count')
plt.title('Search and YouTube Counts with Wednesday Meetings')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('wednesday_time_series.png')
plt.close()

# Machine Learning Model
X = df[['search_count_norm', 'youtube_count_norm']]
y = df['is_wednesday']

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
features = ['Search Count', 'YouTube Count']
plt.figure(figsize=(8, 5))
plt.bar(features, importances, color='teal')
plt.title('Feature Importance for Predicting Wednesday')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("Plots saved: 'wednesday_box_plots.png', 'wednesday_time_series.png', 'feature_importance.png'")
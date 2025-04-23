import json
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

# Load the data
with open('daily_google_search_counts.json', 'r') as f:
    google_data = json.load(f)

with open('daily_watch_counts.json', 'r') as f:
    youtube_data = json.load(f)

# Create DataFrames
google_df = pd.DataFrame({
    'date': google_data['date'],
    'google_count': google_data['number_of_search']
})

youtube_df = pd.DataFrame({
    'date': youtube_data['date'],
    'youtube_count': youtube_data['number_of_watch']
})

# Merge the DataFrames
merged_df = pd.merge(google_df, youtube_df, on='date', how='inner')

# Add the counts
merged_df['total_count'] = merged_df['google_count'] + merged_df['youtube_count']

# Create the merged JSON output
merged_json = {
    'date': merged_df['date'].tolist(),
    'total_count': merged_df['total_count'].tolist()
}

# Save the merged JSON file
with open('merge_youtube_google_counts.json', 'w') as f:
    json.dump(merged_json, f, indent=2)

# Determine the day of the week for each date
merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df['day_of_week'] = merged_df['date'].dt.day_name()

# Create the two groups: Wednesdays and other days
wednesday_group = merged_df[merged_df['day_of_week'] == 'Wednesday']['total_count']
other_days_group = merged_df[merged_df['day_of_week'] != 'Wednesday']['total_count']

# Calculate statistics
wednesday_mean = wednesday_group.mean()
wednesday_std = wednesday_group.std()
other_days_mean = other_days_group.mean()
other_days_std = other_days_group.std()

# Perform Welch's t-test
t_stat, p_value = stats.ttest_ind(wednesday_group, other_days_group, equal_var=False)

# Print results
print("\nStatistical Analysis: Wednesday vs. Other Days")
print("=" * 50)
print(f"Wednesday Mean: {wednesday_mean:.2f}")
print(f"Wednesday Standard Deviation: {wednesday_std:.2f}")
print(f"Other Days Mean: {other_days_mean:.2f}")
print(f"Other Days Standard Deviation: {other_days_std:.2f}")
print(f"Welch's t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print(f"\nResult: The p-value ({p_value:.4f}) is less than {alpha}, suggesting that")
    print("there is a statistically significant difference between Wednesdays and other days.")
else:
    print(f"\nResult: The p-value ({p_value:.4f}) is greater than {alpha}, suggesting that")
    print("there is no statistically significant difference between Wednesdays and other days.")

# Calculate effect size (Cohen's d)
pooled_std = np.sqrt((wednesday_std**2 + other_days_std**2) / 2)
cohen_d = abs(wednesday_mean - other_days_mean) / pooled_std

print(f"\nEffect size (Cohen's d): {cohen_d:.4f}")
if cohen_d < 0.2:
    print("This represents a small effect size.")
elif cohen_d < 0.8:
    print("This represents a medium effect size.")
else:
    print("This represents a large effect size.")
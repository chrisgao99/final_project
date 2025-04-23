import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

# Determine the day of the week for each date
merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df['day_of_week'] = merged_df['date'].dt.day_name()

# Create a new column for grouped days
def group_days(day):
    if day in ['Tuesday', 'Wednesday', 'Thursday']:
        return day
    else:
        return 'Other Days'

merged_df['day_group'] = merged_df['day_of_week'].apply(group_days)

# Create groups
tuesday_data = merged_df[merged_df['day_group'] == 'Tuesday']['total_count']
wednesday_data = merged_df[merged_df['day_group'] == 'Wednesday']['total_count']
thursday_data = merged_df[merged_df['day_group'] == 'Thursday']['total_count']
other_days_data = merged_df[merged_df['day_group'] == 'Other Days']['total_count']

# Print basic stats for each group
print("\nDescriptive Statistics:")
print("=" * 50)
groups = {
    'Tuesday': tuesday_data,
    'Wednesday': wednesday_data,
    'Thursday': thursday_data,
    'Other Days': other_days_data
}

for name, group in groups.items():
    print(f"{name}: Mean = {group.mean():.2f}, SD = {group.std():.2f}, N = {len(group)}")

# Prepare data for ANOVA
data_for_anova = []
labels_for_anova = []

for day, values in groups.items():
    data_for_anova.extend(values.tolist())
    labels_for_anova.extend([day] * len(values))

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(
    tuesday_data, 
    wednesday_data, 
    thursday_data, 
    other_days_data
)

# Print ANOVA results
print("\nANOVA Results:")
print("=" * 50)
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpret ANOVA results
alpha = 0.05
if p_value < alpha:
    print(f"\nThe p-value ({p_value:.4f}) is less than {alpha}.")
    print("There is a statistically significant difference between at least two groups.")
else:
    print(f"\nThe p-value ({p_value:.4f}) is greater than {alpha}.")
    print("There is no statistically significant difference between the groups.")

# Calculate ANOVA components manually to demonstrate the formula
data_array = np.array([tuesday_data.tolist(), wednesday_data.tolist(), 
                       thursday_data.tolist(), other_days_data.tolist()], dtype=object)

# Function to calculate ANOVA components
def calculate_anova_components(data_array):
    # Calculate overall mean
    all_data = np.concatenate(data_array)
    grand_mean = np.mean(all_data)
    
    # Calculate group means
    group_means = [np.mean(group) for group in data_array]
    group_sizes = [len(group) for group in data_array]
    
    # Calculate SS_between
    ss_between = sum(size * (mean - grand_mean)**2 for size, mean in zip(group_sizes, group_means))
    
    # Calculate SS_within
    ss_within = sum(sum((x - mean)**2 for x in group) for group, mean in zip(data_array, group_means))
    
    # Calculate degrees of freedom
    k = len(data_array)  # number of groups
    n = len(all_data)    # total number of observations
    
    # Calculate Mean Squares
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    
    # Calculate F-statistic
    f_stat = ms_between / ms_within
    
    return {
        'grand_mean': grand_mean,
        'group_means': group_means,
        'group_sizes': group_sizes,
        'ss_between': ss_between,
        'ss_within': ss_within,
        'df_between': k - 1,
        'df_within': n - k,
        'ms_between': ms_between,
        'ms_within': ms_within,
        'f_stat': f_stat
    }

# Calculate and print ANOVA components
anova_results = calculate_anova_components(data_array)
print("\nANOVA Components:")
print("=" * 50)
print(f"Between-group SS: {anova_results['ss_between']:.2f}")
print(f"Within-group SS: {anova_results['ss_within']:.2f}")
print(f"Between-group df: {anova_results['df_between']}")
print(f"Within-group df: {anova_results['df_within']}")
print(f"Between-group MS: {anova_results['ms_between']:.2f}")
print(f"Within-group MS: {anova_results['ms_within']:.2f}")
print(f"F = (Between-group MS) / (Within-group MS) = {anova_results['f_stat']:.4f}")

# Perform pairwise t-tests if ANOVA is significant
print("\nPairwise t-tests (Bonferroni-corrected):")
print("=" * 50)

# Number of comparisons for Bonferroni correction
n_comparisons = 6  # (4 choose 2) = 6 pairs

# List of all pairs to compare
pairs = [
    ('Tuesday', 'Wednesday'),
    ('Tuesday', 'Thursday'),
    ('Tuesday', 'Other Days'),
    ('Wednesday', 'Thursday'),
    ('Wednesday', 'Other Days'),
    ('Thursday', 'Other Days')
]

for day1, day2 in pairs:
    t_stat, p_uncorrected = stats.ttest_ind(
        groups[day1], 
        groups[day2], 
        equal_var=False  # Welch's t-test
    )
    p_corrected = min(p_uncorrected * n_comparisons, 1.0)  # Bonferroni correction
    
    print(f"{day1} vs {day2}:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value (uncorrected): {p_uncorrected:.4f}")
    print(f"  p-value (Bonferroni-corrected): {p_corrected:.4f}")
    if p_corrected < alpha:
        print(f"  Significant difference detected (p < {alpha})")
    else:
        print(f"  No significant difference (p > {alpha})")
    print()

# Create a box plot to visualize the differences
plt.figure(figsize=(10, 6))
sns.boxplot(x=pd.Series(labels_for_anova), y=pd.Series(data_for_anova))
plt.title('Comparison of Total Counts by Day Group')
plt.xlabel('Day Group')
plt.ylabel('Total Count')
plt.savefig('day_group_comparison.png')
plt.close()

# Also perform Tukey's HSD test for multiple comparisons
tukey_results = pairwise_tukeyhsd(
    pd.Series(data_for_anova), 
    pd.Series(labels_for_anova),
    alpha=0.05
)

print("\nTukey's HSD Test Results:")
print("=" * 50)
print(tukey_results)

# Print a summary interpretation
print("\nSummary Interpretation:")
print("=" * 50)
if p_value < alpha:
    significant_pairs = [(pair[0], pair[1]) for pair, p_corr in 
                        zip(pairs, [min(p * n_comparisons, 1.0) for _, p in 
                                   [stats.ttest_ind(groups[d1], groups[d2], equal_var=False) 
                                    for d1, d2 in pairs]]) 
                        if p_corr < alpha]
    
    if significant_pairs:
        print(f"The data shows statistically significant differences between the following groups:")
        for day1, day2 in significant_pairs:
            mean1 = groups[day1].mean()
            mean2 = groups[day2].mean()
            print(f"  - {day1} (mean: {mean1:.2f}) vs {day2} (mean: {mean2:.2f})")
    else:
        print("While the overall ANOVA is significant, the pairwise comparisons with")
        print("Bonferroni correction do not show statistically significant differences.")
else:
    print("The data does not show statistically significant differences between")
    print("Tuesday, Wednesday, Thursday, and other days (p = {:.4f}).".format(p_value))
    print("This suggests that the day of the week does not significantly impact")
    print("the total counts in the data.")
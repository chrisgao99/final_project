import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Load the daily_search_counts.json file
with open('daily_search_counts.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dates = data["date"]
search_counts = data["number_of_search"]

# Convert dates to datetime objects for plotting
date_objects = [datetime.strptime(date, "%Y-%m-%d") for date in dates]

# 1. Plot time series of search counts
plt.figure(figsize=(12, 6))
plt.plot(date_objects, search_counts, label="Daily search Counts", color='blue')
plt.xlabel("Date")
plt.ylabel("Number of searches")
plt.title("YouTube search Counts Over Time (2024-12-30 to 2025-03-30)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("search_counts_time_series.png")
plt.close()

# 2. Calculate average search counts by day of the week
# 2024-12-30 is Monday, so we can map days accordingly
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_counts = {day: [] for day in day_names}

for date, count in zip(date_objects, search_counts):
    # weekday() returns 0 for Monday, 6 for Sunday
    day_index = date.weekday()
    day_name = day_names[day_index]
    day_counts[day_name].append(count)

# Compute averages
day_averages = {day: np.mean(counts) if counts else 0 for day, counts in day_counts.items()}

# Print averages
print("Average search Counts by Day of the Week:")
for day, avg in day_averages.items():
    print(f"{day}: {avg:.2f}")

# 3. Plot averages by day of the week
plt.figure(figsize=(8, 5))
plt.bar(day_names, [day_averages[day] for day in day_names], color='green')
plt.xlabel("Day of the Week")
plt.ylabel("Average Number of searches")
plt.title("Average YouTube search Counts by Day of the Week")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("search_counts_by_day.png")
plt.close()

print("Plots saved as 'search_counts_time_series.png' and 'search_counts_by_day.png'")
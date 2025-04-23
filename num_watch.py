import json
from datetime import datetime, timedelta

# Define the date range
start_date = datetime(2024, 12, 30)
end_date = datetime(2025, 3, 30)

# Initialize a dictionary to store daily counts
daily_counts = {}

# Populate the dictionary with all dates in the range, starting with 0 counts
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")
    daily_counts[date_str] = 0
    current_date += timedelta(days=1)

# Read the watch-history.json file
with open('watch-history.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Process each entry and increment the count for the corresponding date
for entry in data:
    timestamp = entry["time"]
    # Try parsing with microseconds first, then without if it fails
    try:
        date = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").date()
    except ValueError:
        date = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").date()
    
    date_str = date.strftime("%Y-%m-%d")
    
    # Only count if the date is within the specified range
    if start_date.date() <= date <= end_date.date():
        daily_counts[date_str] += 1

# Prepare the output data
output_data = {
    "date": list(daily_counts.keys()),
    "number_of_watch": list(daily_counts.values())
}

# Write to output JSON file
with open('daily_watch_counts.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4)

print("Output written to 'daily_watch_counts.json'")
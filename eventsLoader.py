from API import get_events
import json

data = {}

for year in range(2006, 2026):
    events = get_events(year)['Events']
    print("getting events from: ", year)
    matches = get_events(year)['Events']
    data[year] = matches

with open('events.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
from API import get_events
from datetime import date
import json

with open('events.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for year in range(2006, date.today().year+1):
    if data.get(str(year)) == None:
        events = get_events(year)['Events']
        print("getting events for: ", year)
        matches = get_events(year)['Events']
        data[year] = matches

with open('events.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
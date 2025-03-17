from API import get_matches
from datetime import date
import json

events = json.load(open('events.json'))
data = {}
year = date.today().year
for event in events[str(year)]:
        code = event['code']
        try:
            matches = get_matches(year, code)['Matches']
            print("getting matches from: ", year, " ", code)
        except:
            print("failed to get matches from: ", year, " ", code)
        data[code] = matches
with open('matches/', str(year) + '.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
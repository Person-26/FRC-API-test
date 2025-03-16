from API import get_matches
import json

events = json.load(open('events.json'))

for year in range(int(min(events)), int(max(events))+1):
    data = {}
    for event in events[str(year)]:
        code = event['code']
        try:
            matches = get_matches(year, code)['Matches']
            print("getting matches from: ", year, " ", code)
        except:
            print("failed to get matches from: ", year, " ", code)
        data[code] = matches
    with open(str(year) + 'matches.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
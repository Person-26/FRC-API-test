from API import get_qual_matches,get_playoff_matches, get_events
import json

for year in range(2017, 2026):
    with open('matches/' + str(year) + '.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for event in get_events(year)['Events']:
            code = event['code']
            try:
                matches = get_qual_matches(year, code)['Matches'] + get_playoff_matches(year, code)['Matches']
                print("getting matches from: ", year, " ", code)
            except:
                print("failed to get matches from: ", year, " ", code)
            data[code] = matches
    with open('matches/' + str(year) + '.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
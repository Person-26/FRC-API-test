import requests
import base64
import json 

def get_matches(year, code):
    url = f'https://frc-api.firstinspires.org/v3.0/{year}/matches/{code}?tournamentLevel=Qualification'
    auth_string = f"{'Derek'}:{'5464977c-e731-448f-9938-878e45327dc4'}"
    encoded_auth_string = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
    headers = {
        'Authorization': f'Basic {encoded_auth_string}',
        'If-Modified-Since': ''
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}")
    data = response.json()
    return data

events = json.load(open('events.json'))['Events']

data = {}
for event in events:
    code = event['code']
    print("getting matches from: ", code)
    matches = get_matches(2025, code)['Matches']
    data[code] = matches

with open('matches.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

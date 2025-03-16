import requests
import base64
import json 

auth_string = f"{'Derek'}:{'5464977c-e731-448f-9938-878e45327dc4'}"
encoded_auth_string = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

def get_events(year):
    url = f'https://frc-api.firstinspires.org/v3.0/{year}/events'
    headers = {
        'Authorization': f'Basic {encoded_auth_string}',
        'If-Modified-Since': ''
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}")
    return response.json()

def get_matches(year, code):
    url = f'https://frc-api.firstinspires.org/v3.0/{year}/matches/{code}?tournamentLevel=Qualification'
    headers = {
        'Authorization': f'Basic {encoded_auth_string}',
        'If-Modified-Since': ''
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}")
    return response.json()

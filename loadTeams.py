import json

# Load match data from matches/2024.json
with open('matches/2024.json', 'r') as match_file:
    matches = json.load(match_file)

# Set to store unique team numbers
unique_teams = set()

# Loop through matches and extract team numbers
for event in matches.values():
    for match in event:
        for team in match['teams']:
            unique_teams.add(team['teamNumber'])

# Save unique team numbers to 2024Teams.json
with open('2024Teams.json', 'w') as teams_file:
    json.dump(sorted([team for team in unique_teams]), teams_file, indent=4)
from torch.utils.data import TensorDataset
from pandas import DataFrame
import json
import torch
from sklearn.preprocessing import LabelEncoder

# Initialize encoders
event_encoder = LabelEncoder()
team_encoder = LabelEncoder()

# Prepare data structures
dataFrame = DataFrame([], columns=['year', 'event', 'match', 'red1', 'red2', 'red3', 'blue1', 'blue2', 'blue3'], dtype='uint16')
labelFrame = DataFrame([], columns=['redFinal', 'redFoul', 'redAuto', 'blueFinal', 'blueFoul', 'blueAuto'])

# Collect all unique events and teams for encoding
all_events = []
all_teams = []

for year in range(2006, 2025):
    events = json.load(open('matches/' + str(year) + '.json'))
    for eventName, matches in events.items():
        all_events.append(eventName)
        for match in matches:
            all_teams.extend([t['teamNumber'] for t in match['teams']])

# Fit encoders
event_encoder.fit(all_events)
team_encoder.fit(all_teams)

# Process data
for year in range(2006, 2025):
    events = json.load(open('matches/' + str(year) + '.json'))
    for eventName, matches in events.items():
        print("loading", eventName, " in ", year)
        for match in matches:
            encoded_event = event_encoder.transform([eventName])[0]
            encoded_teams = team_encoder.transform([t['teamNumber'] for t in match['teams']])
            dataFrame.loc[len(dataFrame)] = [year, encoded_event, int(match['matchNumber'])] + list(encoded_teams)
            labelFrame.loc[len(labelFrame)] = [match['scoreRedFinal'], match['scoreRedFoul'], match['scoreRedAuto'], match['scoreBlueFinal'], match['scoreBlueFoul'], match['scoreBlueAuto']]

if __name__ == "__main__":
    data = torch.tensor(dataFrame.values, dtype=torch.uint16)
    labels = torch.tensor(labelFrame.values, dtype=torch.uint16)

    dataSet = TensorDataset(data, labels)

    torch.save(dataSet, 'dataSet.pt')
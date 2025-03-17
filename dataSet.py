from torch.utils.data import Dataset
from pandas import DataFrame
import json

class MatchDataSet(Dataset):
    def __init__(self):
        """
        remember to split yearly matches into each event
        Args:
            data (list or numpy array): The input data (e.g., vectors).
            labels (list or numpy array): The corresponding labels for the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        year = 2024
        events = json.load(open('matches\{year}.json'))

        self.data = DataFrame([], columns=['year', 'event', 'match', 'red1', 'red2', 'red3', 'blue1', 'blue2', 'blue3'])
        self.labels = DataFrame([], columns=['redFinal', 'redFoul', 'redAuto', 'blueFinal', 'blueFoul', 'blueAuto'])

        for eventName, matches in events:
            for match in matches:
                self.data.loc[-1] = [year, eventName, match['matchNumber']] + [t['teamNumber'] for t in match['teams']]
                self.labels.loc[-1] = [matches['scoreRedFinal'], matches['scoreRedFoul'], matches['scoreRedAuto'], matches['scoreBlueFinal'], matches['scoreBlueFoul'], matches['scoreBlueAuto']]
        self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'match': self.data[idx], 'score': self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample
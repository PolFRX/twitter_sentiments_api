import os
import pandas as pd


DATASET_PATH = 'dataset/french_tweets.csv'


def get_max_length():
    tweets = pd.read_csv(DATASET_PATH)['text']
    max_length = 0

    for tweet in tweets:
        if len(tweet) > max_length:
            max_length = len(tweet)

    print(f'Max Length: {max_length}')
    return max_length


get_max_length()

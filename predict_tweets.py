from get_tweets import get_tweets
from model import predict
import numpy as np


MODEL_NAME = 'sentiments_bert'


subject = input('Entrez le sujet des tweets: ')

print(f'- Start process')
tweets = get_tweets(subject)
predictions = predict(tweets, MODEL_NAME)

positives = 0
negatives = 0

print(f'- Start computing the score')
for prediction in predictions:
    pred = np.argmax(prediction)
    if pred == 0:
        negatives += 1
    else:
        positives += 1

score = (((positives - negatives) / len(predictions)) + 1) * 50
print(f'\nLe score de positivité de "{subject}" est de {score}/100')

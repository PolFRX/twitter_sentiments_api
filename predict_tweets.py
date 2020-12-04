from get_tweets import get_tweets
from model import predict
import numpy as np
from matplotlib import pyplot as plt


MODEL_NAME = 'sentiments_bert'


subject = input('Entrez le sujet des tweets: ')

print(f'- Start process')
tweets = get_tweets(subject)
predictions = predict(tweets, MODEL_NAME)

positives = 0
negatives = 0
neutrals = 0

print(f'- Start computing the score')
for prediction in predictions:
    pred = np.argmax(prediction)
    if -0.2 < prediction[0] < 0.2 and -0.2 < prediction[1] < 0.2:
        neutrals += 1
    elif pred == 0:
        negatives += 1
    else:
        positives += 1

score = (((positives - negatives) / len(predictions)) + 1) * 50
print(f'\nLe score de positivité de "{subject}" est de {"{:.2f}".format(score)}/100')

labels = ['positives', 'negatives', 'neutrals']
sizes = [positives, negatives, neutrals]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.pie(sizes, labels=labels, autopct='%1.2f%%')
plt.show()


from flask import Flask, jsonify, request
from get_tweets import get_tweets
from model import predict
import numpy as np


MODEL_NAME = 'sentiments_bert'
TOKEN = ''


app = Flask(__name__)


@app.route("/", methods=['GET'])
def get_score():
    if 'token' in request.args:
        token = request.args['token']
    else:
        return error('Unauthorized access')
    if 'subject' in request.args:
        subject = request.args['subject']
    else:
        return error('You need to specify the subject')

    if token == TOKEN:
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
        data = {}
        score = (((positives - negatives) / len(predictions)) + 1) * 50
        data['score'] = "{:.2f}".format(score)

        return jsonify({
            'status': 'ok',
            'data': data
        })
    else:
        return error('Unauthorized access')


def error(msg):
    return jsonify({
        'status': 'error',
        'message': msg}), 500


if __name__ == "__main__":
    app.run(debug=True)

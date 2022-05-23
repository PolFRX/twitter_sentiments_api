from transformers import CamembertTokenizer as Tokenizer, TFCamembertForSequenceClassification as Camembert
import os
from numpy import int32, zeros
from loguru import logger


MAX_LENGTH = 460


def _encode_tweets(tweets, max_length=MAX_LENGTH):
    tokenizer = Tokenizer.from_pretrained("camembert-base")
    encoded = zeros(shape=(len(tweets), max_length), dtype=int32)

    for i, tweet in enumerate(tweets):
        tweet_encoded = tokenizer.encode(tweet)
        encoded[i][:len(tweet_encoded)] = tweet_encoded

    return encoded


def _load_model(name):
    path = os.getcwd() + "/models/" + name + ".h5"
    model = Camembert.from_pretrained('jplu/tf-camembert-base', num_labels=2)
    model.load_weights(path)

    return model


def predict(tweets, model_name):
    model = _load_model(model_name)
    logger.info(f'Model "{model_name}" loaded')
    data = _encode_tweets(tweets, MAX_LENGTH)
    attention_mask = (data != 0).astype(int32)
    data = {"input_ids": data, "attention_mask": attention_mask}
    logger.info(f'Tweets encoded')
    logger.info(f'Predictions in process on {len(tweets)} tweets...')
    predictions = model.predict(data)['logits']

    return predictions

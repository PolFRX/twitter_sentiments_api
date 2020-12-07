from transformers import CamembertTokenizer as Tokenizer, TFCamembertForSequenceClassification as Camembert
import os
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy as SCC


DATASET_PATH = 'dataset/french_tweets.csv'
TRAIN_DATA_PATH = 'dataset/train_data.npy'
TRAIN_LABEL_PATH = 'dataset/train_label.npy'
TEST_DATA_PATH = 'dataset/test_data.npy'
TEST_LABEL_PATH = 'dataset/test_label.npy'
MAX_LENGTH = 460


def get_dataset():
    labels_train = np.load(TRAIN_LABEL_PATH)[:60000]
    tweets_train = np.load(TRAIN_DATA_PATH)[:60000]
    labels_test = np.load(TEST_LABEL_PATH)[:5000]
    tweets_test = np.load(TEST_DATA_PATH)[:5000]

    attention_mask_train = (tweets_train != 0).astype(np.int32)
    attention_mask_test = (tweets_test != 0).astype(np.int32)

    tweets_train = {"input_ids": tweets_train, "attention_mask": attention_mask_train}
    tweets_test = {"input_ids": tweets_test, "attention_mask": attention_mask_test}

    return labels_train, tweets_train, labels_test, tweets_test


def preprocess_dataset():
    df = pd.read_csv(DATASET_PATH)
    df = df.sample(frac=1).reset_index(drop=True)

    train = df.sample(frac=0.9, random_state=200)  # random state is a seed value
    test = df.drop(train.index)
    df = None

    tweets_train = encode_tweets(train['text'])
    tweets_test = encode_tweets(test['text'])
    labels_train = train['label'].to_numpy(dtype=np.uint8)
    labels_test = test['label'].to_numpy(dtype=np.uint8)

    np.save(TRAIN_DATA_PATH, tweets_train)
    np.save(TRAIN_LABEL_PATH, labels_train)
    np.save(TEST_DATA_PATH, tweets_test)
    np.save(TEST_LABEL_PATH, labels_test)

    attention_mask_train = (tweets_train != 0).astype(np.int32)
    attention_mask_test = (tweets_test != 0).astype(np.int32)

    tweets_train = {"input_ids": tweets_train, "attention_mask": attention_mask_train}
    tweets_test = {"input_ids": tweets_test, "attention_mask": attention_mask_test}

    return labels_train, tweets_train, labels_test, tweets_test


def encode_tweets(tweets, max_length=MAX_LENGTH):
    tokenizer = Tokenizer.from_pretrained("camembert-base")
    encoded = np.zeros(shape=(len(tweets), max_length), dtype=np.int32)

    for i, tweet in enumerate(tweets):
        tweet_encoded = tokenizer.encode(tweet)
        encoded[i][:len(tweet_encoded)] = tweet_encoded

    return encoded


def train(data, labels, epochs=3, batch_size=3, model_name='sentiments_bert'):
    model = Camembert.from_pretrained('jplu/tf-camembert-base', num_labels=2)

    learning_rate = 2e-5
    loss_fn = SCC(from_logits=True)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, epsilon=1e-8),
        loss=loss_fn,
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        data, labels, epochs=epochs, batch_size=batch_size,
        validation_split=0.2, verbose=1
    )

    save_model(model, model_name)

    return model


def save_model(model, name):
    """ Save a given model with a particular name.

         Args:
             model: the model to save
             name: the name to use to save the model
         """
    path = os.getcwd()
    path = path + "/models/" + name + ".h5"
    model.save_weights(path)


def test(data, labels, model):
    predictions = model.predict(data)

    good_pred = 0
    for prediction, label in zip(predictions['logits'], labels):
        if np.argmax(prediction) == label:
            good_pred += 1

    res = (good_pred/len(predictions['logits'])) * 100
    print(f'Prediction on test set: {res}')


def load_model(name):
    path = os.getcwd() + "/models/" + name + ".h5"
    model = Camembert.from_pretrained('jplu/tf-camembert-base', num_labels=2)
    model.load_weights(path)

    return model


def model():
    if os.path.isfile(TRAIN_DATA_PATH):
        training_label, training_set, test_label, test_set = get_dataset()
    else:
        training_label, training_set, test_label, test_set = preprocess_dataset()

    model = train(training_set, training_label)
    # model = load_model('sentiments_bert')
    test(test_set, test_label, model)


def predict(tweets, model_name):
    print(f'-- Start predicting tweets')
    model = load_model(model_name)
    print(f'--- Model loaded')
    data = encode_tweets(tweets, MAX_LENGTH)
    attention_mask = (data != 0).astype(np.int32)
    data = {"input_ids": data, "attention_mask": attention_mask}
    print(f'--- Tweets encoded')
    print(f'-- Predictions in process on {len(tweets)} tweets...')
    predictions = model.predict(data)['logits']
    print(f'\t DONE')

    return predictions

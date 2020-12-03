import requests as req
from requests_oauthlib import OAuth1, OAuth2
import requests
import json
import re


def nlp_pipeline(text):
    text = re.sub(r"http\S+", "", text)
    text = remove_emoji(text)
    text = re.sub(r"@(\w+)", ' ', text, flags=re.MULTILINE)
    text = text.replace('RT', '')
    text = text.replace('#', '')
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split())
    text = re.sub(r"[A-Za-z\.]*[0-9]+[A-Za-z%°\.]*", "", text)
    text = re.sub(r"(\s\-\s|-$)", "", text)
    text = re.sub(r"[,\!\?\%\(\)\/\"]", "", text)
    text = re.sub(r"\&\S*\s", "", text)
    text = re.sub(r"\&", "", text)
    text = re.sub(r"\+", "", text)
    text = re.sub(r"\#", "", text)
    text = re.sub(r"\$", "", text)
    text = re.sub(r"\£", "", text)
    text = re.sub(r"\%", "", text)
    text = re.sub(r"\:", "", text)
    text = re.sub(r"\@", "", text)
    text = re.sub(r"\-", "", text)

    return text


def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def get_tweets(subject):
    print(f'-- Start retrieving tweets')
    with open('credentials.json') as json_file:
        data = json.load(json_file)
        api_key = data['api_key']
        api_secret = data['api_secret']
        user_token = data['user_token']
        user_secret = data['user_secret']
        bearer_token = data['bearer_token']

    # auth = OAuth1(api_key, api_secret, user_token, user_secret)
    token = {'access_token': bearer_token, 'token_type': 'bearer'}
    auth = OAuth2(token=token)


    if '#' in subject:
        subject = subject.replace('#', '%23')

    url = f'https://api.twitter.com/2/tweets/search/recent?query={subject}+lang:fr+-is:retweet&max_results=100'
    r = req.get(url, auth=auth)

    txt = json.loads(r.text)
    data = txt["data"]

    for _ in range(10):
        next = txt["meta"]["next_token"]
        url = f'https://api.twitter.com/2/tweets/search/recent?query={subject}+lang:fr+-is:retweet&max_results=100' \
              f'&next_token={next}'
        r = req.get(url, auth=auth)
        txt = json.loads(r.text)
        data += txt["data"]

    dataset = []
    textt = []
    for tweet in data:
        textt.append(tweet["text"])
        txt = nlp_pipeline(tweet["text"])
        dataset.append(txt)

    print(f'--- Get {len(dataset)} tweets')

    return dataset


get_tweets('#chasse')

import requests as req
from requests_oauthlib import OAuth1, OAuth2
import requests
import json
import re
import demoji


def nlp_pipeline(text):
    text = re.sub(r"http\S+", "", text)
    text = demoji.replace(text, '')
    text = re.sub(r"@(\w+)", ' ', text, flags=re.MULTILINE)
    text = text.replace('#', '')
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split())
    # text = re.sub(r"[A-Za-z\.]*[0-9]+[A-Za-z%°\.]*", "", text)
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

    return text


def get_tweets(subject):
    print(f'-- Start retrieving tweets')
    with open('credentials.json') as json_file:
        data = json.load(json_file)
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
        if "next_token" not in txt["meta"]:
            break
        next_token = txt["meta"]["next_token"]
        url = f'https://api.twitter.com/2/tweets/search/recent?query={subject}+lang:fr+-is:retweet&max_results=100' \
              f'&next_token={next_token}'
        r = req.get(url, auth=auth)
        txt = json.loads(r.text)
        data += txt["data"]

    dataset = []
    textt = []
    demoji.download_codes()
    for tweet in data:
        textt.append(tweet["text"])
        txt = nlp_pipeline(tweet["text"])
        dataset.append(txt)

    print(f'--- Get {len(dataset)} tweets')

    return dataset


# get_tweets('#chasse')

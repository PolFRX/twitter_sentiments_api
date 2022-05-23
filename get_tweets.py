import requests as req
from requests_oauthlib import OAuth2
import json
import re
import demoji
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
from loguru import logger


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
    logger.info('-- Start retrieving tweets')
    with open('credentials.json') as json_file:
        data = json.load(json_file)
        bearer_token = data['bearer_token']

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

    #word cloud
    stop_words = set(STOPWORDS)
    with open('stop_words_french.json', encoding='utf-8') as json_file:
        stop_words_french = json.load(json_file)
    stop_words.update(stop_words_french)
    stop_words.add(subject.replace('%23', ''))
    stop_words.add(subject.replace('%23', '').lower())
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stop_words,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1  # chosen at random by flipping a coin; it was heads
    ).generate(str(dataset))

    plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()

    logger.info(f'--- Get {len(dataset)} tweets')

    return dataset

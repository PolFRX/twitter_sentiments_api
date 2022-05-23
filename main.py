from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from numpy import argmax
from loguru import logger

from config import MODEL_NAME
from helpers.model_helper import predict
from helpers.security_helper import api_key_auth
from helpers.twitter_helper import create_word_cloud, get_tweets

import traceback

app = FastAPI()


@app.get("/predict_tweet", dependencies=[Depends(api_key_auth)])
def predict_tweet(subject: str) -> JSONResponse:
    try:
        tweets = get_tweets(subject)
        predictions = predict(tweets, MODEL_NAME)

        positives = 0
        negatives = 0
        neutrals = 0

        logger.info(f'Start computing the score')
        for prediction in predictions:
            pred = argmax(prediction)
            if -0.2 < prediction[0] < 0.2 and -0.2 < prediction[1] < 0.2:
                neutrals += 1
            elif pred == 0:
                negatives += 1
            else:
                positives += 1
        data = {}
        score = (((positives - negatives) / len(predictions)) + 1) * 50
        data['score'] = "{:.2f}".format(score)

        response = {
            "status_code": status.HTTP_200_OK,
            "score": "{:.2f}".format(score),
        }
        logger.info(f"Response: {response}")

        return JSONResponse(content=response)
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred"
        )


@app.get("/word_cloud", dependencies=[Depends(api_key_auth)])
def get_word_cloud(subject: str) -> StreamingResponse:
    try:
        tweets = get_tweets(subject)
        img_word_cloud_stream = create_word_cloud(tweets, subject)
        return StreamingResponse(content=img_word_cloud_stream, media_type="image/png")
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred"
        )

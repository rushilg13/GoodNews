# Import Modules
from fastapi import FastAPI
import pickle
from starlette.responses import Response
import uvicorn
import requests

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('tranform.pkl', 'rb'))

app = FastAPI()

@app.get('/')
def home():
    Response = news()
    return Response["articles"]

@app.post('/')
def news():
    response = requests.get("https://newsapi.org/v2/top-headlines?country=in&apiKey=7bf43f1b86894a14990f6ddeb0b32dac")
    response = response.json()
    for article in response["articles"]:
        headline = article["title"]
        positive_headline = predict_news(headline)
        if int(positive_headline) == 1:
            article["sentiment"] = "positive"
        else:
            article["sentiment"] = "negative"
    return response

def predict_news(data:str):
    msg = [data]
    vect = cv.transform(msg).toarray()
    my_prediction = clf.predict(vect)
    return my_prediction
# Import Modules
from fastapi import FastAPI
import pickle
from starlette.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('tranform.pkl', 'rb'))

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
    "https://goodnewsapi.herokuapp.com/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/{category}/{location}')
def home(category:str, location:str):
    Response = news(category, location)
    return Response["articles"]

@app.post('/')
def news(category, location):
    URL = "https://saurav.tech/NewsAPI/top-headlines/category/{0}/{1}.json".format(category, location)
    response = requests.get(URL)
    # response = requests.get("https://newsapi.org/v2/top-headlines?country=in&apiKey=7bf43f1b86894a14990f6ddeb0b32dac")
    response = response.json()
    for article in response["articles"]:
        headline = article["title"]
        positive_headline = predict_news(headline)
        if int(positive_headline) == 1:
            article["sentiment"] = "positive"
        else:
            article["sentiment"] = "negative"
    print("Prediction done")
    return response

def predict_news(data:str):
    msg = [data]
    vect = cv.transform(msg).toarray()
    my_prediction = clf.predict(vect)
    return my_prediction
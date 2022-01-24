# Import Modules
from fastapi import FastAPI
import pickle
from starlette.responses import Response
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import uvicorn
import requests
from fastapi.responses import FileResponse
import pymongo
import os

# Load Pickle files
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('tranform.pkl', 'rb'))

# Set Middleware
middleware = [ Middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])]

# Set up app
app = FastAPI(middleware=middleware)

# connect to MongoDB
MONGODB_URI = os.environ.get('MONGODB_URI')
myclient = pymongo.MongoClient(MONGODB_URI)
mydb = myclient["GoodNews"]

# Routes
@app.get('/{category}/{location}')
def home(category:str, location:str):
    Response = news(category, location)
    return Response

def news(category, location):
    # For getting all the news articles
    if (category == "all" and location == "all"):
        counter = 0
        all_articles = []
        cats = ['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology']
        locs = ["in", "us", "gb", "fr", "au", "ru"]
        for cat in (cats):
            for loc in (locs):
                URL = "https://saurav.tech/NewsAPI/top-headlines/category/{0}/{1}.json".format(cat, loc)
                response = requests.get(URL)
                response = response.json()
                for article in response["articles"]:
                    headline = article["title"]
                    positive_headline = predict_news(headline)
                    if int(positive_headline) == 1:
                        all_articles.append(article)
                        article["sentiment"] = "positive"
                        article["article_ID"] = counter
                        counter += 1
                    else:
                        article["sentiment"] = "negative"
                        article.clear()
        print("Prediction done")
        return all_articles
    
    # For particular category and location
    else:
        counter = 0
        all_articles = []
        URL = "https://saurav.tech/NewsAPI/top-headlines/category/{0}/{1}.json".format(category, location)
        response = requests.get(URL)
        # response = requests.get("https://newsapi.org/v2/top-headlines?country=in&apiKey=7bf43f1b86894a14990f6ddeb0b32dac")
        response = response.json()
        for article in response["articles"]:
            headline = article["title"]
            positive_headline = predict_news(headline)
            if int(positive_headline) == 1:
                all_articles.append(article)
                article["sentiment"] = "positive"
                article["article_ID"] = counter
                counter += 1
            else:
                article["sentiment"] = "negative"
                article.clear()
    print("Prediction done")
    return all_articles

# Prediction Function
def predict_news(data:str):
    msg = [data]
    vect = cv.transform(msg).toarray()
    my_prediction = clf.predict(vect)
    return my_prediction

# View Wrongly Predicted Records
@app.get("/records")
def records():
    return FileResponse("a.txt")

# Save Reported News Headlines to MongoDB
@app.post("/save")
def save(title: str):
    mycol = mydb["Reported_Titles"]
    rec = {"title":title}
    x = mycol.insert_one(rec)
    print("Document inserted with id: ", x.inserted_id)
    return {"Document inserted" : True}

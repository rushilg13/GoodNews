# Import Modules
from fastapi import FastAPI
import pickle
from starlette.responses import Response
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
from fastapi.responses import FileResponse
import pymongo  
import json
from bson import json_util
from datetime import datetime

# Load Pickle files
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('tranform.pkl', 'rb'))

# Set Middleware
middleware = [ Middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])]

# Set up app
app = FastAPI(middleware=middleware)

# connect to MongoDB
myclient = pymongo.MongoClient("mongodb+srv://VIT_Admin:pizza@vitdiaries.tpuku.mongodb.net/GoodNews?retryWrites=true&w=majority")
mydb = myclient["GoodNews"]

# Routes
@app.get('/{category}/{location}')
async def home(category:str, location:str):
    news(category, location)
    data = []
    if category == "all" and location == "all":
        cats = ['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology']
        locs = ["in", "us", "gb", "fr", "au", "ru"]
        for cat in (cats):
            for loc in (locs):
                col_name = "{0}_{1}".format(cat, loc)
                mycol = mydb[col_name]
                Response = mycol.find({}).sort('publishedAt',pymongo.DESCENDING)
                Response = parse_json(Response)
                data = data + Response
        return data
    else:
        col_name = "{0}_{1}".format(category, location)
        mycol = mydb[col_name]
        Response = mycol.find({}).sort('publishedAt',pymongo.DESCENDING)
        Response = parse_json(Response) 
        return Response

# @app.post('/')
def news(category, location):
    # For getting all the news articles
    if (category == "all" and location == "all"):
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
                        d_t = article["publishedAt"].split("T")
                        d = d_t[0].split("-")
                        year, month, date = d[0], d[1], d[2]
                        t = d_t[1].split(":")
                        hour, minute, sec = t[0], t[1], t[2]
                        sec = sec[:1]
                        pub_at = datetime(int(year), int(month), int(date), int(hour), int(minute), int(sec))
                        article["publishedAt"] = pub_at
                        col_name = "{0}_{1}".format(cat, loc)
                        mycol = mydb[col_name]
                        Response = mycol.find({"title" : headline})
                        Response = parse_json(Response)
                        if Response == []:
                            x = mycol.insert_one(article)
                            print("Document inserted with id: ", x.inserted_id)
                        else:
                            print("Record already exists")
                    else:
                        article["sentiment"] = "negative"
                        article.clear()
        print("Prediction done")
        return all_articles
    
    # For particular category and location
    else:
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
                d_t = article["publishedAt"].split("T")
                d = d_t[0].split("-")
                year, month, date = d[0], d[1], d[2]
                t = d_t[1].split(":")
                hour, minute, sec = t[0], t[1], t[2]
                sec = sec[:1]
                pub_at = datetime(int(year), int(month), int(date), int(hour), int(minute), int(sec))
                article["publishedAt"] = pub_at
                col_name = "{0}_{1}".format(category, location)
                mycol = mydb[col_name]
                Response = mycol.find({"title" : headline})
                Response = parse_json(Response)
                if Response == []:
                    x = mycol.insert_one(article)
                    print("Document inserted with id: ", x.inserted_id)
                else:
                    print("Record already exists")
            else:
                article["sentiment"] = "negative"
                article.clear()
    print("Prediction done")

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

def parse_json(data):
    return json.loads(json_util.dumps(data))

# Insert if not exist   
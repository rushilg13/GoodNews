# Import Modules
from fastapi import FastAPI
import pickle
import uvicorn

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('tranform.pkl', 'rb'))

app = FastAPI()

@app.get('/')
def home():
    return {"Hello" : "Go to /predict"}

@app.get('/{name}')
def home(name:str):
    return {"Hello" : name}

@app.post('/predict')
def predict_news(data:str):
    msg = [data]
    vect = cv.transform(msg).toarray()
    my_prediction = clf.predict(vect)
    if (int(my_prediction) == 1):
        return {"Prediction" : "Positive"}
    else:
        return {"Prediction" : "Negative"}
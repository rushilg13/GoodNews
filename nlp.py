import pandas as pd
import pickle
import warnings 
warnings.filterwarnings('ignore')

# Import dataset
df = pd.read_excel('dataset.xlsx')
# print(df['Headlines'])

# Remove special characters
df['Headlines'] = df["Headlines"].str.replace("[^a-zA-Z#]", " ")
# print(df.head())

# Remove Short words
df['Headlines'] = df['Headlines'].apply(lambda x: " ". join(w for w in x.split() if len(w) > 3))
# print(df.head())

# Individual words considered as tokens
tokenized_headlines = df["Headlines"].apply(lambda x : x.split())

# Lemmatization the words
from nltk.stem import WordNetLemmatizer
Lemmatizer = WordNetLemmatizer()
tokenized_headlines = tokenized_headlines.apply(lambda sentence: [Lemmatizer.lemmatize(word) for word in sentence])
# print(tokenized_headlines.head())
for i in range(len(tokenized_headlines)):
    tokenized_headlines[i] = " ".join(tokenized_headlines[i])
df["Clean Headlines"] = tokenized_headlines
# print(df['Clean Headlines'])

# Drop NA values
df.dropna(inplace=True)

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df= 0.9, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['Clean Headlines'])

# Model Training
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score, accuracy_score
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(bow, df['Sentiment'], random_state=42, test_size=0.25)

model = LogisticRegression()
model.fit(bow, df['Sentiment'])

headline = ["Magnitude 4 earthquake strikes near Hyderabad"]
headline_bow = bow_vectorizer.transform(headline)

pred = model.predict(headline_bow)
print(pred)

filename = 'nlp_test.pkl'
pickle.dump(model, open(filename, 'wb'))
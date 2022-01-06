import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

df = pd.read_excel("C:\\Users\\sanje\\Desktop\\RG\\FastAPI\\datasets\\dataset_health.xlsx")
df.dropna(inplace=True)

# Features and Labels
X = df['Headlines']
y = df['Sentiment']

# tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word')
X = vectorizer.fit_transform(X) 

# Extract Feature With CountVectorizer
# cv = CountVectorizer()
# X = cv.fit_transform(X) 

pickle.dump(vectorizer, open('tranform.pkl', 'wb'))

print(list(y).count(0))
print(list(y).count(1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
print("fitting NB...")
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

from sklearn.linear_model import LogisticRegression
lgs = LogisticRegression()
print("fitting Logistic Reg...")
lgs.fit(X_train,y_train)
print(lgs.score(X_test, y_test)) 

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=42, criterion="gini")
print("fitting Decision Tree...")
clf = clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
print("fitting Random Forest...")
clf = clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# "Support vector classifier"  
from sklearn.svm import SVC 
classifier = SVC(kernel='linear', random_state=0)  
print("Fitting SVM...")
classifier.fit(X_train, y_train)
y_pred= classifier.predict(X_test)  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)  
print(cm)

filename = 'nlp_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

# Alternative Usage of Saved Model
# joblib.dump(clf, 'NB_spam_model.pkl')
# NB_spam_model = open('NB_spam_model.pkl','rb')
# clf = joblib.load(NB_spam_model)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('Email_dataset.csv')
#CLEANINg TEXTS
import re # used for search,clean,modify patterns
import nltk
nltk.download('stopwords') #remove uneccsary oneslike the,are,is
from nltk.corpus import stopwords #collection of data like books articles
from nltk.stem.porter import PorterStemmer #root of word like love,worst
corpus=[] #diff reviews
for i in range(0,101):
    review = re.sub('[^a-zA-Z]', ' ', df['Email'][i]) #rep in list if string,spl with spaces
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not')
    review=[ps.stem(word) for word in review if not word in set(all_stopwords)] #taking stem words
    review=' '.join(review)
    corpus.append(review)
    # print(corpus)
#CREATING BAG OF WORDS
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=100)
X=cv.fit_transform(corpus).toarray()
y=df.iloc[:,-1].values
s=len(X[0])
# print(s)
#SPLITTING THE DATSET
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Training the naive bayes model
# from sklearn.naive_bayes import GaussianNB
# classifier=GaussianNB()
# classifier.fit(X_train, y_train)
#Training the randomforest
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
# PREDICT TEST RES
y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#mAKING CONFUSION MATRIX
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
ac=accuracy_score(y_test,y_pred)
print(cm)
print(ac)
# Predicting a new email
new_email = "Congratulations! You've won a free ticket to Paris."

# Clean and preprocess the new email
review = re.sub('[^a-zA-Z]', ' ', new_email)
review = review.lower()
review = review.split()
review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
processed_email = ' '.join(review)

# Transform into Bag-of-Words format
email_vector = cv.transform([processed_email]).toarray()

# Predict
prediction = classifier.predict(email_vector)[0]

# Show result
print(f"Email: {new_email}")
if prediction == 1:
    print("Spam❌" )
else:
    print("Not Spam✅")



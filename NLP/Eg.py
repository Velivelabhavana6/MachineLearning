import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# IMPORT DATASET
df=pd.read_csv('Reviews.csv') #dataset seperaring txt
#CLEANINg TEXTS
import re # used for search,clean,modify patterns
import nltk
nltk.download('stopwords') #remove uneccsary oneslike the,are,is
from nltk.corpus import stopwords #collection of data like books articles
from nltk.stem.porter import PorterStemmer #root of word like love,worst
corpus=[] #diff reviews
for i in range(0,500):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i]) #rep in list if string,spl with spaces
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not')
    review=[ps.stem(word) for word in review if not word in set(all_stopwords)] #taking stem words
    # It goes through each word in the review.
    # Removes common stopwords.
    # Converts the remaining words to their stemmed (root) form.
    # Collects all these processed words back into a new list called review.


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
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train, y_train)
# PREDICT TEST RES
y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#mAKING CONFUSION MATRIX
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
ac=accuracy_score(y_test,y_pred)
print(ac)
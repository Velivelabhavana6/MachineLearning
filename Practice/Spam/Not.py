import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
nltk.download('stopwords')
emails = [
    "Win a brand new car, click here now!",  # spam
    "Meeting tomorrow at 10 AM",             # not spam
    "Congratulations, you've won a lottery!" # spam
]
labels = [1, 0, 1]  # 1 = spam, 0 = not spam

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')  # Keep 'not' in case of negations

# Define preprocess function
def preprocess(text):
    review = re.sub('[^a-zA-Z]', ' ', text)  
    review = review.lower().split()           
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]  
    return ' '.join(review)

# Preprocess training emails
corpus = [preprocess(email) for email in emails]

# Convert text to numerical data (Bag of Words)
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = np.array(labels)
#Train naive bayes
classifier = GaussianNB()
classifier.fit(X, y)

# multiple new emails
new_emails = ["We Miss You! Update Your Profile on Upadhi.ai", "Can we have meeting tmrw?"]
processed = [preprocess(email) for email in new_emails]
X_new = cv.transform(processed).toarray()

predictions = classifier.predict(X_new)
for email, result in zip(new_emails, predictions):
    print(f"Email: {email}")
    if result == 1:
     print("Spam❌" )
    else:
     print("Not Spam✅")

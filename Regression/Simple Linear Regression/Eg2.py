import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
df=pd.read_csv('bike_price_data.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
#OneHotEncoding the bike names
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X) 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Training the data set
re=LinearRegression()
re.fit(X_train,y_train)
#Predicting
y_pred=re.predict(X_train)
#Visulaizing training set
# plt.scatter(X_train,y_train,color='red')
# plt.plot(X_train,y_pred,color='blue')
plt.scatter(range(len(y_train)), y_train, color='red', label='Actual Price')
plt.plot(range(len(y_pred)), y_pred, color='blue', label='Predicted Price')
plt.title('Bike vs price')
plt.xlabel('Bike')
plt.ylabel('Price')
plt.show()
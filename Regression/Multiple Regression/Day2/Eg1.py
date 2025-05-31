import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
df=pd.read_csv('company_data.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Training the Multiple Linear Regression model on the Training set
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred=regressor.predict(X_test)
np.set_printoptions(precision=2)
# #DISPLAYING 2 VEC
# #reshape the vector len of noof rows is noofcols
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))






import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
Dataset=pd.read_csv('Missing_data.csv')
X=Dataset.iloc[:,:-1].values
y=Dataset.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# print(X_train)
# print(X_test)
# print(y_train)
print(y_test)


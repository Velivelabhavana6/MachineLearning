import numpy as np
import pandas as pd
from sklearn import set_config
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
df=pd.read_csv('Sal_exp.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
# print(regressor)
y_pred=regressor.predict([[6]])
print(y_pred)
#VISUALIZING TRAINING SET
# plt.scatter(X_train,y_train,color='red')
# plt.plot(X_train,regressor.predict(X_train),color='blue')
# plt.title('Salary vs Experience(Training set)')
# plt.xlabel('YEARSOFEXPERIENCE')
# plt.ylabel('SALARY')
# plt.show()
#VISUALIZING TEST SET
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('YEARSOFEXPERIENCE')
plt.ylabel('SALARY')
plt.show()
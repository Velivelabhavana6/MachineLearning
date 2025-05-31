import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
df=pd.read_csv('Nonlineardat.csv')
X=df.iloc[:,1:-1].values  # level
y=df.iloc[:,-1].values   #sal
#TRAINING THE LINEAR REGRESSION
# lin_reg=LinearRegression()
# lin_reg.fit(X,y)
#TRAINIG THE POLYNOMIAL REGRESSION
poly_reg=PolynomialFeatures(degree=4)
#TRANSFORMATION
X_poly=poly_reg.fit_transform(X)
# Train Linear Regression model on polynomial features
linreg=LinearRegression()
linreg.fit(X_poly,y)
#VISUALIZING THE LINEAR REGRESSION 
plt.scatter(X,y,color='red')
#salary should predict
plt.plot(X,linreg.predict(X_poly),color='blue')
plt.title('POS VS SAL')
plt.xlabel('Pos level')
plt.ylabel('sal')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
df=pd.read_csv('Nonlineardat.csv')
X=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
# print(X)

#FOR RESHAPE THE Y IN 2D
#len(y) rep the len of depedent
#1 rep how many cols
y=y.reshape(len(y),1)
# print(y)

#FEATURE SCALING
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)
# print(X)
# print(y)
#TRAINING THE SVR MODEL ON WHOLE DATASET
regressor=SVR(kernel='rbf')
regressor.fit(X,y)

#PREDICTING A NEW RESULT
#for predict method use evrytimr double square braces bcz its except in form of 2d array
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))  #reverse the scaling of y # format error


#VISUALIZING THE SVAR RESULTS
# plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color='red')
# plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)),color='blue')
# plt.title('Truth')
# plt.xlabel('pos level')
# plt.ylabel('sal')
# plt.show()

#VISUALIZING THE SVAR RESULTS(FOR HIGHER RESOLUTION AND SMOOTHER CURVE)
X_grid=np.arange(min(sc_X.inverse_transform(X)),max(sc_X.inverse_transform(X)),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color='red')
plt.plot(X_grid,sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)),color='blue')
plt.title('Truth')
plt.xlabel('pos level')
plt.ylabel('sal')
plt.show()
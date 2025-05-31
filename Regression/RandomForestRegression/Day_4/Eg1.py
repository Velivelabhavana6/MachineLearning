import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
df=pd.read_csv('Nonlineardat.csv')
X=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
#TRAINING
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,y)
regressor.predict([[6.5]])

#VISUALIZING
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('POS vs sal')
plt.xlabel('Position level')
plt.ylabel('Sal')
plt.show()


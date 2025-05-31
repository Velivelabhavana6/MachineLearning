import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
df=pd.read_csv('Nonlineardat.csv')
X=df.iloc[:, 1:-1].values
y=df.iloc[:,-1].values
 #TRAINING THE DATA SET
Regressor=DecisionTreeRegressor(random_state=0)
res=Regressor.fit(X,y)
# print(res)
 # PREDICT
print(res.predict([[6.5]]))
#VISUALIZING 
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,res.predict(X_grid),color='blue')
plt.title('POS vs sal')
plt.xlabel('Position level')
plt.ylabel('Sal')
plt.show()
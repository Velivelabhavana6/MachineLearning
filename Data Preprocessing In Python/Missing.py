import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
dataset=pd.read_csv('Missing_data.csv')
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
X=dataset.iloc[:,:-1].values
imputer.fit(X[:,1:3 ])
X[:,1:3 ]=imputer.transform(X[:,1:3 ])  
print(X)
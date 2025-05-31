import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('Missing_data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],remainder='passthrough')
X = ct.fit_transform(X)
X=np.array(ct.fit_transform(X))
# le=LabelEncoder()
# y=le.fit_transform(y)
print(X)
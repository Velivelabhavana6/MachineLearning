import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#PRINTING DATA SET
data_set = pd.read_csv('Eg1.csv')
# X is independent on2 -1 range is except last col
x=data_set.iloc[:,:-1].values
# Y is dependent vector tyo extract last col 
y=data_set.iloc[:,-1].values
# print(data_set)
print(x)
print(y)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Market.csv',header=None)
transactions=[]
for i in range(0,200):
    transactions.append([str(df.values[i,j]) for j in range(0,15)])
    print("     ")
#trainig the apriori model
from apyori import apriori
rules=apriori(transactions=transactions,min_support=0.01,min_confidence=0.2,min_lift=1.2,min_length=2,max_length=2)
#Diaplaying the first results
results=list(rules)
# print(results)
#putting the results well oraganised in to a pandas data frame
def inspect(results):
    lhs=[tuple(result[2][0][0])for result in results]
    rhs=[tuple(result[2][1][0])for result in results]
    supports=[result[1] for result in results]
    confidences=[result[2][0][2] for result in results]
    lifts=[result[2][0][3] for result in results]
    return list(zip(lhs,rhs,supports,confidences,lifts))
resultsinDataFrame=pd.DataFrame(inspect(results),columns=['Left Hand Side','Right Hand Side','Support','Confidence','Lift'])
#Displaying the results non sorted
# print(resultsinDataFrame)
#Displaying the res sorted by desceding using lifts
print(resultsinDataFrame.nlargest(n=10,columns='Lift')) #top 10 rows  

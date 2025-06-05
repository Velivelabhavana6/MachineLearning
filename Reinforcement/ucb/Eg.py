import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Ads1.csv')
#Implementing UCB
import math
N=100 # rows
d=10  #cols
ads_select=[]
no_of_selections=[0]*d #Getting 10
sums_of_rewards=[0]*d
total_reward=0
for n in range(0,N):
    ad=0
    max_upper_bound=0
    for i in range(0,d):
        if (no_of_selections[i]>0): #it checks whether it is selected or not
            average_reward=sums_of_rewards[i]/no_of_selections[i]
            delta_i=math.sqrt(3/2 *math.log(n+1)/no_of_selections[i])#bcz 0 is infinity
            upper_bound=average_reward+delta_i
        else:
            upper_bound = float('inf')
        if(upper_bound > max_upper_bound):
           max_upper_bound=upper_bound
           ad=i
    ads_select.append(ad)
    no_of_selections[ad]= no_of_selections[ad]+1
    reward=df.values[n,ad]
    sums_of_rewards[ad]=sums_of_rewards[ad]+reward
    total_reward=total_reward+reward


#Visualizing the results
plt.hist(ads_select)
plt.title('ADS Selection')
plt.xlabel('Ads')
plt.ylabel('Number of times with each ad we selected')
plt.show()
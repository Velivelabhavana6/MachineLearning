import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Ads1.csv')
#Implement Thompson Sampling
import random
N=100
d=10
ads_select=[]
number_of_rewards_1=[0] *d
number_of_rewards_0=[0] *d
total_reward=0
for n in range(0,N):
    ad=0
    max_random=0
    for i in range(0,d):
        random_beta=random.betavariate(number_of_rewards_1[i]+1,number_of_rewards_0[i]+1)
        if(random_beta>max_random):
            max_random=random_beta
            ad=i
    ads_select.append(ad)
    reward=df.values[n,ad]
    if reward==1:
        number_of_rewards_1[ad]+=1
    else:
        number_of_rewards_0[ad]+=1
    total_reward+=reward
random.seed(0)
#Visualizing
plt.hist(ads_select)
plt.title('ADS Selection')
plt.xlabel('Ads')
plt.ylabel('Number of times with each ad we selected')
plt.show()
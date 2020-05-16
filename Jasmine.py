#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import pickle
from scipy.stats import norm


# In[7]:


pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.metrics import roc_auc_score


# In[10]:


import numpy as np
from sklearn.preprocessing import StandardScaler


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score 
from matplotlib import pyplot


# In[12]:


from datetime import datetime


# # 1. Priliminary data exploration

# ## 1.1Priliminary exploration on 'customer_service' data set

# In[13]:


datafile = open("customer_service_reps", "rb")
customer_service = pickle.load(datafile)
datafile.close()
    


# In[14]:


customer_service.reset_index()


# In[15]:


customer_service.shape


# In[16]:


customer_service['subid'].nunique()


# In[17]:


cs = customer_service


# In[18]:


##let us see which channels do we have
#we found that we have 3 channels: itunes, google and OTT
customer_service['billing_channel'].unique()


# In[19]:


customer_service.to_csv('customer_service.csv', encoding='utf_8_sig', index=True)


# In[20]:


customer_service['revenue_net_1month'].isnull().sum


# In[21]:


#check the distribution of 'revenue_net_1month', the majority of them is 0.
customer_service['revenue_net_1month'].value_counts(dropna=False)


# In[22]:


# replace the missing value of 'revenue_net_1month' with mean

customer_service['revenue_net_1month'] = customer_service['revenue_net_1month'].fillna(customer_service['revenue_net_1month'].mean())


# In[23]:


# The mean of revenue_net_1month is 2.248 and almost 1/4 of the 'revenue_net_1month' data is 0
customer_service['revenue_net_1month'].describe()


# In[ ]:





# In[ ]:


# I check 3 different channels: 'itunes', 'google' and 'OTT' separately


# In[24]:


##For 'OTT' channel, the num_trail_days is either 14 or 0
customer_ott = customer_service[ customer_service['billing_channel'] == 'OTT']
customer_ott = customer_ott.reset_index(drop=True)
customer_ott['num_trial_days'].unique()


# In[25]:


customer_ott.head(2)


# In[26]:


#let's see how do the OTT subscribers look like? How many of them appear more than once in our customer_ott data set?
customer_ott.shape


# In[27]:


len(customer_ott['subid'])


# In[28]:


##Among 1,848,663 customer_service records, there are 1,209,872 distinct users
customer_ott['subid'].value_counts()


# In[29]:


##Since 'renew' or not is the symbol of a successful conversion, let us check how many missing values are there in 'renew' column
# I found that among 1,848,663 customer_service records,1242416(67%) are missing
customer_ott['renew'].value_counts(dropna=False)


# In[30]:


##Let us fill in the missing value
# if in payment_peirod of '0', there exists 'next_payment' time, then we assume the 'renew' value should be 'True'
def fill_renew(payment_period, next_payment):
    if payment_period == 0:
        if str( next_payment ) == 'NaT':
            return(False)
        else:
            return(True)
            
customer_ott['renew'] = customer_ott.apply(lambda row: fill_renew( row['payment_period'], row['next_payment'] ), axis = 1)


# # 2. A/B testing

# ## 2.1 OTT channel for 14_days and 0_day

# In[31]:


##Now let us calculate the conversion rate for 14_days_trial.

customer_ott_14_base = customer_ott[np.logical_and( customer_ott['num_trial_days'] == 14 , customer_ott['payment_period'] == 0 )].reset_index(drop=True)
customer_ott_14_base = customer_ott_14_base[ customer_ott_14_base['trial_completed_TF'] == True].reset_index(drop=True)


# In[32]:


customer_ott_14_convert_rate = customer_ott_14_base['renew'].sum() / customer_ott_14_base['renew'].count()
customer_ott_14_convert_rate


# In[33]:


##Then calculate the conversion rate for 0_days_trial.

customer_ott_0_base = customer_ott[np.logical_and( customer_ott['num_trial_days'] == 0 , customer_ott['payment_period'] == 0 )].reset_index(drop=True)
customer_ott_0_base = customer_ott_0_base[ customer_ott_0_base['trial_completed_TF'] == True].reset_index(drop=True)


# In[34]:


customer_ott_0_convert_rate = customer_ott_0_base['renew'].sum() / customer_ott_0_base['renew'].count()
customer_ott_0_convert_rate


# In[35]:


## Now we set up our test
#Hypothesis: Alternative 14_days trial improved conversion rates over alternative 0_days trial

#H0: conversion rate of 14_days trial <= conversion rate of 0_days trial
#H1: conversion rate of 14_days trial > conversion rate of 0_days trial

#alpha = 0.05


# In[36]:



p_a = customer_ott_0_convert_rate
p_b = customer_ott_14_convert_rate
N_b = customer_ott_14_base['renew'].count()

z_stats = (p_b-p_a)/np.sqrt(p_a*(1-p_a)/N_b)


# In[37]:


z_stats


# In[38]:


alpha = 0.1
z_alpha = norm.ppf(1-alpha/2)
if z_stats<z_alpha:
    print("Accept Null Hypothesis，14_days trial is not better than 0_days trial")
else:
    print("Reject Null Hypothesis, 14_days trial is better than 0_days trial")


# In[39]:


z_alpha


# ## 2.2 itunes channel for 7_days, 14_days and 0_day

# In[40]:


##For 'itunes' channel, the num_trail_days is either 14, 7 or 0
customer_itunes = customer_service[ customer_service['billing_channel'] == 'itunes']
customer_itunes = customer_itunes.reset_index(drop=True)
customer_itunes['num_trial_days'].unique()


# In[41]:


customer_itunes.head(2)


# In[42]:


#let's see how do the itunes subscribers look like? How many of them appear more than once in our customer_itunes data set?
customer_itunes.shape


# In[43]:


len(customer_itunes['subid'])


# In[44]:


##Among 301,713 customer_service records, there are 142,253 distinct users
customer_itunes['subid'].value_counts()


# In[45]:


##Since 'renew' or not is the symbol of a successful conversion, let us check how many missing values are there in 'renew' column
# we found that all 301,713 are missing values
customer_itunes['renew'].value_counts(dropna=False)


# In[46]:


#so, we need to fill in the missing results first

# if in payment_peirod of '0', there exists 'next_payment' time, then we assume the 'renew' value should be 'True'

            
customer_itunes['renew'] = customer_itunes.apply(lambda row: fill_renew( row['payment_period'], row['next_payment'] ), axis = 1)


# In[47]:


##Now let us calculate the conversion rate for 14_days_trial.

customer_itunes_14_base = customer_itunes[np.logical_and( customer_itunes['num_trial_days'] == 14 , customer_itunes['payment_period'] == 0 )].reset_index(drop=True)
customer_itunes_14_base = customer_itunes_14_base[ customer_itunes_14_base['trial_completed_TF'] == True].reset_index(drop=True)


# In[48]:


customer_itunes_14_convert_rate = customer_itunes_14_base['renew'].sum() / customer_itunes_14_base['renew'].count()
customer_itunes_14_convert_rate


# In[49]:


##Then calculate the conversion rate for 0_days_trial.

customer_itunes_0_base = customer_itunes[np.logical_and( customer_itunes['num_trial_days'] == 0 , customer_itunes['payment_period'] == 0 )].reset_index(drop=True)
customer_itunes_0_base = customer_itunes_0_base[ customer_itunes_0_base['trial_completed_TF'] == True].reset_index(drop=True)


# In[50]:


customer_itunes_0_base


# In[51]:


customer_itunes_0_convert_rate = customer_itunes_0_base['renew'].sum() / customer_itunes_0_base['renew'].count()
customer_itunes_0_convert_rate


# In[52]:


##data imbalance(1/2): for customer_itunes_0, there is only 1 user!
customer_itunes_0_base[customer_itunes_0_base['num_trial_days'] == 0].info()


# In[53]:


##Then calculate the conversion rate for 7_days_trial.

customer_itunes_7_base = customer_itunes[np.logical_and( customer_itunes['num_trial_days'] == 7 , customer_itunes['payment_period'] == 0 )].reset_index(drop=True)
customer_itunes_7_base = customer_itunes_7_base[ customer_itunes_7_base['trial_completed_TF'] == True].reset_index(drop=True)


# In[54]:


customer_itunes_7_convert_rate = customer_itunes_7_base['renew'].sum() / customer_itunes_7_base['renew'].count()
customer_itunes_7_convert_rate


# In[55]:


## Now we set up our test
#Hypothesis01: Alternative 14_days trial improved conversion rates over alternative 0_days trial

#H0: conversion rate of 14_days trial <= conversion rate of 0_days trial
#H1: conversion rate of 14_days trial > conversion rate of 0_days trial

#alpha = 0.05


# In[56]:


p_a = customer_itunes_0_convert_rate
p_b = customer_itunes_14_convert_rate
N_b = customer_itunes_14_base['renew'].count()

z_stats = (p_b-p_a)/np.sqrt(p_a*(1-p_a)/N_b)


# In[57]:


z_stats


# In[58]:


alpha = 0.1
z_alpha = norm.ppf(1-alpha/2)
if z_stats<z_alpha:
    print("Accept Null Hypothesis，14_days trial is not better than 0_days trial")
else:
    print("Reject Null Hypothesis, 14_days trial is better than 0_days trial")


# In[59]:


#Hypothesis02: Alternative 7_days tiral improved conversion rates over alternative 0_days tiral

#H0: conversion rate of 7_days trial <= conversion rate of 0_days trial
#H1: conversion rate of 7_days trial > conversion rate of 0_days trial

#alpha = 0.05


# In[60]:


p_a = customer_itunes_0_convert_rate
p_b = customer_itunes_7_convert_rate
N_b = customer_itunes_7_base['renew'].count()

z_stats = (p_b-p_a)/np.sqrt(p_a*(1-p_a)/N_b)


# In[61]:


z_stats


# In[62]:


alpha = 0.1
z_alpha = norm.ppf(1-alpha/2)
if z_stats<z_alpha:
    print("Accept Null Hypothesis，7_days trial is not better than 0_days trial")
else:
    print("Reject Null Hypothesis, 7_days trial better than 0_days trial")


# In[63]:


#Hypothesis03: Alternative 7_days tiral improved conversion rates over alternative 14_days tiral

#H0: conversion rate of 7_days tiral <= conversion rate of 14_days tiral
#H1: conversion rate of 7_days tiral > conversion rate of 14_days tiral

#alpha = 0.05


# In[64]:


p_a = customer_itunes_14_convert_rate
p_b = customer_itunes_7_convert_rate
N_b = customer_itunes_7_base['renew'].count()

z_stats = (p_b-p_a)/np.sqrt(p_a*(1-p_a)/N_b)


# In[65]:


z_stats


# In[66]:


alpha = 0.1
z_alpha = norm.ppf(1-alpha/2)
if z_stats<z_alpha:
    print("Accept Null Hypothesis，7_days trial is not better than 14_days trial")
else:
    print("Reject Null Hypothesis, 7_days trial is better than 14_days trial")


# In[67]:


##data imbalace(2.1/2)
#For 'OTT' channel, number of distinct users of 14_days_trial is much more than number of distinct users of 0_days_trial

ott_0_period = customer_ott[customer_ott['payment_period'] == 0]
ott_0_period['account_creation_year'] = pd.DatetimeIndex(ott_0_period['account_creation_date']).year
ott_0_period['account_creation_month'] = pd.DatetimeIndex(ott_0_period['account_creation_date']).month

OTT_days_distribution = ott_0_period.groupby(['account_creation_year', 'account_creation_month', 'num_trial_days'])[['subid','renew']].agg({'subid':'count','renew':'sum'}).rename(columns={'subid': 'num_disntinct_users', 'renew': 'num_true_renewal'})
OTT_days_distribution


# In[68]:


OTT_days_distribution.to_csv('OTT_days_distribution.csv', encoding='utf_8_sig', index=True)


# In[69]:


##data imbalace(2.2/2)
#For 'itunes' channel, number of distinct users of 7_days_trial is much more than number of distinct users of 14_days_trial

itunes_0_period = customer_itunes[customer_itunes['payment_period'] == 0]
itunes_0_period['account_creation_year'] = pd.DatetimeIndex(itunes_0_period['account_creation_date']).year
itunes_0_period['account_creation_month'] = pd.DatetimeIndex(itunes_0_period['account_creation_date']).month

itunes_days_distribution = itunes_0_period.groupby(['account_creation_year', 'account_creation_month', 'num_trial_days'])[['subid','renew']].agg({'subid':'count','renew':'sum'}).rename(columns={'subid': 'num_disntinct_users', 'renew': 'num_true_renewal'})
itunes_days_distribution


# In[70]:


itunes_days_distribution.to_csv('itunes_days_distribution.csv', encoding='utf_8_sig', index=True)


# In[ ]:





# ## 1.2 Exploration on "subscribers" dataset

# In[71]:


datafile = open("subscribers", "rb")
subscribers = pickle.load(datafile)
datafile.close()


# In[72]:


subscribers.reset_index().sort_values('index',ascending=True)


# In[73]:


subscribers.columns


# In[74]:


len(subscribers['subid'])


# In[75]:


# We first check how many unique subscribers. All 227,628 are unique subscribers 
subscribers['subid'].nunique()


# In[76]:


#Then I checked for age , and i found out that there are 35,169 missing values.
subscribers['age'].isnull().sum()


# In[77]:


#I used the mean to fulfill the null value in age column
subscribers['age'] = subscribers['age'].fillna(subscribers['age'].mean())


# In[78]:


#Also, there are weird value in 'age' column.
subscribers['age'].describe()


# In[79]:


subscribers['age'].mean(axis=0)


# In[80]:


#I separated different age group, and I will transfer age into categorical data

def agegroup(age):
    if age >= 10 and age < 18:
        return('Teenager')
    elif age >= 18 and age < 30:
        return('Young Adult')
    elif age >= 30 and age < 55:
        return('Middle-Aged Adult')
    elif age >= 55 and age < 110:
        return('Senior Adult')
    else:
        return('Unknown')


# In[81]:


subscribers['age'] = subscribers['age'].apply(agegroup)


# In[82]:


subscribers['age'].value_counts(dropna=False)


# In[83]:


#I checked the distribution of gender, and found out that the majority of the subscribers are females, and still, there are some missing vlaues.
subscribers['male_TF'].value_counts(dropna=False)


# In[84]:


# I made the missing values in the gender as 'unknown'

subscribers['male_TF'] = subscribers['male_TF'].fillna('unknown')


# In[85]:


#check the distribution of 'preferred_genre', comedy and drama are the top 2 popular genre.
subscribers['preferred_genre'].value_counts(dropna=False)


# In[86]:



#check the distribution of 'intended_use','access to exclusive content' and 'replace OTT' are the top 2 intension

subscribers['intended_use'].value_counts(dropna=False)


# In[87]:


#check the distribution of 'payment_type', i will replace the missing values with 'unknown'
subscribers['payment_type'].value_counts(dropna=False)


# In[88]:


# replace the missing value of 'payment_type' with'unknown'

subscribers['payment_type'] = subscribers['payment_type'].fillna('unknown')


# In[89]:


#check the distribution of 'num_weekly_services_utilized', i will replace the null value with the mean.
subscribers['num_weekly_services_utilized'].value_counts(dropna=False)


# In[90]:


# replace the missing value of 'num_weekly_services_utilized' with'mean

subscribers['num_weekly_services_utilized'] = subscribers['num_weekly_services_utilized'].fillna(subscribers['num_weekly_services_utilized'].mean())
                                                                                                                                                                      


# In[91]:


# The majority of 'num_weekly_services_utilized' ranges from 2.7-3, while there are still outliers like: 14.335
subscribers['num_weekly_services_utilized'].describe()


# In[92]:


#check the distribution of 'weekly_consumption_hour', i will replace the null value with the mean.
subscribers['weekly_consumption_hour'].value_counts(dropna=False)


# In[93]:


# replace the missing value of 'weekly_consumption_hour' with mean

subscribers['weekly_consumption_hour'] = subscribers['weekly_consumption_hour'].fillna(subscribers['weekly_consumption_hour'].mean())
  


# In[94]:


# The majority of 'num_weekly_services_utilized' ranges from 25-30, while there are still outliers like: -32 and 76
subscribers['weekly_consumption_hour'].describe()


# In[95]:



#check the distribution of 'num_ideal_streaming_services', i will replace the null value with the mean.
subscribers['num_ideal_streaming_services'].value_counts(dropna=False)


# In[96]:


# replace the missing value of 'num_ideal_streaming_services' with mean

subscribers['num_ideal_streaming_services'] = subscribers['num_ideal_streaming_services'].fillna(subscribers['num_ideal_streaming_services'].mean())
 


# In[97]:


# On average, subsribers prefer to take 2-3 serivces
subscribers['num_ideal_streaming_services'].describe()


# In[98]:


#check the distribution of 'retarget_TF', no missing data.
subscribers['retarget_TF'].value_counts(dropna=False)


# In[99]:


#check the distribution of 'revenue_net', the majority of them is 0.
subscribers['revenue_net'].value_counts(dropna=False)


# In[100]:


# replace the missing value of 'revenue_net' with mean

subscribers['revenue_net'] = subscribers['revenue_net'].fillna(subscribers['revenue_net'].mean())
 
    


# In[101]:


# The mean of net revenue is 1.5 and most of the net revenue data is 0
subscribers['revenue_net'].describe()


# ## 1.3 Priliminary exploration on 'engagement' dataset

# In[102]:


datafile = open("engagement", "rb")
engagement = pickle.load(datafile)
datafile.close()


# In[103]:


engagement.reset_index()


# In[104]:


engagement.columns


# In[105]:


#first check how many distinct users in 'engagement' dataset
len(engagement['subid'])


# In[106]:


#Among 2,585,724 engagement data, there are only 135,019 distinct users
engagement['subid'].nunique()


# In[107]:


#check the distribution of 'app_opens'.
engagement['app_opens'].value_counts(dropna=False)


# In[108]:


#There are 34,611 null values so i will replace it with 0.

engagement['app_opens'].isnull().sum()


# In[109]:


#replace the null value with value: 0 
engagement['app_opens']= engagement['app_opens'].fillna(0)


# In[110]:


engagement['app_opens'].describe()


# In[111]:


#check the distribution of 'cust_service_mssgs'.
engagement['cust_service_mssgs'].value_counts(dropna=False)


# In[112]:


#There are 34,611 null values so i will replace it with 0.

engagement['cust_service_mssgs'].isnull().sum()


# In[113]:


#replace the null value of 'app_opens' with value: 0 
engagement['cust_service_mssgs']= engagement['cust_service_mssgs'].fillna(0)


# In[114]:


engagement['cust_service_mssgs'].describe()


# In[115]:


#check the distribution of 'cust_service_mssgs'.
engagement['num_videos_completed'].value_counts(dropna=False)


# In[116]:


#There are 34,611 null values so i will replace it with 0.

engagement['num_videos_completed'].isnull().sum()


# In[117]:


#replace the null value of 'app_opens' with value: 0 
engagement['num_videos_completed']= engagement['num_videos_completed'].fillna(0)


# In[118]:


engagement['num_videos_completed'].describe()


# In[119]:


#check the distribution of 'cust_service_mssgs'.
engagement['num_videos_more_than_30_seconds'].value_counts(dropna=False)


# In[120]:


#There are 34,611 null values so i will replace it with 0.

engagement['num_videos_more_than_30_seconds'].isnull().sum()


# In[121]:


#replace the null value of 'app_opens' with value: 0 
engagement['num_videos_more_than_30_seconds']= engagement['num_videos_more_than_30_seconds'].fillna(0)


# In[122]:


engagement['num_videos_more_than_30_seconds'].describe()


# In[123]:


#check the distribution of 'num_videos_rated'.
engagement['num_videos_rated'].value_counts(dropna=False)


# In[124]:


#There are 34,611 null values so i will replace it with 0.

engagement['num_videos_rated'].isnull().sum()


# In[125]:


#replace the null value of 'num_videos_rated' with value: 0 
engagement['num_videos_rated']= engagement['num_videos_rated'].fillna(0)


# In[126]:


engagement['num_videos_rated'].describe()


# In[127]:


#check the distribution of 'num_series_started.
engagement['num_series_started'].value_counts(dropna=False)


# In[128]:


#There are 34,611 null values so i will replace it with 0.

engagement['num_series_started'].isnull().sum()


# In[129]:


#replace the null value of 'num_videos_rated' with value: 0 
engagement['num_series_started']= engagement['num_series_started'].fillna(0)


# In[130]:


engagement['num_series_started'].describe()


# In[ ]:





# In[ ]:





# In[ ]:





# # 3 Time to do the clustering

# ## 3.1 To do the cluster, let us first try to merge 3 data sets together: Subscribers, Engagement, and Customer Service Reps

# In[131]:


#Let us first see how many user data are there after the merge of 3 data set.


# In[132]:


customer_service['subid01'] = customer_service['subid']
customer_service = customer_service.drop(['subid'],axis = 1)


# In[133]:


subscribers['subid02'] = subscribers['subid']
subscribers= subscribers.drop(['subid'],axis = 1)


# In[134]:


engagement['subid03'] = engagement['subid']
engagement= engagement.drop(['subid'],axis = 1)


# In[135]:


a = set(customer_service['subid01'])
b =set(subscribers['subid02'])
c =set(engagement['subid03'])


# In[136]:


##135,019 distinct users after the merge
ab = a.intersection(b)
abc = ab.intersection(c)
len(abc)


# Here, we start to merge the 3 datasets

# In[137]:



##Here, we start to merge the 3 datasets
pd01 = pd.merge(customer_service,subscribers, left_on='subid01', right_on= 'subid02', how = 'inner')
pd02 = pd.merge(pd01,engagement, left_on='subid01', right_on= 'subid03', how = 'inner')
df = pd02


# In[138]:


df['subid'] = df['subid01'] 
df = df.drop(['subid02','subid03', 'subid01'],axis = 1)
df['subid'].nunique()


# In[139]:


df.columns


# In[140]:


#check the'renew'column, there are missing values, so i fill in the missing values first.
df['renew'].value_counts(dropna=False)


# In[141]:


##Let us fill in the missing value
# if in payment_peirod of '0', there exists 'next_payment' time, then we assume the 'renew' value should be 'True'
def fill_renew02(payment_period_x, next_payment):
    if payment_period_x == 0:
        if str( next_payment ) == 'NaT':
            return(False)
        else:
            return(True)


# In[142]:



#fill in the missing vlaues using the function 'fill_renew' written before
df['renew'] = df.apply(lambda row: fill_renew( row['payment_period_x'], row['next_payment'] ), axis = 1)


# In[143]:


df['renew'].value_counts(dropna=False)


# ## 3.2 Get dummy variables

# In[144]:



##get dummy
df['retarget_TF'] = df['retarget_TF'].astype('object')
cat_features = ['preferred_genre','intended_use', 'retarget_TF', 'male_TF', 'payment_type', 'age', 'renew']
df02 = df.drop(cat_features,axis=1) #keep the numeric and time features

for feature in cat_features:
    dummy = pd.get_dummies(df[cat_features])
df02 = pd.concat( [df02, dummy], axis = 1 )


# In[145]:


df02.columns


# In[146]:


dummy.columns


# ## 3.3 Aggregated dummy variables and numerical variables that would be used to make cluster

# In[147]:


group_by_dummy_features = ['retarget_TF_False', 'retarget_TF_True', 'preferred_genre_comedy', 'preferred_genre_drama',
       'preferred_genre_international', 'preferred_genre_other',
       'preferred_genre_regional', 'intended_use_access to exclusive content',
       'intended_use_education', 'intended_use_expand international access',
       'intended_use_expand regional access', 'intended_use_other',
       'intended_use_replace OTT', 'intended_use_supplement OTT',
       'male_TF_False', 'male_TF_True', 'payment_type_Apple Pay',
       'payment_type_CBD', 'payment_type_Najim', 'payment_type_Paypal',
       'payment_type_RAKBANK', 'payment_type_Standard Charter',
       'payment_type_unknown', 'age_Middle-Aged Adult', 'age_Senior Adult',
       'age_Unknown', 'age_Young Adult','renew_False', 'renew_True','num_trial_days', 'num_weekly_services_utilized', 'weekly_consumption_hour', 'num_ideal_streaming_services',
'app_opens', 'cust_service_mssgs', 'num_videos_completed', 'num_videos_more_than_30_seconds', 'num_videos_rated', 'num_series_started', 'revenue_net_1month', 'revenue_net']


df03 = df02.groupby('subid')[group_by_dummy_features].mean().reset_index()


# In[148]:


#Now all the data in df03 are distinct users!
len(df03['subid'])


# In[149]:


df04 = df03


# In[150]:


df04.to_csv('df04.csv', encoding='utf_8_sig', index=False)


# ## 3.4 Split df03 into train set and test set

# In[151]:


df_train,df_test=train_test_split(df03,test_size=0.3)


# In[152]:


df_test


# ## 3.5 Standardize the variables of All_data set, train set and test set separately

# In[153]:


##Standardize the all data
df_std = df03.drop([ 'revenue_net_1month', 'revenue_net'],axis = 1)
df_std = np.asarray(df_std)

scaler = StandardScaler()
df_std = scaler.fit(df_std).transform(df_std)


# In[154]:


##Standardize the test set
df_test_std = df_test.drop([ 'revenue_net_1month', 'revenue_net'],axis = 1)
df_test_std = np.asarray(df_test_std)

scaler = StandardScaler()
df_test_std = scaler.fit(df_test_std).transform(df_test_std)


# In[155]:


##Standardize the train set
df_train_std = df_train.drop([ 'revenue_net_1month', 'revenue_net'],axis = 1)
df_train_std = np.asarray(df_train_std)

scaler = StandardScaler()
df_train_std = scaler.fit(df_train_std).transform(df_train_std)


# ## 3.6 Use elbow method to find the best k and conduct clustering with k-means method

# In[156]:


##k-means clustering for test set
#Building the Model
#KMeans Algorithm to decide the optimum cluster number , KMeans++ using Elbow Mmethod

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(df_test_std)
    wcss.append(kmeans.inertia_)

    #inertia_ is the formula used to segregate the data points into clusters


# In[157]:


plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()


# In[158]:


#We choose k as 4
kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=4, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
kmeans.fit(df_test_std)
clusters = kmeans.predict(df_test_std)


# In[159]:


df_test['Cluster'] = clusters
df_test_cluster = df_test.groupby(['Cluster']).mean()
df_test_cluster


# In[ ]:





# In[160]:


##k-means clustering for train set
#Building the Model
#KMeans Algorithm to decide the optimum cluster number , KMeans++ using Elbow Mmethod

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(df_train_std)
    wcss.append(kmeans.inertia_)

   


# In[161]:


plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()


# In[162]:


#We choose k as 4
kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=4, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
kmeans.fit(df_train_std)
clusters = kmeans.predict(df_train_std)


# In[163]:


df_train['Cluster'] = clusters
df_train_cluster = df_train.groupby(['Cluster']).mean()
df_train_cluster


# In[164]:


##k-means clustering for all_data set
#Building the Model
#KMeans Algorithm to decide the optimum cluster number , KMeans++ using Elbow Mmethod

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(df_std)
    wcss.append(kmeans.inertia_)


# In[165]:


plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()


# In[166]:


#We choose k as 4
kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=4, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
kmeans.fit(df_std)
clusters = kmeans.predict(df_std)


# In[167]:


df04['Cluster'] = clusters
df_cluster = df04.groupby(['Cluster']).mean()
df_cluster


# In[168]:


##Check how many users each cluster have
df04['Cluster'].value_counts(dropna=False)


# In[169]:


##Save the cluster results into 'csv' format
df_cluster.to_csv('df_cluster.csv', encoding='utf_8_sig', index=True)


# # 4. Churn model
# 

# In[170]:


##Let us fill in the missing value
# if for one typical customer, there exists 'next_payment' time, then we assume the 'churn' value should be 'False'
def fill_renew03(payment_period, next_payment):
    if payment_period == 0:
        if str( next_payment ) == 'NaT':
            return(True)
        else:
            return(False)
    if payment_period == 1:
        if str( next_payment ) == 'NaT':
            return(True)
        else:
            return(False)        
    if payment_period == 2:
        if str( next_payment ) == 'NaT':
            return(True)
        else:
            return(False)        
    if payment_period == 3:
        if str( next_payment ) == 'NaT':
            return(True)
        else:
            return(False)        
        
    if payment_period == 4:
        if str( next_payment ) == 'NaT':
            return(True)
        else:
            return(False)        
        
    if payment_period == 5:
        if str( next_payment ) == 'NaT':
            return(True)
        else:
            return(False)
     
    if payment_period == 6:
        if str( next_payment ) == 'NaT':
            return(True)
        else:
            return(False)        
            
     
    if payment_period == 7:
        if str( next_payment ) == 'NaT':
            return(True)
        else:
            return(False)             
            
cs['churn'] = cs.apply(lambda row: fill_renew03( row['payment_period'], row['next_payment'] ), axis = 1)


# In[171]:


cs['churn'].value_counts(dropna=False)


# In[172]:


cs = cs.drop(['renew'], axis = 1)
cs = cs.drop(['subid01'], axis = 1)


# In[173]:


cs.head(50)


# In[174]:


cs['subid04'] = cs['subid']
cs= cs.drop(['subid'],axis = 1)


# In[175]:



##Here, we start to merge the 3 datasets:cs, engagement and subsribers
pd03 = pd.merge(cs,subscribers, left_on='subid04', right_on= 'subid02', how = 'inner')
pd04 = pd.merge(pd03,engagement, left_on='subid04', right_on= 'subid03', how = 'inner')
df02 = pd04


# In[176]:


df02 = df02.drop(['subid02'],axis = 1)
df02 = df02.drop(['subid03'],axis = 1)


# In[177]:


df02['churn'].value_counts(dropna=False)


# In[178]:


len(df02['churn'])


# In[179]:


df02['date'].describe()


# In[180]:


#extract the 'month' from 'date' column

df02['date'] = df02['date'].astype('datetime64')
df02['month'] = pd.DatetimeIndex(df02['date']).month


# In[ ]:





# In[181]:


df02['month'].value_counts(dropna=False)


# In[182]:


#separate df02 into month, i will calculate churn month by month


# In[183]:


df02_June = df02.loc[df02['month'] == 6]
df02_July = df02.loc[df02['month'] == 7]
df02_August = df02.loc[df02['month'] == 8]
df02_September= df02.loc[df02['month'] == 9]
df02_October = df02.loc[df02['month'] == 10]
df02_November = df02.loc[df02['month'] == 11]
df02_December = df02.loc[df02['month'] == 12]
df02_January = df02.loc[df02['month'] == 1]
df02_February= df02.loc[df02['month'] == 2]
df02_March = df02.loc[df02['month'] == 3]
df02_April= df02.loc[df02['month'] == 4]


# In[184]:


len(df02_June['subid04'])


# In[185]:


df02_June['CAC'] = 59670/len(df02_June['subid04'])


# In[186]:


df02_June


# # 4.1 Building Churn Model and calculated CLV for June.
# 

# In[187]:


#All the feature i use would be :
#categorical:['preferred_genre','intended_use', 'retarget_TF','age']
#numerical: ['num_weekly_services_utilized', 'weekly_consumption_hour', 'num_ideal_streaming_services',
#'app_opens', 'cust_service_mssgs', 'num_videos_completed', 'num_videos_more_than_30_seconds', 'num_videos_rated', 'num_series_started', 'revenue_net_1month', 'revenue_net']


# In[188]:


df02_June = df02_June[['preferred_genre','intended_use', 'retarget_TF','num_weekly_services_utilized', 'weekly_consumption_hour',
            'app_opens', 'cust_service_mssgs', 'num_videos_rated', 'churn','subid04', 'CAC','monthly_price']]

categoriacal_features = ['preferred_genre','intended_use', 'retarget_TF']
numerical_features = ['num_weekly_services_utilized', 'weekly_consumption_hour', 
            'app_opens', 'cust_service_mssgs', 'num_videos_rated']


# In[189]:


##get dummy
df02_June['retarget_TF'] = df02_June['retarget_TF'].astype('object')


# In[190]:


##get dummy


df02_June02 = df02_June.drop(categoriacal_features,axis=1)

for feature in categoriacal_features:
    dummy = pd.get_dummies(df02_June [categoriacal_features])
df02_June02 = pd.concat( [df02_June02, dummy], axis = 1 )


# In[191]:


#transfer the 'churn'column type
df02_June03 = df02_June02
df02_June03["churn"] = df02_June03 ["churn"].astype(int)
df02_June03


# In[192]:


df = df02_June03.reset_index(drop = True)


# In[193]:


df


# In[194]:


#train test split
X = df.drop('churn', axis = 1)
y= df['churn']
X_noid = X.drop(['subid04','CAC','monthly_price'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_noid, y, test_size = .3,stratify = y, random_state = 30)
#Standardization
sc = StandardScaler() 
X_train_= sc.fit_transform(X_train) 
X_test= sc.transform(X_test) 
# Instantiate the classifier 
clf = RandomForestClassifier(n_estimators=200) 
# Fit to the training data 
clf.fit(X_train, y_train)


# In[195]:


# Predict the labels for the test set 
y_pred = clf.predict(X_test) 
accuracy_score(y_test, y_pred) 
# predict probabilities
probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate scores
auc = roc_auc_score(y_test, probs)
# summarize scores
print('LROC AUC=%.3f' % (auc))


# In[196]:


# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[197]:


#assign the probability of churn to each user
index_y = y_test.index
churnRate = pd.DataFrame(probs, index=index_y)
E= pd.merge(y_test, churnRate, left_index=True, right_index=True)
June_churnRate = pd.merge(X, E,  left_index=True, right_index=True, how='inner')
June_churnRate['churnRate'] = June_churnRate[0]
June_churnRate = June_churnRate.drop([0],axis = 1).reset_index()


# In[198]:


#assign predicted churn to each user with threshold of 0.5
threshold = 0.5
predicted = (probs >= threshold).astype('int')
accuracy = accuracy_score(y_test, predicted)
index_y = y_test.index
B = pd.DataFrame(predicted, index=index_y)

C= pd.merge(y_test, B, left_index=True, right_index=True)
June_predictedChurn = pd.merge(X, C,  left_index=True, right_index=True, how='inner')
June_predictedChurn['predicted churn'] = June_predictedChurn[0]
June_predictedChurn = June_predictedChurn.drop([0],axis = 1)


# In[199]:


monthly_price = June_churnRate.groupby('subid04')['monthly_price'].sum().reset_index()
monthly_price['monthly_price02'] = monthly_price['monthly_price']
monthly_price = monthly_price.drop(['monthly_price'],axis = 1)
June_churnRate02 = pd.merge(June_churnRate,monthly_price,left_on='subid04', right_on= 'subid04', how = 'inner')
June_churnRate02 = June_churnRate02.drop(['monthly_price'], axis = 1)
June_churnRate02['monthly_price'] = June_churnRate02['monthly_price02']
June_churnRate02 = June_churnRate02.drop(['monthly_price02'], axis = 1)
June = June_churnRate02.groupby('subid04')['monthly_price','churnRate','CAC'].mean().reset_index()


# In[200]:


June['clv'] = June['monthly_price']*1.1/(0.1+June['churnRate'])-June['CAC']
June['clv'].sum()/len(June['clv'])


# # 4.2 Building Churn Model and calculated CLV for July.

# In[201]:


df02_July['CAC'] = 48906/len(df02_July['subid04'])


# In[202]:


df02_July = df02_July[['preferred_genre','intended_use', 'retarget_TF','num_weekly_services_utilized', 'weekly_consumption_hour',
            'app_opens', 'cust_service_mssgs', 'num_videos_rated', 'churn','subid04', 'CAC','monthly_price']]

categoriacal_features = ['preferred_genre','intended_use', 'retarget_TF']
numerical_features = ['num_weekly_services_utilized', 'weekly_consumption_hour', 
            'app_opens', 'cust_service_mssgs', 'num_videos_rated']


# In[203]:


##get dummy
df02_July['retarget_TF'] = df02_July['retarget_TF'].astype('object')


# In[204]:


##get dummy


df02_July02 = df02_July.drop(categoriacal_features,axis=1)

for feature in categoriacal_features:
    dummy = pd.get_dummies(df02_July [categoriacal_features])
df02_July02 = pd.concat( [df02_July02, dummy], axis = 1 )


# In[205]:


#transfer the 'churn'column type
df02_July03 = df02_July02
df02_July03["churn"] = df02_July03 ["churn"].astype(int)
df02_July03


# In[208]:


df = df02_July03.reset_index(drop = True)


# In[209]:


#train test split
X = df.drop('churn', axis = 1)
y= df['churn']
X_noid = X.drop(['subid04','CAC','monthly_price'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_noid, y, test_size = .3,stratify = y, random_state = 30)
#Standardization
sc = StandardScaler() 
X_train_= sc.fit_transform(X_train) 
X_test= sc.transform(X_test) 
# Instantiate the classifier 
clf = RandomForestClassifier(n_estimators=200) 
# Fit to the training data 
clf.fit(X_train, y_train)


# In[210]:


# Predict the labels for the test set 
y_pred = clf.predict(X_test) 
accuracy_score(y_test, y_pred) 
# predict probabilities
probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate scores
auc = roc_auc_score(y_test, probs)
# summarize scores
print('LROC AUC=%.3f' % (auc))


# In[211]:


# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[212]:


#assign the probability of churn to each user
index_y = y_test.index
churnRate = pd.DataFrame(probs, index=index_y)
E= pd.merge(y_test, churnRate, left_index=True, right_index=True)
July_churnRate = pd.merge(X, E,  left_index=True, right_index=True, how='inner')
July_churnRate['churnRate'] = July_churnRate[0]
July_churnRate = July_churnRate.drop([0],axis = 1).reset_index()


# In[213]:


#assign predicted churn to each user with threshold of 0.5
threshold = 0.5
predicted = (probs >= threshold).astype('int')
accuracy = accuracy_score(y_test, predicted)
index_y = y_test.index
B = pd.DataFrame(predicted, index=index_y)

C= pd.merge(y_test, B, left_index=True, right_index=True)
July_predictedChurn = pd.merge(X, C,  left_index=True, right_index=True, how='inner')
July_predictedChurn['predicted churn'] = July_predictedChurn[0]
July_predictedChurn = July_predictedChurn.drop([0],axis = 1)


# In[214]:


monthly_price = July_churnRate.groupby('subid04')['monthly_price'].sum().reset_index()
monthly_price['monthly_price02'] = monthly_price['monthly_price']
monthly_price = monthly_price.drop(['monthly_price'],axis = 1)
July_churnRate02 = pd.merge(July_churnRate,monthly_price,left_on='subid04', right_on= 'subid04', how = 'inner')
July_churnRate02 = July_churnRate02.drop(['monthly_price'], axis = 1)
July_churnRate02['monthly_price'] = July_churnRate02['monthly_price02']
July_churnRate02 = July_churnRate02.drop(['monthly_price02'], axis = 1)
July = July_churnRate02.groupby('subid04')['monthly_price','churnRate','CAC'].mean().reset_index()


# In[215]:


July['clv'] = July['monthly_price']*1.1/(0.1+July['churnRate'])-July['CAC']
July['clv'].sum()/len(July['clv'])


# In[ ]:





# # 4.3 Building Churn Model and calculated CLV for August

# In[216]:


df02_August['CAC'] = 53236/len(df02_August['subid04'])


# In[217]:


df02_August = df02_August[['preferred_genre','intended_use', 'retarget_TF','num_weekly_services_utilized', 'weekly_consumption_hour',
            'app_opens', 'cust_service_mssgs', 'num_videos_rated', 'churn','subid04', 'CAC','monthly_price']]

categoriacal_features = ['preferred_genre','intended_use', 'retarget_TF']
numerical_features = ['num_weekly_services_utilized', 'weekly_consumption_hour', 
            'app_opens', 'cust_service_mssgs', 'num_videos_rated']


# In[218]:


##get dummy
df02_August['retarget_TF'] = df02_August['retarget_TF'].astype('object')


# In[219]:


##get dummy


df02_August02 = df02_August.drop(categoriacal_features,axis=1)

for feature in categoriacal_features:
    dummy = pd.get_dummies(df02_August [categoriacal_features])
df02_August02 = pd.concat( [df02_August02, dummy], axis = 1 )


# In[220]:


#transfer the 'churn'column type
df02_August03 = df02_August02
df02_August03["churn"] = df02_August03 ["churn"].astype(int)
df02_August03


# In[221]:


df = df02_August03.reset_index(drop = True)


# In[222]:


#train test split
X = df.drop('churn', axis = 1)
y= df['churn']
X_noid = X.drop(['subid04','CAC','monthly_price'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_noid, y, test_size = .3,stratify = y, random_state = 30)
#Standardization
sc = StandardScaler() 
X_train_= sc.fit_transform(X_train) 
X_test= sc.transform(X_test) 
# Instantiate the classifier 
clf = RandomForestClassifier(n_estimators=200) 
# Fit to the training data 
clf.fit(X_train, y_train)


# In[223]:


# Predict the labels for the test set 
y_pred = clf.predict(X_test) 
accuracy_score(y_test, y_pred) 
# predict probabilities
probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate scores
auc = roc_auc_score(y_test, probs)
# summarize scores
print('LROC AUC=%.3f' % (auc))


# In[224]:


# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[225]:


#assign the probability of churn to each user
index_y = y_test.index
churnRate = pd.DataFrame(probs, index=index_y)
E= pd.merge(y_test, churnRate, left_index=True, right_index=True)
August_churnRate = pd.merge(X, E,  left_index=True, right_index=True, how='inner')
August_churnRate['churnRate'] =August_churnRate[0]
August_churnRate = August_churnRate.drop([0],axis = 1).reset_index()


# In[226]:


#assign predicted churn to each user with threshold of 0.5
threshold = 0.5
predicted = (probs >= threshold).astype('int')
accuracy = accuracy_score(y_test, predicted)
index_y = y_test.index
B = pd.DataFrame(predicted, index=index_y)

C= pd.merge(y_test, B, left_index=True, right_index=True)
August_predictedChurn = pd.merge(X, C,  left_index=True, right_index=True, how='inner')
August_predictedChurn['predicted churn'] = August_predictedChurn[0]
August_predictedChurn = August_predictedChurn.drop([0],axis = 1)


# In[227]:


monthly_price = August_churnRate.groupby('subid04')['monthly_price'].sum().reset_index()
monthly_price['monthly_price02'] = monthly_price['monthly_price']
monthly_price = monthly_price.drop(['monthly_price'],axis = 1)
August_churnRate02 = pd.merge(August_churnRate,monthly_price,left_on='subid04', right_on= 'subid04', how = 'inner')
August_churnRate02 = August_churnRate02.drop(['monthly_price'], axis = 1)
August_churnRate02['monthly_price'] = August_churnRate02['monthly_price02']
August_churnRate02 = August_churnRate02.drop(['monthly_price02'], axis = 1)
August = August_churnRate02.groupby('subid04')['monthly_price','churnRate','CAC'].mean().reset_index()


# In[228]:


August['clv'] = August['monthly_price']*1.1/(0.1+August['churnRate'])-August['CAC']
August['clv'].sum()/len(August['clv'])


# In[ ]:





# In[ ]:





# # 4.4 Building Churn Model and calculated CLV for September

# In[229]:


df02_September['CAC'] = 54931/len(df02_September['subid04'])


# In[230]:


df02_September= df02_September[['preferred_genre','intended_use', 'retarget_TF','num_weekly_services_utilized', 'weekly_consumption_hour',
            'app_opens', 'cust_service_mssgs', 'num_videos_rated', 'churn','subid04', 'CAC','monthly_price']]

categoriacal_features = ['preferred_genre','intended_use', 'retarget_TF']
numerical_features = ['num_weekly_services_utilized', 'weekly_consumption_hour', 
            'app_opens', 'cust_service_mssgs', 'num_videos_rated']


# In[231]:


##get dummy
df02_September['retarget_TF'] = df02_September['retarget_TF'].astype('object')


# In[232]:


##get dummy


df02_September02 = df02_September.drop(categoriacal_features,axis=1)

for feature in categoriacal_features:
    dummy = pd.get_dummies(df02_September [categoriacal_features])
df02_September02 = pd.concat( [df02_September02, dummy], axis = 1 )


# In[233]:


#transfer the 'churn'column type
df02_September03 = df02_September02
df02_September03["churn"] = df02_September03 ["churn"].astype(int)
df02_September03


# In[234]:


df = df02_September03.reset_index(drop = True)


# In[235]:


#train test split
X = df.drop('churn', axis = 1)
y= df['churn']
X_noid = X.drop(['subid04','CAC','monthly_price'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_noid, y, test_size = .3,stratify = y, random_state = 30)
#Standardization
sc = StandardScaler() 
X_train_= sc.fit_transform(X_train) 
X_test= sc.transform(X_test) 
# Instantiate the classifier 
clf = RandomForestClassifier(n_estimators=200) 
# Fit to the training data 
clf.fit(X_train, y_train)


# In[236]:


# Predict the labels for the test set 
y_pred = clf.predict(X_test) 
accuracy_score(y_test, y_pred) 
# predict probabilities
probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate scores
auc = roc_auc_score(y_test, probs)
# summarize scores
print('LROC AUC=%.3f' % (auc))


# In[237]:


# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[238]:


#assign the probability of churn to each user
index_y = y_test.index
churnRate = pd.DataFrame(probs, index=index_y)
E= pd.merge(y_test, churnRate, left_index=True, right_index=True)
September_churnRate = pd.merge(X, E,  left_index=True, right_index=True, how='inner')
September_churnRate['churnRate'] =September_churnRate[0]
September_churnRate = September_churnRate.drop([0],axis = 1).reset_index()


# In[239]:


#assign predicted churn to each user with threshold of 0.5
threshold = 0.5
predicted = (probs >= threshold).astype('int')
accuracy = accuracy_score(y_test, predicted)
index_y = y_test.index
B = pd.DataFrame(predicted, index=index_y)

C= pd.merge(y_test, B, left_index=True, right_index=True)
September_predictedChurn = pd.merge(X, C,  left_index=True, right_index=True, how='inner')
September_predictedChurn['predicted churn'] = September_predictedChurn[0]
September_predictedChurn = September_predictedChurn.drop([0],axis = 1)


# In[240]:


monthly_price = September_churnRate.groupby('subid04')['monthly_price'].sum().reset_index()
monthly_price['monthly_price02'] = monthly_price['monthly_price']
monthly_price = monthly_price.drop(['monthly_price'],axis = 1)
September_churnRate02 = pd.merge(September_churnRate,monthly_price,left_on='subid04', right_on= 'subid04', how = 'inner')
September_churnRate02 = September_churnRate02.drop(['monthly_price'], axis = 1)
September_churnRate02['monthly_price'] = September_churnRate02['monthly_price02']
September_churnRate02 = September_churnRate02.drop(['monthly_price02'], axis = 1)
September = September_churnRate02.groupby('subid04')['monthly_price','churnRate','CAC'].mean().reset_index()


# In[241]:


September['clv'] = September['monthly_price']*1.1/(0.1+September['churnRate'])-September['CAC']
September['clv'].sum()/len(September['clv'])


# In[ ]:





# In[ ]:





# # 4.5 Building Churn Model and calculated CLV for October

# In[242]:


df02_October['CAC'] = 46437/len(df02_October['subid04'])


# In[243]:


df02_October= df02_October[['preferred_genre','intended_use', 'retarget_TF','num_weekly_services_utilized', 'weekly_consumption_hour',
            'app_opens', 'cust_service_mssgs', 'num_videos_rated', 'churn','subid04', 'CAC','monthly_price']]

categoriacal_features = ['preferred_genre','intended_use', 'retarget_TF']
numerical_features = ['num_weekly_services_utilized', 'weekly_consumption_hour', 
            'app_opens', 'cust_service_mssgs', 'num_videos_rated']


# In[244]:


##get dummy
df02_October['retarget_TF'] = df02_October['retarget_TF'].astype('object')


# In[245]:


##get dummy


df02_October02 = df02_October.drop(categoriacal_features,axis=1)

for feature in categoriacal_features:
    dummy = pd.get_dummies(df02_October [categoriacal_features])
df02_October02 = pd.concat( [df02_October02, dummy], axis = 1 )


# In[246]:


#transfer the 'churn'column type
df02_October03 = df02_October02
df02_October03["churn"] = df02_October03 ["churn"].astype(int)
df02_October03


# In[247]:


df = df02_October03.reset_index(drop = True)


# In[248]:


#train test split
X = df.drop('churn', axis = 1)
y= df['churn']
X_noid = X.drop(['subid04','CAC','monthly_price'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_noid, y, test_size = .3,stratify = y, random_state = 30)
#Standardization
sc = StandardScaler() 
X_train_= sc.fit_transform(X_train) 
X_test= sc.transform(X_test) 
# Instantiate the classifier 
clf = RandomForestClassifier(n_estimators=200) 
# Fit to the training data 
clf.fit(X_train, y_train)


# In[249]:


# Predict the labels for the test set 
y_pred = clf.predict(X_test) 
accuracy_score(y_test, y_pred) 
# predict probabilities
probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate scores
auc = roc_auc_score(y_test, probs)
# summarize scores
print('LROC AUC=%.3f' % (auc))


# In[250]:


# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[251]:


#assign the probability of churn to each user
index_y = y_test.index
churnRate = pd.DataFrame(probs, index=index_y)
E= pd.merge(y_test, churnRate, left_index=True, right_index=True)
October_churnRate = pd.merge(X, E,  left_index=True, right_index=True, how='inner')
October_churnRate['churnRate'] =October_churnRate[0]
October_churnRate = October_churnRate.drop([0],axis = 1).reset_index()


# In[252]:


#assign predicted churn to each user with threshold of 0.5
threshold = 0.5
predicted = (probs >= threshold).astype('int')
accuracy = accuracy_score(y_test, predicted)
index_y = y_test.index
B = pd.DataFrame(predicted, index=index_y)

C= pd.merge(y_test, B, left_index=True, right_index=True)
October_predictedChurn = pd.merge(X, C,  left_index=True, right_index=True, how='inner')
October_predictedChurn['predicted churn'] = October_predictedChurn[0]
October_predictedChurn = October_predictedChurn.drop([0],axis = 1)


# In[253]:


monthly_price = October_churnRate.groupby('subid04')['monthly_price'].sum().reset_index()
monthly_price['monthly_price02'] = monthly_price['monthly_price']
monthly_price = monthly_price.drop(['monthly_price'],axis = 1)
October_churnRate02 = pd.merge(October_churnRate,monthly_price,left_on='subid04', right_on= 'subid04', how = 'inner')
October_churnRate02 = October_churnRate02.drop(['monthly_price'], axis = 1)
October_churnRate02['monthly_price'] = October_churnRate02['monthly_price02']
October_churnRate02 = October_churnRate02.drop(['monthly_price02'], axis = 1)
October = October_churnRate02.groupby('subid04')['monthly_price','churnRate','CAC'].mean().reset_index()


# In[254]:


October['clv'] = October['monthly_price']*1.1/(0.1+October['churnRate'])-October['CAC']
October['clv'].sum()/len(October['clv'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 4.6 Building Churn Model and calculated CLV for November

# In[255]:


df02_November['CAC'] = 48351/len(df02_November['subid04'])


# In[256]:


df02_November= df02_November[['preferred_genre','intended_use', 'retarget_TF','num_weekly_services_utilized', 'weekly_consumption_hour',
            'app_opens', 'cust_service_mssgs', 'num_videos_rated', 'churn','subid04', 'CAC','monthly_price']]

categoriacal_features = ['preferred_genre','intended_use', 'retarget_TF']
numerical_features = ['num_weekly_services_utilized', 'weekly_consumption_hour', 
            'app_opens', 'cust_service_mssgs', 'num_videos_rated']


# In[257]:


##get dummy
df02_November['retarget_TF'] = df02_November['retarget_TF'].astype('object')


# In[258]:


##get dummy


df02_November02 = df02_November.drop(categoriacal_features,axis=1)

for feature in categoriacal_features:
    dummy = pd.get_dummies(df02_November[categoriacal_features])
df02_November02 = pd.concat( [df02_November02, dummy], axis = 1 )


# In[259]:


#transfer the 'churn'column type
df02_November03 = df02_November02
df02_November03["churn"] = df02_November03 ["churn"].astype(int)
df02_November03


# In[260]:


df = df02_November03.reset_index(drop = True)


# In[261]:


#train test split
X = df.drop('churn', axis = 1)
y= df['churn']
X_noid = X.drop(['subid04','CAC','monthly_price'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_noid, y, test_size = .3,stratify = y, random_state = 30)
#Standardization
sc = StandardScaler() 
X_train_= sc.fit_transform(X_train) 
X_test= sc.transform(X_test) 
# Instantiate the classifier 
clf = RandomForestClassifier(n_estimators=200) 
# Fit to the training data 
clf.fit(X_train, y_train)


# In[262]:


# Predict the labels for the test set 
y_pred = clf.predict(X_test) 
accuracy_score(y_test, y_pred) 
# predict probabilities
probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate scores
auc = roc_auc_score(y_test, probs)
# summarize scores
print('LROC AUC=%.3f' % (auc))


# In[263]:


# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[264]:


#assign the probability of churn to each user
index_y = y_test.index
churnRate = pd.DataFrame(probs, index=index_y)
E= pd.merge(y_test, churnRate, left_index=True, right_index=True)
November_churnRate = pd.merge(X, E,  left_index=True, right_index=True, how='inner')
November_churnRate['churnRate'] =November_churnRate[0]
November_churnRate = November_churnRate.drop([0],axis = 1).reset_index()


# In[265]:


#assign predicted churn to each user with threshold of 0.5
threshold = 0.5
predicted = (probs >= threshold).astype('int')
accuracy = accuracy_score(y_test, predicted)
index_y = y_test.index
B = pd.DataFrame(predicted, index=index_y)

C= pd.merge(y_test, B, left_index=True, right_index=True)
November_predictedChurn = pd.merge(X, C,  left_index=True, right_index=True, how='inner')
November_predictedChurn['predicted churn'] = November_predictedChurn[0]
November_predictedChurn = November_predictedChurn.drop([0],axis = 1)


# In[266]:


monthly_price = November_churnRate.groupby('subid04')['monthly_price'].sum().reset_index()
monthly_price['monthly_price02'] = monthly_price['monthly_price']
monthly_price = monthly_price.drop(['monthly_price'],axis = 1)
November_churnRate02 = pd.merge(November_churnRate,monthly_price,left_on='subid04', right_on= 'subid04', how = 'inner')
November_churnRate02 = November_churnRate02.drop(['monthly_price'], axis = 1)
November_churnRate02['monthly_price'] = November_churnRate02['monthly_price02']
November_churnRate02 = November_churnRate02.drop(['monthly_price02'], axis = 1)
November = November_churnRate02.groupby('subid04')['monthly_price','churnRate','CAC'].mean().reset_index()


# In[267]:


November['clv'] = November['monthly_price']*1.1/(0.1+November['churnRate'])-November['CAC']
November['clv'].sum()/len(November['clv'])


# In[ ]:





# # 4.7 Building Churn Model and calculated CLV for December

# In[268]:


df02_December['CAC'] = 48123/len(df02_December['subid04'])


# In[269]:


df02_December= df02_December[['preferred_genre','intended_use', 'retarget_TF','num_weekly_services_utilized', 'weekly_consumption_hour',
            'app_opens', 'cust_service_mssgs', 'num_videos_rated', 'churn','subid04', 'CAC','monthly_price']]

categoriacal_features = ['preferred_genre','intended_use', 'retarget_TF']
numerical_features = ['num_weekly_services_utilized', 'weekly_consumption_hour', 
            'app_opens', 'cust_service_mssgs', 'num_videos_rated']


# In[270]:


##get dummy
df02_December['retarget_TF'] = df02_December['retarget_TF'].astype('object')


# In[271]:


##get dummy


df02_December02 = df02_December.drop(categoriacal_features,axis=1)

for feature in categoriacal_features:
    dummy = pd.get_dummies(df02_December[categoriacal_features])
df02_December02 = pd.concat( [df02_December02, dummy], axis = 1 )


# In[272]:


#transfer the 'churn'column type
df02_December03 = df02_December02
df02_December03["churn"] = df02_December03 ["churn"].astype(int)
df02_December03


# In[273]:


df = df02_December03.reset_index(drop = True)


# In[274]:


#train test split
X = df.drop('churn', axis = 1)
y= df['churn']
X_noid = X.drop(['subid04','CAC','monthly_price'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_noid, y, test_size = .3,stratify = y, random_state = 30)
#Standardization
sc = StandardScaler() 
X_train_= sc.fit_transform(X_train) 
X_test= sc.transform(X_test) 
# Instantiate the classifier 
clf = RandomForestClassifier(n_estimators=200) 
# Fit to the training data 
clf.fit(X_train, y_train)


# In[275]:


# Predict the labels for the test set 
y_pred = clf.predict(X_test) 
accuracy_score(y_test, y_pred) 
# predict probabilities
probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate scores
auc = roc_auc_score(y_test, probs)
# summarize scores
print('LROC AUC=%.3f' % (auc))


# In[276]:


# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[277]:


#assign the probability of churn to each user
index_y = y_test.index
churnRate = pd.DataFrame(probs, index=index_y)
E= pd.merge(y_test, churnRate, left_index=True, right_index=True)
December_churnRate = pd.merge(X, E,  left_index=True, right_index=True, how='inner')
December_churnRate['churnRate'] =December_churnRate[0]
December_churnRate = December_churnRate.drop([0],axis = 1).reset_index()


# In[278]:


#assign predicted churn to each user with threshold of 0.5
threshold = 0.5
predicted = (probs >= threshold).astype('int')
accuracy = accuracy_score(y_test, predicted)
index_y = y_test.index
B = pd.DataFrame(predicted, index=index_y)

C= pd.merge(y_test, B, left_index=True, right_index=True)
December_predictedChurn = pd.merge(X, C,  left_index=True, right_index=True, how='inner')
December_predictedChurn['predicted churn'] = December_predictedChurn[0]
December_predictedChurn = December_predictedChurn.drop([0],axis = 1)


# In[279]:


monthly_price = December_churnRate.groupby('subid04')['monthly_price'].sum().reset_index()
monthly_price['monthly_price02'] = monthly_price['monthly_price']
monthly_price = monthly_price.drop(['monthly_price'],axis = 1)
December_churnRate02 = pd.merge(December_churnRate,monthly_price,left_on='subid04', right_on= 'subid04', how = 'inner')
December_churnRate02 = December_churnRate02.drop(['monthly_price'], axis = 1)
December_churnRate02['monthly_price'] = December_churnRate02['monthly_price02']
December_churnRate02 = December_churnRate02.drop(['monthly_price02'], axis = 1)
December = December_churnRate02.groupby('subid04')['monthly_price','churnRate','CAC'].mean().reset_index()


# In[280]:


December['clv'] = December['monthly_price']*1.1/(0.1+December['churnRate'])-December['CAC']
December['clv'].sum()/len(December['clv'])


# In[ ]:





# In[ ]:





# In[ ]:





# # 4.8 Building Churn Model and calculated CLV for January

# In[281]:


df02_January['CAC'] = 47970/len(df02_January['subid04'])


# In[282]:


df02_January= df02_January[['preferred_genre','intended_use', 'retarget_TF','num_weekly_services_utilized', 'weekly_consumption_hour',
            'app_opens', 'cust_service_mssgs', 'num_videos_rated', 'churn','subid04', 'CAC','monthly_price']]

categoriacal_features = ['preferred_genre','intended_use', 'retarget_TF']
numerical_features = ['num_weekly_services_utilized', 'weekly_consumption_hour', 
            'app_opens', 'cust_service_mssgs', 'num_videos_rated']


# In[283]:


##get dummy
df02_January['retarget_TF'] = df02_January['retarget_TF'].astype('object')


# In[284]:


##get dummy


df02_January02 = df02_January.drop(categoriacal_features,axis=1)

for feature in categoriacal_features:
    dummy = pd.get_dummies(df02_January[categoriacal_features])
df02_January02 = pd.concat( [df02_January02, dummy], axis = 1 )


# In[285]:


#transfer the 'churn'column type
df02_January03 = df02_January02
df02_January03["churn"] = df02_January03 ["churn"].astype(int)
df02_January03


# In[286]:


df = df02_January03.reset_index(drop = True)


# In[287]:


#train test split
X = df.drop('churn', axis = 1)
y= df['churn']
X_noid = X.drop(['subid04','CAC','monthly_price'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_noid, y, test_size = .3,stratify = y, random_state = 30)
#Standardization
sc = StandardScaler() 
X_train_= sc.fit_transform(X_train) 
X_test= sc.transform(X_test) 
# Instantiate the classifier 
clf = RandomForestClassifier(n_estimators=200) 
# Fit to the training data 
clf.fit(X_train, y_train)


# In[288]:


# Predict the labels for the test set 
y_pred = clf.predict(X_test) 
accuracy_score(y_test, y_pred) 
# predict probabilities
probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate scores
auc = roc_auc_score(y_test, probs)
# summarize scores
print('LROC AUC=%.3f' % (auc))


# In[289]:


# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[290]:


#assign the probability of churn to each user
index_y = y_test.index
churnRate = pd.DataFrame(probs, index=index_y)
E= pd.merge(y_test, churnRate, left_index=True, right_index=True)
January_churnRate = pd.merge(X, E,  left_index=True, right_index=True, how='inner')
January_churnRate['churnRate'] =January_churnRate[0]
January_churnRate = January_churnRate.drop([0],axis = 1).reset_index()


# In[291]:


#assign predicted churn to each user with threshold of 0.5
threshold = 0.5
predicted = (probs >= threshold).astype('int')
accuracy = accuracy_score(y_test, predicted)
index_y = y_test.index
B = pd.DataFrame(predicted, index=index_y)

C= pd.merge(y_test, B, left_index=True, right_index=True)
January_predictedChurn = pd.merge(X, C,  left_index=True, right_index=True, how='inner')
January_predictedChurn['predicted churn'] = January_predictedChurn[0]
January_predictedChurn = January_predictedChurn.drop([0],axis = 1)


# In[292]:


monthly_price = January_churnRate.groupby('subid04')['monthly_price'].sum().reset_index()
monthly_price['monthly_price02'] = monthly_price['monthly_price']
monthly_price = monthly_price.drop(['monthly_price'],axis = 1)
January_churnRate02 = pd.merge(January_churnRate,monthly_price,left_on='subid04', right_on= 'subid04', how = 'inner')
January_churnRate02 = January_churnRate02.drop(['monthly_price'], axis = 1)
January_churnRate02['monthly_price'] = January_churnRate02['monthly_price02']
January_churnRate02 = January_churnRate02.drop(['monthly_price02'], axis = 1)
January = January_churnRate02.groupby('subid04')['monthly_price','churnRate','CAC'].mean().reset_index()


# In[293]:


January['clv'] = January['monthly_price']*1.1/(0.1+January['churnRate'])-January['CAC']
January['clv'].sum()/len(January['clv'])


# In[ ]:





# # 4.9 Building Churn Model and calculated CLV for February

# In[294]:


df02_February['CAC'] = 48584/len(df02_February['subid04'])


# In[295]:


df02_February= df02_February[['preferred_genre','intended_use', 'retarget_TF','num_weekly_services_utilized', 'weekly_consumption_hour',
            'app_opens', 'cust_service_mssgs', 'num_videos_rated', 'churn','subid04', 'CAC','monthly_price']]

categoriacal_features = ['preferred_genre','intended_use', 'retarget_TF']
numerical_features = ['num_weekly_services_utilized', 'weekly_consumption_hour', 
            'app_opens', 'cust_service_mssgs', 'num_videos_rated']


# In[296]:


##get dummy
df02_February['retarget_TF'] = df02_February['retarget_TF'].astype('object')


# In[297]:


##get dummy


df02_February02 = df02_February.drop(categoriacal_features,axis=1)

for feature in categoriacal_features:
    dummy = pd.get_dummies(df02_February[categoriacal_features])
df02_February02 = pd.concat( [df02_February02, dummy], axis = 1 )


# In[298]:


#transfer the 'churn'column type
df02_February03 = df02_February02
df02_February03["churn"] = df02_February03 ["churn"].astype(int)
df02_February03


# In[299]:


df = df02_February03.reset_index(drop = True)


# In[300]:


#train test split
X = df.drop('churn', axis = 1)
y= df['churn']
X_noid = X.drop(['subid04','CAC','monthly_price'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_noid, y, test_size = .3,stratify = y, random_state = 30)
#Standardization
sc = StandardScaler() 
X_train_= sc.fit_transform(X_train) 
X_test= sc.transform(X_test) 
# Instantiate the classifier 
clf = RandomForestClassifier(n_estimators=200) 
# Fit to the training data 
clf.fit(X_train, y_train)


# In[301]:


# Predict the labels for the test set 
y_pred = clf.predict(X_test) 
accuracy_score(y_test, y_pred) 
# predict probabilities
probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate scores
auc = roc_auc_score(y_test, probs)
# summarize scores
print('LROC AUC=%.3f' % (auc))


# In[302]:


# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[303]:


#assign the probability of churn to each user
index_y = y_test.index
churnRate = pd.DataFrame(probs, index=index_y)
E= pd.merge(y_test, churnRate, left_index=True, right_index=True)
February_churnRate = pd.merge(X, E,  left_index=True, right_index=True, how='inner')
February_churnRate['churnRate'] =February_churnRate[0]
February_churnRate = February_churnRate.drop([0],axis = 1).reset_index()


# In[304]:


#assign predicted churn to each user with threshold of 0.5
threshold = 0.5
predicted = (probs >= threshold).astype('int')
accuracy = accuracy_score(y_test, predicted)
index_y = y_test.index
B = pd.DataFrame(predicted, index=index_y)

C= pd.merge(y_test, B, left_index=True, right_index=True)
February_predictedChurn = pd.merge(X, C,  left_index=True, right_index=True, how='inner')
February_predictedChurn['predicted churn'] = February_predictedChurn[0]
February_predictedChurn = February_predictedChurn.drop([0],axis = 1)


# In[305]:


monthly_price = February_churnRate.groupby('subid04')['monthly_price'].sum().reset_index()
monthly_price['monthly_price02'] = monthly_price['monthly_price']
monthly_price = monthly_price.drop(['monthly_price'],axis = 1)
February_churnRate02 = pd.merge(February_churnRate,monthly_price,left_on='subid04', right_on= 'subid04', how = 'inner')
February_churnRate02 = February_churnRate02.drop(['monthly_price'], axis = 1)
February_churnRate02['monthly_price'] = February_churnRate02['monthly_price02']
February_churnRate02 = February_churnRate02.drop(['monthly_price02'], axis = 1)
February = February_churnRate02.groupby('subid04')['monthly_price','churnRate','CAC'].mean().reset_index()


# In[306]:


February['clv'] = February['monthly_price']*1.1/(0.1+February['churnRate'])-February['CAC']
February['clv'].sum()/len(February['clv'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 4.10 Building Churn Model and calculated CLV for March

# In[307]:


df02_March['CAC'] = 48360*(len(df02_March['subid04'])/(len(df02_March['subid04'])+len(df02_April['subid04'])))/len(df02_March['subid04'])


# In[308]:


df02_March= df02_March[['preferred_genre','intended_use', 'retarget_TF','num_weekly_services_utilized', 'weekly_consumption_hour',
            'app_opens', 'cust_service_mssgs', 'num_videos_rated', 'churn','subid04', 'CAC','monthly_price']]

categoriacal_features = ['preferred_genre','intended_use', 'retarget_TF']
numerical_features = ['num_weekly_services_utilized', 'weekly_consumption_hour', 
            'app_opens', 'cust_service_mssgs', 'num_videos_rated']


# In[309]:


##get dummy
df02_March['retarget_TF'] = df02_March['retarget_TF'].astype('object')


# In[310]:


##get dummy


df02_March02 = df02_March.drop(categoriacal_features,axis=1)

for feature in categoriacal_features:
    dummy = pd.get_dummies(df02_March[categoriacal_features])
df02_March02 = pd.concat( [df02_March02, dummy], axis = 1 )


# In[311]:


#transfer the 'churn'column type
df02_March03 = df02_March02
df02_March03["churn"] = df02_March03 ["churn"].astype(int)
df02_March03


# In[312]:


df = df02_March03.reset_index(drop = True)


# In[313]:


#train test split
X = df.drop('churn', axis = 1)
y= df['churn']
X_noid = X.drop(['subid04','CAC','monthly_price'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_noid, y, test_size = .3,stratify = y, random_state = 30)
#Standardization
sc = StandardScaler() 
X_train_= sc.fit_transform(X_train) 
X_test= sc.transform(X_test) 
# Instantiate the classifier 
clf = RandomForestClassifier(n_estimators=200) 
# Fit to the training data 
clf.fit(X_train, y_train)


# In[314]:


# Predict the labels for the test set 
y_pred = clf.predict(X_test) 
accuracy_score(y_test, y_pred) 
# predict probabilities
probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate scores
auc = roc_auc_score(y_test, probs)
# summarize scores
print('LROC AUC=%.3f' % (auc))


# In[315]:


# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[316]:


#assign the probability of churn to each user
index_y = y_test.index
churnRate = pd.DataFrame(probs, index=index_y)
E= pd.merge(y_test, churnRate, left_index=True, right_index=True)
March_churnRate = pd.merge(X, E,  left_index=True, right_index=True, how='inner')
March_churnRate['churnRate'] =March_churnRate[0]
March_churnRate = March_churnRate.drop([0],axis = 1).reset_index()


# In[317]:


#assign predicted churn to each user with threshold of 0.5
threshold = 0.5
predicted = (probs >= threshold).astype('int')
accuracy = accuracy_score(y_test, predicted)
index_y = y_test.index
B = pd.DataFrame(predicted, index=index_y)

C= pd.merge(y_test, B, left_index=True, right_index=True)
March_predictedChurn = pd.merge(X, C,  left_index=True, right_index=True, how='inner')
March_predictedChurn['predicted churn'] = March_predictedChurn[0]
March_predictedChurn = March_predictedChurn.drop([0],axis = 1)


# In[318]:


monthly_price = March_churnRate.groupby('subid04')['monthly_price'].sum().reset_index()
monthly_price['monthly_price02'] = monthly_price['monthly_price']
monthly_price = monthly_price.drop(['monthly_price'],axis = 1)
March_churnRate02 = pd.merge(March_churnRate,monthly_price,left_on='subid04', right_on= 'subid04', how = 'inner')
March_churnRate02 = March_churnRate02.drop(['monthly_price'], axis = 1)
March_churnRate02['monthly_price'] = March_churnRate02['monthly_price02']
March_churnRate02 = March_churnRate02.drop(['monthly_price02'], axis = 1)
March = March_churnRate02.groupby('subid04')['monthly_price','churnRate','CAC'].mean().reset_index()


# In[319]:


March['clv'] = March['monthly_price']*1.1/(0.1+March['churnRate'])-March['CAC']
March['clv'].sum()/len(March['clv'])


# In[ ]:





# # 4.11 Building Churn Model and calculated CLV for April

# In[320]:


df02_April['CAC'] = 48360*(len(df02_April['subid04'])/(len(df02_March['subid04'])+len(df02_April['subid04'])))/len(df02_April['subid04'])


# In[321]:


df02_April= df02_April[['preferred_genre','intended_use', 'retarget_TF','num_weekly_services_utilized', 'weekly_consumption_hour',
            'app_opens', 'cust_service_mssgs', 'num_videos_rated', 'churn','subid04', 'CAC','monthly_price']]

categoriacal_features = ['preferred_genre','intended_use', 'retarget_TF']
numerical_features = ['num_weekly_services_utilized', 'weekly_consumption_hour', 
            'app_opens', 'cust_service_mssgs', 'num_videos_rated']


# In[322]:


##get dummy
df02_April['retarget_TF'] = df02_April['retarget_TF'].astype('object')


# In[323]:


##get dummy


df02_April02 = df02_April.drop(categoriacal_features,axis=1)

for feature in categoriacal_features:
    dummy = pd.get_dummies(df02_April[categoriacal_features])
df02_April02 = pd.concat( [df02_April02, dummy], axis = 1 )


# In[324]:


#transfer the 'churn'column type
df02_April03 = df02_April02
df02_April03["churn"] = df02_April03 ["churn"].astype(int)
df02_April03


# In[325]:


df = df02_April03.reset_index(drop = True)


# In[326]:


#train test split
X = df.drop('churn', axis = 1)
y= df['churn']
X_noid = X.drop(['subid04','CAC','monthly_price'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_noid, y, test_size = .3,stratify = y, random_state = 30)
#Standardization
sc = StandardScaler() 
X_train_= sc.fit_transform(X_train) 
X_test= sc.transform(X_test) 
# Instantiate the classifier 
clf = RandomForestClassifier(n_estimators=200) 
# Fit to the training data 
clf.fit(X_train, y_train)


# In[327]:


# Predict the labels for the test set 
y_pred = clf.predict(X_test) 
accuracy_score(y_test, y_pred) 
# predict probabilities
probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate scores
auc = roc_auc_score(y_test, probs)
# summarize scores
print('LROC AUC=%.3f' % (auc))


# In[328]:


# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(y_test, probs)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[329]:


#assign the probability of churn to each user
index_y = y_test.index
churnRate = pd.DataFrame(probs, index=index_y)
E= pd.merge(y_test, churnRate, left_index=True, right_index=True)
April_churnRate = pd.merge(X, E,  left_index=True, right_index=True, how='inner')
April_churnRate['churnRate'] =April_churnRate[0]
April_churnRate = April_churnRate.drop([0],axis = 1).reset_index()


# In[330]:


#assign predicted churn to each user with threshold of 0.5
threshold = 0.5
predicted = (probs >= threshold).astype('int')
accuracy = accuracy_score(y_test, predicted)
index_y = y_test.index
B = pd.DataFrame(predicted, index=index_y)

C= pd.merge(y_test, B, left_index=True, right_index=True)
April_predictedChurn = pd.merge(X, C,  left_index=True, right_index=True, how='inner')
April_predictedChurn['predicted churn'] = April_predictedChurn[0]
April_predictedChurn = April_predictedChurn.drop([0],axis = 1)


# In[331]:


monthly_price = April_churnRate.groupby('subid04')['monthly_price'].sum().reset_index()
monthly_price['monthly_price02'] = monthly_price['monthly_price']
monthly_price = monthly_price.drop(['monthly_price'],axis = 1)
April_churnRate02 = pd.merge(April_churnRate,monthly_price,left_on='subid04', right_on= 'subid04', how = 'inner')
April_churnRate02 = April_churnRate02.drop(['monthly_price'], axis = 1)
April_churnRate02['monthly_price'] = April_churnRate02['monthly_price02']
April_churnRate02 = April_churnRate02.drop(['monthly_price02'], axis = 1)
April = April_churnRate02.groupby('subid04')['monthly_price','churnRate','CAC'].mean().reset_index()


# In[332]:


April['clv'] = April['monthly_price']*1.1/(0.1+April['churnRate'])-April['CAC']
April['clv'].sum()/len(March['clv'])


# # 5. Aggregate the data of each month and get the ultimate clv and churn probability

# In[333]:


ALLDATA = pd.concat([June,July,August,September,October,November,December,January,February,March,April], sort=True)


# In[335]:


ALLDATA02 = ALLDATA.groupby('subid04')['clv','churnRate'].mean().reset_index()


# In[336]:


monthly_revenue_per_subscriber = ALLDATA02 ['clv'].sum()/len(ALLDATA02['clv'])


# In[337]:


monthly_churn = ALLDATA02 ['churnRate'].sum()/len(ALLDATA02['churnRate'])


# In[339]:


#monthly_revenue_per_subscriber = 49
monthly_revenue_per_subscriber


# In[340]:


#monthly_churn rate = 0.63
monthly_churn


# In[342]:


ALLDATA02


# In[341]:


ALLDATA02['subid04'].nunique


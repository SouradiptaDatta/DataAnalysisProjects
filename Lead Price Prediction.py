#!/usr/bin/env python
# coding: utf-8

# In[53]:


#IMPORTING THE LIBRARIRES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[54]:


#DATA COLLECTION AND PROCESSING 
#loading the csv data into a Pandas DataFrame
lead_data=pd.read_csv('Lead Sheet 1.csv')


# In[55]:


#print first 5 rows in the dataframe
lead_data.head()


# In[56]:


#print last 5 rows of the database
lead_data.tail()


# In[57]:


#number of rows and columns
lead_data.shape


# In[58]:


#getting some basic information about the data
lead_data.info


# In[59]:


#checking number of missing values
lead_data.isnull().sum()


# In[60]:


#Getting the statistical measures of the data
lead_data.describe()


# In[62]:


#Correlation:
#1.Position Correlation
#2.Negative Correlation

correlation=lead_data.corr()


# In[63]:


#Constructing a heatmap to understand the correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')


# In[64]:


#Checking the distribution of the Lead price
sns.displot(lead_data['LME Lead Cash-Settlement'],color='red')


# In[65]:


#Splitting the Features and Target
X=lead_data.drop(['date','Month-Year','LME Lead Cash-Settlement'],axis=1)
Y=lead_data['LME Lead Cash-Settlement']


# In[66]:


print(X)


# In[67]:


print(Y)


# In[68]:


#Splitting into Training data and Test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[69]:


#Model Training:
#Random Forest Regressor

regressor=RandomForestRegressor(n_estimators=100)


# In[70]:


#Training the model

regressor.fit(X_train,Y_train)


# In[71]:


test_data_prediction=regressor.predict(X_test)


# In[72]:


print(test_data_prediction)


# In[74]:


#R squared error
error_score=metrics.r2_score(Y_test,test_data_prediction)
print("R squared error: ",error_score)


# In[75]:


#Compare the Actual Values and Predicted Values in a Plot

Y_test=list(Y_test)


# In[76]:


plt.plot(Y_test,color='green',label='Actual Value')
plt.plot(test_data_prediction,color='red',label='Predicted Value')
plt.title('Actual Price Vs Predicted Price')
plt.xlabel('Number of Values')
plt.ylabel('Lead Price')
plt.legend()
plt.show()


# In[6]:


test_data_prediction.head(5)


# In[ ]:





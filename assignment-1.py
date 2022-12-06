#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np


# In[3]:


house=pd.read_excel("DS - Assignment Part 1 data set.xlsx")


# In[4]:


house.head(5)


# In[6]:


house.isnull().sum()


# In[197]:


house.shape


# In[196]:


house.corr()


# In[201]:


house.describe()


# In[203]:


house.duplicated().sum()


# In[ ]:





# In[16]:


new_house=house.drop(columns=['Transaction date','latitude','longitude'],axis=1)


# # linear regression

# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[21]:


x=new_house.drop('House price of unit area',axis=1)


# In[24]:


y=new_house['House price of unit area']


# In[27]:


lr=LinearRegression()


# In[129]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.25)


# In[130]:


lr.fit(x_train,y_train)


# In[131]:


from sklearn.metrics import r2_score


# In[132]:


y_pred=lr.predict(x_test)


# In[133]:


r2_score(y_test,y_pred)


# # descision tree

# In[134]:


from sklearn.tree import DecisionTreeRegressor


# In[135]:


dr=DecisionTreeRegressor(criterion='squared_error')


# In[136]:


dr.fit(x_train,y_train)


# In[137]:


y_pred_dt=dr.predict(x_test)


# In[138]:


r2_score(y_test,y_pred_dt)


# # random forest

# In[139]:


from sklearn.ensemble import RandomForestRegressor


# In[140]:


rf=RandomForestRegressor(criterion='squared_error')


# In[141]:


rf.fit(x_train,y_train)


# In[142]:


y_pred_rf=rf.predict(x_test)


# In[143]:


r2_score(y_test,y_pred_rf)


# #  svm

# In[144]:


from sklearn.svm import SVR


# In[145]:


sv=SVR(kernel='rbf')


# In[146]:


sv.fit(x_test,y_test)


# In[147]:


y_pred_svm=sv.predict(x_test)


# In[148]:


r2_score(y_test,y_pred_svm)


# #  ridge regression

# In[149]:


from sklearn.linear_model import Ridge


# In[190]:


rr=Ridge(alpha=.001)


# In[191]:


rr.fit(x_train,y_train)


# In[192]:


y_pred_ridge=rr.predict(x_test)


# In[193]:


r2_score(y_test,y_pred_ridge)


# # xgboost

# In[ ]:





# In[152]:


import xgboost as xgb


# In[153]:


xg=xgb.XGBRegressor()


# In[154]:


xg.fit(x_train,y_train)


# In[155]:


y_pred_xgb=xg.predict(x_test)


# In[156]:


r2_score(y_test,y_pred_xgb)


# In[219]:


linear_regression_r2score=r2_score(y_test,y_pred)
decision_tre_r2score=r2_score(y_test,y_pred_dt)
random_forest_r2score=r2_score(y_test,y_pred_rf)
svm_r2score=r2_score(y_test,y_pred_svm)
ridge_regression_r2score=r2_score(y_test,y_pred_ridge)
xg_boost_r2score=r2_score(y_test,y_pred_xgb)


# In[220]:


print(linear_regression_r2score,decision_tre_r2score,random_forest_r2score,svm_r2score,ridge_regression_r2score,xg_boost_r2score)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[87]:


import pandas as pd
import numpy as np


# In[88]:


amazone=pd.read_csv("amz_com-ecommerce_sample.csv",error_bad_lines=False,encoding='latin-1')
flipkart=pd.read_csv("flipkart_com-ecommerce_sample.csv",error_bad_lines=False,encoding='latin-1')


# In[89]:


flipkart.tail(5)


# In[90]:


amazone_data=amazone[["product_name","retail_price","discounted_price"]]


# In[91]:


amazone_data


# In[92]:


# amazone_data=amazone_data.rename(columns={"product_name":"amazone_product_name","retail_price":"amazone_retail_price","discounted_price":"amazone_discounted_price"})


# In[ ]:





# In[93]:


flipkart_data=flipkart[["product_name","retail_price","discounted_price"]]


# In[94]:


flipkart_data


# In[95]:


# flipkart_data=flipkart_data.rename(columns={"product_name":"flipkart_product_name","retail_price":"flipkart_retail_price","discounted_price":"flipkart_discounted_price"})


# In[98]:


new_dataframe=pd.concat([amazone_data,flipkart_data],axis=1,keys=["amazone_data","flipkart_data"])


# In[99]:


new_dataframe


# In[103]:


new_dataframe[~new_dataframe['amazone_data']['product_name'].isin(new_dataframe['flipkart_data']['product_name'])]


# # since both the data set have same product names if the names were different in the dataset then we can use the NLP

# # and if we have to select the similar products then we can use the cosine similarity for that 

# In[ ]:





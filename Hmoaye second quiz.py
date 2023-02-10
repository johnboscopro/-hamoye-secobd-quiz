#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[3]:


df = pd.read_csv('energydata_complete (1).csv')


# In[5]:


#select a sample of the dataset
simple_linear_reg_df = df[['T2', 'T6']].sample(15, random_state=2)


# In[6]:


#regression plot
sns.regplot(x="T2", y="T6",
data=simple_linear_reg_df)


# In[11]:


df.drop('date', inplace=True, axis=1)


# In[12]:


df.drop('lights', inplace=True, axis=1)


# In[16]:


#Firstly, we normalise our dataset to a common scale using the min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalised_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
target_variable = normalised_df['Appliances']


# In[17]:


#Now, we split our dataset into the training and testing dataset. Recall that we had earlier segmented the features and target variables.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, target_variable,
test_size=0.3, random_state=42)


# In[18]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
#fit the model to the training dataset
linear_model.fit(x_train, y_train)
#obtain predictions
predicted_values = linear_model.predict(x_test)


# In[ ]:


y_pred = linear_model


# In[22]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predicted_values))


# In[33]:


print('Root Mean Square Eror',np.sqrt(metrics.mean_absolute_error(y_test, predicted_values)))


# In[19]:


#MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predicted_values)
round(mae, 3)


# ● Root Mean Square Error (RMSE)

# In[21]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 3) 


# In[32]:


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(x_train, y_train)


# In[25]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(x_train, y_train)


# In[26]:


#comparing the effects of regularisation
def get_weights_df(model, feat, col_name):
    #this function returns the weight of every feature
    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weights_df = pd.DataFrame(weights).reset_index()
    weights_df.columns = ['Features', col_name]
    weights_df[col_name].round(3)
    return weights_df


# In[29]:


linear_model_weights = get_weights_df(linear_model, x_train, 'Linear_Model_Weight')
ridge_weights_df = get_weights_df(ridge_reg, x_train, 'Ridge_Weight')
lasso_weights_df = get_weights_df(lasso_reg, x_train, 'Lasso_weight')


# In[30]:


final_weights = pd.merge(linear_model_weights, ridge_weights_df, on='Features')
final_weights = pd.merge(final_weights, lasso_weights_df, on='Features')


# In[31]:


final_weights 


# In[ ]:





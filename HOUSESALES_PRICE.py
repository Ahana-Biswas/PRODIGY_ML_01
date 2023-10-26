#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd

dataset = pd.read_csv("dataset.csv")

dataset.head()


# In[11]:


print(dataset.columns)


# In[13]:


dataset_2 = dataset[['Id', 'LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'SalePrice']]

dataset_2.head()


# In[14]:


dataset_2.info()


# In[15]:


# Check for duplicate rows

duplicates = dataset_2.duplicated().sum()
print(f"{duplicates} duplicate rows found")


# In[16]:


dataset_2.describe()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
correlation_matrix = dataset_2.corr()

# Generate a heatmap
plt.figure()
sns.heatmap(correlation_matrix[['SalePrice']], annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation with SalePrice")
plt.show()


# In[18]:


X = dataset_2[['LotArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr']]
y = dataset_2[['SalePrice']]


# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


# In[20]:


print(model.intercept_, model.coef_, model.score(X_train, y_train))


# In[21]:


y_pred = model.predict(X_test)


# In[22]:


difference = y_test-y_pred
pred_df=pd.DataFrame({'Actual Value': y_test.squeeze(), 'Predicted Value': y_pred.squeeze(), 'Difference': difference.squeeze()})
pred_df.head()


# In[23]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()


# In[24]:


sns.regplot(x=y_test,y=y_pred,ci=None,color ='red')


# In[25]:


from sklearn.metrics import r2_score

Accuracy=r2_score(y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy)


# In[ ]:





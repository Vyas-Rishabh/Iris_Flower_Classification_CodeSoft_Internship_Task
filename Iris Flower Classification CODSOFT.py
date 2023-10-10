#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


iris_data = pd.read_csv("IRIS Dataset.csv")
iris_data.head()


# In[3]:


iris_data.tail()


# ### Statistical Data Analysis

# In[4]:


iris_data.describe()


# In[6]:


#Length of Data
iris_data.shape


# ### summary of a DataFrame

# In[7]:


iris_data.info()


# In[8]:


#Checking null value
iris_data.isnull().sum()


# In[9]:


iris_data['species'].value_counts()


# In[10]:


iris_data['sepal_width'].hist()


# In[11]:


iris_data['sepal_length'].hist()


# In[12]:


iris_data['petal_width'].hist()


# In[13]:


iris_data['petal_length'].hist()


# In[18]:


s = sns.FacetGrid(iris_data, height=8, hue="species")
s.map(plt.scatter, "petal_length", "petal_width")
s.add_legend()
sns.set_style("whitegrid")
plt.show()


# In[19]:


s = sns.FacetGrid(iris_data, height=8, hue="species")
s.map(plt.scatter, "sepal_length", "sepal_width")
s.add_legend()
sns.set_style("whitegrid")
plt.show()


# In[22]:


sns.pairplot(iris_data, height=2.5, hue="species")
plt.show()


# In[25]:


#Checking Correlation use of Heatmap
sns.heatmap(iris_data.corr(), annot=True)
plt.show()


# ### Split the data into training and testing

# In[30]:


from sklearn.model_selection import train_test_split

X = iris_data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = iris_data["species"]


# In[31]:


X


# In[32]:


y


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=12)


# ### Logistic regression model

# In[34]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[35]:


model.fit(X_train,y_train)


# In[39]:


#metrics to get performance
print('Accuracy',model.score(X_test,y_test)*100)


# ### K-Nearest Neighbours model

# In[40]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()


# In[41]:


model.fit(X_train,y_train)


# In[42]:


#metrics to get performance
print('Accuracy',model.score(X_test,y_test)*100)


# ### Decision tree model

# In[43]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()


# In[44]:


model.fit(X_train,y_train)


# In[45]:


#metrics to get performance
print('Accuracy',model.score(X_test,y_test)*100)


# You can find this project on <a href="https://github.com/Vyas-Rishabh/Iris_Flower_Classification_CodeSoft_Internship_Task"><b>GitHub.</b></a>

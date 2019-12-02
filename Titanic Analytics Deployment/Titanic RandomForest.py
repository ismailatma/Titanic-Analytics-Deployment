#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender = pd.read_csv('gender_submission.csv')


# In[3]:


train[["Pclass", "Sex", "SibSp", "Parch"]]


# In[4]:


train["Pclass"].unique()


# In[9]:


train["Sex"].unique()


# In[10]:


train["SibSp"].unique()


# In[11]:


train["Parch"].unique()


# In[6]:


gender


# In[7]:


women = train.loc[train.Sex == 'female']['Survived']
# rate_women = sum(women)/len(women)

# print("% of women who survived:", rate_women)
sum(women)
len(women)


# In[12]:


y = train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X)

output = pd.DataFrame({'PassengerId': train.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[14]:


pickle.dump(model, open('model.pkl','wb'))


# In[18]:


# # Loading model to compare the results
# model = pickle.load( open('model.pkl','rb'))
# print(model.predict([[0,3,1,1,0]]))


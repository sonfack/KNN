
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import  train_test_split
fruits = pd.read_table('/home/Fruits/data.txt')


# In[2]:


fruits.head()


# In[3]:


look_up_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
look_up_fruit_name 


# In[4]:


X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[6]:


from sklearn.neighbors import KNeighborsClassifier


# In[7]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[8]:


knn.fit(X_train, y_train)


# In[9]:


knn.score(X_test, y_test)


# In[10]:


fruit_prediction = knn.predict([[20, 4.5, 5.5]])
print(fruit_prediction[0])
look_up_fruit_name[fruit_prediction[0]]


# In[11]:


fruit_prediction = knn.predict([[100, 6.5, 8.5]])
print(fruit_prediction[0])
look_up_fruit_name[fruit_prediction[0]]


# In[20]:


from sklearn.metrics import classification_report
pred = knn.predict(X_test)
print(classification_report(y_test, pred))


# In[22]:


k_range = range(1,20)
score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, score)
plt.xticks([0, 5, 10, 15, 20])


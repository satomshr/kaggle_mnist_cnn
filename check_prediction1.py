#!/usr/bin/env python
# coding: utf-8

# # check_prediction1.ipynb
# Check prediction and see what are good and what are wrong

# ## Read and check data
# - `train_data` ; train data
# - `pred_data` ; prediction by CNN (results of `pred_classes` and `pred_proba`)

# In[69]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data
train_data = pd.read_csv("../kaggle_mnist/data/train.csv")
pred_data = pd.read_csv("./prediction_CNN1a.csv")

# check data
train_data.info()


# In[70]:


train_data.describe()


# In[71]:


pred_data.info()


# In[72]:


pred_data.describe()


# In[73]:


print(train_data.head())
print(pred_data.head())


# ## Concat train_data and pred_data

# In[74]:


df = pd.concat([train_data, pred_data], axis=1)
df.describe()


# In[75]:


df.head()


# ## Check confusion matrix, and devide into matched and mismatched data

# In[76]:


# check confusion matrix
from sklearn import metrics

co_mat = metrics.confusion_matrix(df["label"], df["Prediction"])
print(co_mat)


# In[77]:


# Devide into matched and mismatched data
df_matched = df.query('label == Prediction')
df_mismatched = df.query('label != Prediction')

df_matched.describe()


# In[78]:


df_mismatched.describe()


# ## Check matched data
# ### Seperate data in each label

# In[79]:


df_matched_ar = []

for i in range(10):
    tmp_ar = df_matched.query('label == {}'.format(i))
    df_matched_ar.append(tmp_ar.sort_values('p{}'.format(i), ascending=False))
    
df_matched_ar[3].head()           


# In[80]:


df_matched_ar[3].tail()


# ### See each data

# In[81]:


# high probability data
h_num = 10
w_num = 10

fig = plt.figure(figsize=(h_num, w_num))
fig.subplots_adjust(hspace=0.5)

for j in range(h_num):
    for i in range(w_num):
        ax = fig.add_subplot(h_num, w_num, i + j*w_num + 1, xticks=[], yticks=[], title=j)
        ax.imshow(df_matched_ar[j].iloc[i, 1:28*28+1].values.reshape((28, 28)), cmap='gray')

plt.show()


# In[82]:


# low probability data
h_num = 10
w_num = 12

fig = plt.figure(figsize=(h_num, w_num))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
fig.subplots_adjust(hspace=0.5)

for j in range(h_num):
    for i in range(w_num):
        label_2nd = df_matched_ar[j].iloc[i, (-10):].nlargest(2).idxmin().replace('p', '')
        ax = fig.add_subplot(h_num, w_num, i + j*w_num + 1, xticks=[], yticks=[], title="{}, {}".format(j, label_2nd))
        ax.imshow(df_matched_ar[j].iloc[-(i+1), 1:28*28+1].values.reshape((28, 28)), cmap='gray')

plt.show()


# ## Check mismatched data
# ### Seperate data in each label

# In[83]:


df_mismatched_ar = []

for i in range(10):
    tmp_ar = df_mismatched.query('label == {}'.format(i))
    df_mismatched_ar.append(tmp_ar.sort_values('p{}'.format(i), ascending=False))
    
df_mismatched_ar[3].head()    


# ### See each data

# In[84]:


h_num = 10
w_num = 12

fig = plt.figure(figsize=(h_num, w_num))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
fig.subplots_adjust(hspace=0.5)

for j in range(h_num):
    w_loop = w_num if w_num < len(df_mismatched_ar[j]) else len(df_mismatched_ar[j])
    for i in range(w_loop):
        pred_label = df_mismatched_ar[j].iloc[i, (-10):].idxmax().replace('p', '')
        ax = fig.add_subplot(h_num, w_num, i + j*w_num + 1, xticks=[], yticks=[], title="{}, {}".format(j, pred_label))
        ax.imshow(df_mismatched_ar[j].iloc[-(i+1), 1:28*28+1].values.reshape((28, 28)), cmap='gray')

plt.show()


# In[ ]:





# In[ ]:





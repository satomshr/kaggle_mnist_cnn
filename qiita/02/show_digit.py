#!/usr/bin/env python
# coding: utf-8

# # show_digit.ipynb
# show the digits in MNIST data

# ## Read and check data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data
train_data = pd.read_csv("../../../kaggle_mnist/data/train.csv")

# check data
train_data.info()


# In[2]:


train_data.describe()


# ## show digits

# In[4]:


h_num = 10
w_num = 10

fig = plt.figure(figsize=(h_num, w_num))
fig.subplots_adjust(hspace=0.5)

for j in range(h_num):
    for i in range(w_num):
        ax = fig.add_subplot(h_num, w_num, i + j*w_num + 1, xticks=[], yticks=[])
        ax.imshow(train_data.iloc[i+j*w_num, 1:28*28+1].values.reshape((28, 28)), cmap='gray')

plt.show()


# In[ ]:





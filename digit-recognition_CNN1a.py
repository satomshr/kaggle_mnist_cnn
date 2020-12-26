#!/usr/bin/env python
# coding: utf-8

# # TensorFlow ; simple network
# Create simple network according to TensorFlow tutorial ( https://www.tensorflow.org/tutorials/images/cnn?hl=ja )

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## load data

# In[2]:


train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[3]:


train_data.info()


# In[4]:


test_data.info()


# In[5]:


train_data_len = len(train_data)
test_data_len = len(test_data)
print("Length of train_data ; {}".format(train_data_len))
print("Length of test_data ; {}".format(test_data_len))


# - Length of train_data ; 42000
# - Length of test_data ; 28000

# In[6]:


train_data_y = train_data["label"]
train_data_x = train_data.drop(columns="label")
train_data_x.head()


# In[7]:


train_data_x = train_data_x.astype('float64').values.reshape((train_data_len, 28, 28, 1))
test_data = test_data.astype('float64').values.reshape((test_data_len, 28, 28, 1))
train_data_x /= 255.0
test_data /= 255.0


# ## Create the convolutional base

# In[8]:


import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()


# ## Add Dense layers on top

# In[9]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


# ## compile and train the model

# In[10]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(train_data_x, train_data_y, epochs=5)


# ## Prediction

# In[ ]:


prediction = model.predict_classes(test_data, verbose=0)
output = pd.DataFrame({"ImageId" : np.arange(1, 28000+1), "Label":prediction})
output.head()


# In[ ]:


output.to_csv('digit_recognizer_CNN1a.csv', index=False)
print("Your submission was successfully saved!")


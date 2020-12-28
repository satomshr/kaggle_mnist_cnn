#!/usr/bin/env python
# coding: utf-8

# # TensorFlow ; simple network
# Create simple network according to TensorFlow tutorial ( https://www.tensorflow.org/tutorials/images/cnn?hl=ja )

# In[13]:


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


# ## Load data

# In[14]:


train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[15]:


train_data.info()


# In[16]:


test_data.info()


# In[17]:


train_data_len = len(train_data)
test_data_len = len(test_data)
print("Length of train_data ; {}".format(train_data_len))
print("Length of test_data ; {}".format(test_data_len))


# - Length of train_data ; 42000
# - Length of test_data ; 28000

# In[18]:


train_data_y = train_data["label"]
train_data_x = train_data.drop(columns="label")
train_data_x.head()


# In[19]:


train_data_x = train_data_x.astype('float64').values.reshape((train_data_len, 28, 28, 1))
test_data = test_data.astype('float64').values.reshape((test_data_len, 28, 28, 1))
train_data_x /= 255.0
test_data /= 255.0

from sklearn.model_selection import train_test_split
X, X_cv, y, y_cv = train_test_split(train_data_x, train_data_y, test_size=0.2, random_state=0)


# ## Create the convolutional base

# In[20]:


import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()


# ## Add Dense layers on top

# In[21]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


# ## Compile and train the model

# In[22]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ## Fit and check

# In[23]:


# history = model.fit(X, y, validation_data=(X_cv, y_cv), epochs=20)
# Saturate ; epochs = 6, Maximum of val_accuracy ; epochs = 14
history = model.fit(train_data_x, train_data_y, epochs=14)


# In[24]:


# import matplotlib.pyplot as plt

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.9, 1])
# plt.legend(loc='lower right')
# plt.show()


# ## Save weights

# In[25]:


# model.save_weights('digit_recognizer_CNN2a_weights')


# ## Prediction

# In[26]:


prediction = model.predict_classes(test_data, verbose=0)
output = pd.DataFrame({"ImageId" : np.arange(1, 28000+1), "Label":prediction})
output.head()


# In[27]:


output.to_csv('digit_recognizer_CNN1e_epochs14.csv', index=False)
print("Your submission was successfully saved!")


# ## Save probability for further study

# In[28]:


pred = model.predict_classes(train_data_x, verbose=0)
pred_proba = model.predict_proba(train_data_x, verbose=0)

pred_df = pd.DataFrame(pred, index=np.arange(train_data_len), columns=["Prediction"])
pred_proba_df = pd.DataFrame(pred_proba, index=np.arange(train_data_len), columns=["p{}".format(i) for i in range(10)])

output = pd.concat([pred_df, pred_proba_df], axis=1)
output.head()


# In[29]:


output.to_csv("prediction_CNN1e_epochs14.csv", index=False)
print("Your prediction was successfully saved!")


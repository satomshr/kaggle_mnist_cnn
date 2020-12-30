#!/usr/bin/env python
# coding: utf-8

# # TensorFlow ; simple network
# - Create simple network according to TensorFlow tutorial ( https://www.tensorflow.org/tutorials/images/cnn?hl=ja )
# - Change kernel_size of first layer (from (3,3) to (5,5)), and epochs=14
# - Results ; 0.99035

# In[82]:


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

# In[83]:


train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[84]:


train_data.info()


# In[85]:


test_data.info()


# In[86]:


train_data_len = len(train_data)
test_data_len = len(test_data)
print("Length of train_data ; {}".format(train_data_len))
print("Length of test_data ; {}".format(test_data_len))


# - Length of train_data ; 42000
# - Length of test_data ; 28000

# In[87]:


train_data_y = train_data["label"]
train_data_x = train_data.drop(columns="label")
train_data_x.head()


# In[88]:


train_data_x = train_data_x.astype('float64').values.reshape((train_data_len, 28, 28, 1))
test_data = test_data.astype('float64').values.reshape((test_data_len, 28, 28, 1))
train_data_x /= 255.0
test_data /= 255.0

from sklearn.model_selection import train_test_split
X, X_cv, y, y_cv = train_test_split(train_data_x, train_data_y, test_size=0.2, random_state=0)


# ## Create the convolutional base

# In[89]:


import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.summary()


# ## Add Dense layers on top

# In[90]:


model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


# ## Compile and train the model

# In[91]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ## Fit and check

# In[92]:


# history = model.fit(X, y, validation_data=(X_cv, y_cv), epochs=50)
# when epochs = 42, 43, 22, val_accuracy > 0.992
# when epochs = 18, 48, val_accuracy > 0.9915
ep = 22
history = model.fit(train_data_x, train_data_y, epochs=ep)


# ```
# Epoch 1/50
# 1050/1050 [==============================] - 4s 3ms/step - loss: 0.1455 - accuracy: 0.9544 - val_loss: 0.0656 - val_accuracy: 0.9808
# Epoch 2/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0479 - accuracy: 0.9853 - val_loss: 0.0385 - val_accuracy: 0.9881
# Epoch 3/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0353 - accuracy: 0.9893 - val_loss: 0.0422 - val_accuracy: 0.9868
# Epoch 4/50
# 1050/1050 [==============================] - 4s 4ms/step - loss: 0.0244 - accuracy: 0.9926 - val_loss: 0.0384 - val_accuracy: 0.9883
# Epoch 5/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0230 - accuracy: 0.9926 - val_loss: 0.0476 - val_accuracy: 0.9869
# Epoch 6/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0162 - accuracy: 0.9948 - val_loss: 0.0397 - val_accuracy: 0.9887
# Epoch 7/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0159 - accuracy: 0.9951 - val_loss: 0.0350 - val_accuracy: 0.9905
# Epoch 8/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0121 - accuracy: 0.9960 - val_loss: 0.0484 - val_accuracy: 0.9883
# Epoch 9/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0126 - accuracy: 0.9957 - val_loss: 0.0504 - val_accuracy: 0.9889
# Epoch 10/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0104 - accuracy: 0.9967 - val_loss: 0.0404 - val_accuracy: 0.9911
# Epoch 11/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0093 - accuracy: 0.9972 - val_loss: 0.0485 - val_accuracy: 0.9899
# Epoch 12/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0087 - accuracy: 0.9976 - val_loss: 0.0561 - val_accuracy: 0.9865
# Epoch 13/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0096 - accuracy: 0.9971 - val_loss: 0.0638 - val_accuracy: 0.9887
# Epoch 14/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0087 - accuracy: 0.9977 - val_loss: 0.0433 - val_accuracy: 0.9896
# Epoch 15/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0061 - accuracy: 0.9983 - val_loss: 0.0437 - val_accuracy: 0.9913
# Epoch 16/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0097 - accuracy: 0.9971 - val_loss: 0.0522 - val_accuracy: 0.9890
# Epoch 17/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0067 - accuracy: 0.9983 - val_loss: 0.0466 - val_accuracy: 0.9892
# Epoch 18/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0072 - accuracy: 0.9977 - val_loss: 0.0456 - val_accuracy: 0.9917 *
# Epoch 19/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0059 - accuracy: 0.9984 - val_loss: 0.0599 - val_accuracy: 0.9906
# Epoch 20/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0059 - accuracy: 0.9987 - val_loss: 0.0521 - val_accuracy: 0.9904
# Epoch 21/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0092 - accuracy: 0.9978 - val_loss: 0.0619 - val_accuracy: 0.9904
# Epoch 22/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0048 - accuracy: 0.9986 - val_loss: 0.0579 - val_accuracy: 0.9924 *
# Epoch 23/50
# 1050/1050 [==============================] - 4s 3ms/step - loss: 0.0069 - accuracy: 0.9983 - val_loss: 0.0812 - val_accuracy: 0.9881
# Epoch 24/50
# 1050/1050 [==============================] - 4s 3ms/step - loss: 0.0060 - accuracy: 0.9985 - val_loss: 0.0649 - val_accuracy: 0.9882
# Epoch 25/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0080 - accuracy: 0.9984 - val_loss: 0.0576 - val_accuracy: 0.9906
# Epoch 26/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0083 - accuracy: 0.9980 - val_loss: 0.0807 - val_accuracy: 0.9900
# Epoch 27/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0023 - accuracy: 0.9994 - val_loss: 0.0881 - val_accuracy: 0.9904
# Epoch 28/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0041 - accuracy: 0.9990 - val_loss: 0.0868 - val_accuracy: 0.9892
# Epoch 29/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0104 - accuracy: 0.9982 - val_loss: 0.0746 - val_accuracy: 0.9901
# Epoch 30/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0056 - accuracy: 0.9987 - val_loss: 0.0688 - val_accuracy: 0.9902
# Epoch 31/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0037 - accuracy: 0.9991 - val_loss: 0.0794 - val_accuracy: 0.9896
# Epoch 32/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0082 - accuracy: 0.9985 - val_loss: 0.0697 - val_accuracy: 0.9896
# Epoch 33/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0032 - accuracy: 0.9992 - val_loss: 0.0731 - val_accuracy: 0.9893
# Epoch 34/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0070 - accuracy: 0.9985 - val_loss: 0.1055 - val_accuracy: 0.9882
# Epoch 35/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0067 - accuracy: 0.9986 - val_loss: 0.0937 - val_accuracy: 0.9902
# Epoch 36/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0051 - accuracy: 0.9991 - val_loss: 0.0981 - val_accuracy: 0.9894
# Epoch 37/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0075 - accuracy: 0.9986 - val_loss: 0.0842 - val_accuracy: 0.9910
# Epoch 38/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0034 - accuracy: 0.9992 - val_loss: 0.0781 - val_accuracy: 0.9917 *
# Epoch 39/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0082 - accuracy: 0.9987 - val_loss: 0.1548 - val_accuracy: 0.9856
# Epoch 40/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0047 - accuracy: 0.9993 - val_loss: 0.0882 - val_accuracy: 0.9905
# Epoch 41/50
# 1050/1050 [==============================] - 4s 3ms/step - loss: 0.0061 - accuracy: 0.9989 - val_loss: 0.0941 - val_accuracy: 0.9915
# Epoch 42/50
# 1050/1050 [==============================] - 4s 3ms/step - loss: 0.0090 - accuracy: 0.9987 - val_loss: 0.0824 - val_accuracy: 0.9927 *
# Epoch 43/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 3.3617e-04 - accuracy: 0.9999 - val_loss: 0.0774 - val_accuracy: 0.9927 *
# Epoch 44/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0106 - accuracy: 0.9984 - val_loss: 0.0905 - val_accuracy: 0.9911
# Epoch 45/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0030 - accuracy: 0.9994 - val_loss: 0.0845 - val_accuracy: 0.9902
# Epoch 46/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0040 - accuracy: 0.9992 - val_loss: 0.1251 - val_accuracy: 0.9888
# Epoch 47/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0103 - accuracy: 0.9980 - val_loss: 0.0962 - val_accuracy: 0.9889
# Epoch 48/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0028 - accuracy: 0.9993 - val_loss: 0.0881 - val_accuracy: 0.9919 *
# Epoch 49/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0017 - accuracy: 0.9997 - val_loss: 0.1368 - val_accuracy: 0.9875
# Epoch 50/50
# 1050/1050 [==============================] - 3s 3ms/step - loss: 0.0046 - accuracy: 0.9993 - val_loss: 0.1207 - val_accuracy: 0.9906
# ```

# In[93]:


# import matplotlib.pyplot as plt

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.95, 1])
# plt.legend(loc='lower right')
# plt.show()


# ## Save weights

# In[94]:


# model.save_weights('digit_recognizer_CNN2a_weights')


# ## Prediction

# In[95]:


prediction = model.predict_classes(test_data, verbose=0)
output = pd.DataFrame({"ImageId" : np.arange(1, 28000+1), "Label":prediction})
output.head()


# In[96]:


output.to_csv('digit_recognizer_CNN1f_epochs{}.csv'.format(ep), index=False)
print("Your submission was successfully saved!")


# ## Save probability for further study

# In[97]:


pred = model.predict_classes(train_data_x, verbose=0)
pred_proba = model.predict_proba(train_data_x, verbose=0)

pred_df = pd.DataFrame(pred, index=np.arange(train_data_len), columns=["Prediction"])
pred_proba_df = pd.DataFrame(pred_proba, index=np.arange(train_data_len), columns=["p{}".format(i) for i in range(10)])

output = pd.concat([pred_df, pred_proba_df], axis=1)
output.head()


# In[98]:


output.to_csv("prediction_CNN1f_epochs{}.csv".format(ep), index=False)
print("Your prediction was successfully saved!")


#!/usr/bin/env python
# coding: utf-8

# # TensorFlow ; simple network
# - Create simple network according to TensorFlow tutorial ( https://www.tensorflow.org/tutorials/images/cnn?hl=ja )
# - Change kernel_size of first layer (from (3,3) to (5,5)), channels are increased
# - ImageDataGenerator is used for training
# - Results ; 0.99228 (epochs=30)

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


# ## Load data

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


# ## Set ImageDataGenerator & show examples

# In[8]:


# test of ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=35,
                             width_shift_range=0.25,
                             height_shift_range=0.20,
                             shear_range=0.2,
                             zoom_range=0.2,
                             fill_mode='nearest')


# In[9]:


from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(array_to_img(datagen.random_transform(train_data_x[i])), cmap='gray')
    plt.axis('off')
    
plt.show()


# ## Scaling and split data

# In[10]:


train_data_x /= 255.0
test_data /= 255.0


# In[11]:


from sklearn.model_selection import train_test_split
X, X_cv, y, y_cv = train_test_split(train_data_x, train_data_y, test_size=0.2, random_state=0)


# ## Make random_transform of X_cv

# In[12]:


# I think there are more clever ways to do it :-)
for i in range(len(X_cv)):
    X_cv[i] = datagen.random_transform(X_cv[i])

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(array_to_img(datagen.random_transform(train_data_x[i])), cmap='gray')
    plt.axis('off')
    
plt.show()


# ## Create the convolutional base

# In[13]:


import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(128, (5, 5), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.Dropout(0.2))

model.summary()


# ## Add Dense layers on top

# In[14]:


model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


# ## Compile

# In[15]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ## Set file path of parameter data and callback

# In[16]:


model_dir = "./weights_a/"
data_dir = "./data_a/"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

from keras.callbacks import ModelCheckpoint

model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(model_dir, "model-{epoch:02d}-{val_accuracy:.4f}.hdf5"),
                                            monitor='val_accuracy',
                                            mode='max',
                                            save_best_only=True)


# ## Fit and check

# In[17]:


# history = model.fit(train_data_x, train_data_y, 
#                     validation_split=0.2,
#                     epochs=20,
#                     callbacks=[model_checkpoint_callback])

history = model.fit_generator(datagen.flow(X, y, batch_size=32),
                              steps_per_epoch=len(X)/32,
                              validation_data=(X_cv, y_cv),
                              epochs=99,
                              callbacks=[model_checkpoint_callback])

# history = model.fit(train_data_x, train_data_y, epochs=ep)


# ```
# Epoch 1/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.5829 - accuracy: 0.8045 - val_loss: 0.2298 - val_accuracy: 0.9313
# Epoch 2/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.1968 - accuracy: 0.9391 - val_loss: 0.1992 - val_accuracy: 0.9388
# Epoch 3/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.1487 - accuracy: 0.9539 - val_loss: 0.1478 - val_accuracy: 0.9525
# Epoch 4/99
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.1271 - accuracy: 0.9606 - val_loss: 0.1121 - val_accuracy: 0.9656
# Epoch 5/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.1161 - accuracy: 0.9647 - val_loss: 0.1270 - val_accuracy: 0.9617
# Epoch 6/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.1047 - accuracy: 0.9685 - val_loss: 0.1133 - val_accuracy: 0.9658
# Epoch 7/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0958 - accuracy: 0.9718 - val_loss: 0.1021 - val_accuracy: 0.9692
# Epoch 8/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0949 - accuracy: 0.9724 - val_loss: 0.0950 - val_accuracy: 0.9713
# Epoch 9/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0852 - accuracy: 0.9745 - val_loss: 0.1012 - val_accuracy: 0.9698
# Epoch 10/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0840 - accuracy: 0.9750 - val_loss: 0.0732 - val_accuracy: 0.9787
# Epoch 11/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0846 - accuracy: 0.9746 - val_loss: 0.0934 - val_accuracy: 0.9715
# Epoch 12/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0796 - accuracy: 0.9765 - val_loss: 0.0722 - val_accuracy: 0.9790
# Epoch 13/99
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.0774 - accuracy: 0.9772 - val_loss: 0.0771 - val_accuracy: 0.9793
# Epoch 14/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0787 - accuracy: 0.9769 - val_loss: 0.0773 - val_accuracy: 0.9787
# Epoch 15/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0716 - accuracy: 0.9784 - val_loss: 0.0730 - val_accuracy: 0.9781
# Epoch 16/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0742 - accuracy: 0.9783 - val_loss: 0.0728 - val_accuracy: 0.9787
# Epoch 17/99
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.0693 - accuracy: 0.9801 - val_loss: 0.0781 - val_accuracy: 0.9794
# Epoch 18/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0704 - accuracy: 0.9790 - val_loss: 0.0725 - val_accuracy: 0.9788
# Epoch 19/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0712 - accuracy: 0.9788 - val_loss: 0.0769 - val_accuracy: 0.9790
# Epoch 20/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0641 - accuracy: 0.9805 - val_loss: 0.0677 - val_accuracy: 0.9806
# Epoch 21/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0672 - accuracy: 0.9800 - val_loss: 0.0630 - val_accuracy: 0.9812
# Epoch 22/99
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.0639 - accuracy: 0.9810 - val_loss: 0.0686 - val_accuracy: 0.9801
# Epoch 23/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0666 - accuracy: 0.9813 - val_loss: 0.0696 - val_accuracy: 0.9805
# Epoch 24/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0602 - accuracy: 0.9816 - val_loss: 0.0597 - val_accuracy: 0.9827
# Epoch 25/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0641 - accuracy: 0.9809 - val_loss: 0.0651 - val_accuracy: 0.9799
# Epoch 26/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0626 - accuracy: 0.9819 - val_loss: 0.0640 - val_accuracy: 0.9824
# Epoch 27/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0662 - accuracy: 0.9807 - val_loss: 0.0670 - val_accuracy: 0.9806
# Epoch 28/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0630 - accuracy: 0.9824 - val_loss: 0.0608 - val_accuracy: 0.9831
# Epoch 29/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0595 - accuracy: 0.9829 - val_loss: 0.0741 - val_accuracy: 0.9799
# Epoch 30/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0648 - accuracy: 0.9811 - val_loss: 0.0560 - val_accuracy: 0.9854
# Epoch 31/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0616 - accuracy: 0.9824 - val_loss: 0.0739 - val_accuracy: 0.9787
# Epoch 32/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0569 - accuracy: 0.9834 - val_loss: 0.0758 - val_accuracy: 0.9787
# Epoch 33/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0613 - accuracy: 0.9824 - val_loss: 0.0695 - val_accuracy: 0.9813
# Epoch 34/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0556 - accuracy: 0.9835 - val_loss: 0.0686 - val_accuracy: 0.9824
# Epoch 35/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0581 - accuracy: 0.9828 - val_loss: 0.0637 - val_accuracy: 0.9823
# Epoch 36/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0563 - accuracy: 0.9843 - val_loss: 0.0794 - val_accuracy: 0.9789
# Epoch 37/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0534 - accuracy: 0.9840 - val_loss: 0.0759 - val_accuracy: 0.9798
# Epoch 38/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0566 - accuracy: 0.9831 - val_loss: 0.0670 - val_accuracy: 0.9806
# Epoch 39/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0557 - accuracy: 0.9832 - val_loss: 0.0665 - val_accuracy: 0.9813
# Epoch 40/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0544 - accuracy: 0.9844 - val_loss: 0.0716 - val_accuracy: 0.9806
# Epoch 41/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0560 - accuracy: 0.9835 - val_loss: 0.0811 - val_accuracy: 0.9790
# Epoch 42/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0539 - accuracy: 0.9846 - val_loss: 0.0686 - val_accuracy: 0.9827
# Epoch 43/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0579 - accuracy: 0.9843 - val_loss: 0.0594 - val_accuracy: 0.9826
# Epoch 44/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0564 - accuracy: 0.9837 - val_loss: 0.0603 - val_accuracy: 0.9832
# Epoch 45/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0576 - accuracy: 0.9837 - val_loss: 0.0673 - val_accuracy: 0.9808
# Epoch 46/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0540 - accuracy: 0.9849 - val_loss: 0.0674 - val_accuracy: 0.9814
# Epoch 47/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0570 - accuracy: 0.9832 - val_loss: 0.0706 - val_accuracy: 0.9802
# Epoch 48/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0563 - accuracy: 0.9836 - val_loss: 0.0538 - val_accuracy: 0.9844
# Epoch 49/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0549 - accuracy: 0.9842 - val_loss: 0.0640 - val_accuracy: 0.9819
# Epoch 50/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0553 - accuracy: 0.9843 - val_loss: 0.0605 - val_accuracy: 0.9815
# Epoch 51/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0515 - accuracy: 0.9857 - val_loss: 0.0603 - val_accuracy: 0.9850
# Epoch 52/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0540 - accuracy: 0.9842 - val_loss: 0.0687 - val_accuracy: 0.9821
# Epoch 53/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0570 - accuracy: 0.9839 - val_loss: 0.0659 - val_accuracy: 0.9808
# Epoch 54/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0498 - accuracy: 0.9859 - val_loss: 0.0741 - val_accuracy: 0.9827
# Epoch 55/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0528 - accuracy: 0.9850 - val_loss: 0.0725 - val_accuracy: 0.9814
# Epoch 56/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0578 - accuracy: 0.9850 - val_loss: 0.0630 - val_accuracy: 0.9837
# Epoch 57/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0513 - accuracy: 0.9851 - val_loss: 0.0645 - val_accuracy: 0.9826
# Epoch 58/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0542 - accuracy: 0.9851 - val_loss: 0.0615 - val_accuracy: 0.9825
# Epoch 59/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0510 - accuracy: 0.9852 - val_loss: 0.0581 - val_accuracy: 0.9839
# Epoch 60/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0538 - accuracy: 0.9846 - val_loss: 0.0620 - val_accuracy: 0.9820
# Epoch 61/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0538 - accuracy: 0.9844 - val_loss: 0.0587 - val_accuracy: 0.9842
# Epoch 62/99
# 1050/1050 [==============================] - 15s 14ms/step - loss: 0.0538 - accuracy: 0.9851 - val_loss: 0.0645 - val_accuracy: 0.9833
# Epoch 63/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0547 - accuracy: 0.9855 - val_loss: 0.0651 - val_accuracy: 0.9820
# Epoch 64/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0509 - accuracy: 0.9863 - val_loss: 0.0629 - val_accuracy: 0.9806
# Epoch 65/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0554 - accuracy: 0.9843 - val_loss: 0.0597 - val_accuracy: 0.9833
# Epoch 66/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0532 - accuracy: 0.9845 - val_loss: 0.0649 - val_accuracy: 0.9813
# Epoch 67/99
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.0542 - accuracy: 0.9857 - val_loss: 0.0677 - val_accuracy: 0.9829
# Epoch 68/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0495 - accuracy: 0.9860 - val_loss: 0.0686 - val_accuracy: 0.9825
# Epoch 69/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0486 - accuracy: 0.9865 - val_loss: 0.0731 - val_accuracy: 0.9821
# Epoch 70/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0518 - accuracy: 0.9854 - val_loss: 0.0806 - val_accuracy: 0.9795
# Epoch 71/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0541 - accuracy: 0.9857 - val_loss: 0.0688 - val_accuracy: 0.9808
# Epoch 72/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0533 - accuracy: 0.9855 - val_loss: 0.0706 - val_accuracy: 0.9832
# Epoch 73/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0472 - accuracy: 0.9865 - val_loss: 0.0642 - val_accuracy: 0.9845
# Epoch 74/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0512 - accuracy: 0.9864 - val_loss: 0.0553 - val_accuracy: 0.9846
# Epoch 75/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0508 - accuracy: 0.9861 - val_loss: 0.0604 - val_accuracy: 0.9821
# Epoch 76/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0531 - accuracy: 0.9846 - val_loss: 0.0608 - val_accuracy: 0.9835
# Epoch 77/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0493 - accuracy: 0.9867 - val_loss: 0.0645 - val_accuracy: 0.9832
# Epoch 78/99
# 1050/1050 [==============================] - 12s 12ms/step - loss: 0.0548 - accuracy: 0.9848 - val_loss: 0.0642 - val_accuracy: 0.9823
# Epoch 79/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0489 - accuracy: 0.9861 - val_loss: 0.0569 - val_accuracy: 0.9846
# Epoch 80/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0502 - accuracy: 0.9863 - val_loss: 0.0620 - val_accuracy: 0.9846
# Epoch 81/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0539 - accuracy: 0.9855 - val_loss: 0.0716 - val_accuracy: 0.9824
# Epoch 82/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0524 - accuracy: 0.9854 - val_loss: 0.0663 - val_accuracy: 0.9829
# Epoch 83/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0515 - accuracy: 0.9859 - val_loss: 0.0703 - val_accuracy: 0.9833
# Epoch 84/99
# 1050/1050 [==============================] - 12s 12ms/step - loss: 0.0497 - accuracy: 0.9860 - val_loss: 0.0792 - val_accuracy: 0.9827
# Epoch 85/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0482 - accuracy: 0.9860 - val_loss: 0.0607 - val_accuracy: 0.9855
# Epoch 86/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0551 - accuracy: 0.9857 - val_loss: 0.0632 - val_accuracy: 0.9836
# Epoch 87/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0484 - accuracy: 0.9862 - val_loss: 0.0634 - val_accuracy: 0.9846
# Epoch 88/99
# 1050/1050 [==============================] - 12s 12ms/step - loss: 0.0545 - accuracy: 0.9854 - val_loss: 0.0663 - val_accuracy: 0.9840
# Epoch 89/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0504 - accuracy: 0.9860 - val_loss: 0.0550 - val_accuracy: 0.9855
# Epoch 90/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0475 - accuracy: 0.9862 - val_loss: 0.0576 - val_accuracy: 0.9843
# Epoch 91/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0518 - accuracy: 0.9855 - val_loss: 0.0729 - val_accuracy: 0.9830
# Epoch 92/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0538 - accuracy: 0.9857 - val_loss: 0.0543 - val_accuracy: 0.9855
# Epoch 93/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0492 - accuracy: 0.9860 - val_loss: 0.0756 - val_accuracy: 0.9826
# Epoch 94/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0491 - accuracy: 0.9859 - val_loss: 0.0586 - val_accuracy: 0.9845
# Epoch 95/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0473 - accuracy: 0.9866 - val_loss: 0.0642 - val_accuracy: 0.9849
# Epoch 96/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0523 - accuracy: 0.9855 - val_loss: 0.0679 - val_accuracy: 0.9838
# Epoch 97/99
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0492 - accuracy: 0.9863 - val_loss: 0.0717 - val_accuracy: 0.9843
# Epoch 98/99
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0487 - accuracy: 0.9866 - val_loss: 0.0783 - val_accuracy: 0.9808
# Epoch 99/99
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0526 - accuracy: 0.9861 - val_loss: 0.0625 - val_accuracy: 0.9836
# ```

# In[18]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.95, 1])
plt.legend(loc='lower right')
plt.show()


# ## Prediction & Save probability for further study

# In[19]:


import re
postfix = "CNN1i"

for dirname, _, filenames in os.walk(model_dir):
    for filename in filenames:
        model.load_weights(os.path.join(dirname, filename))
        
        # get epochs number and create file name
        ep = re.search(r'\d+', filename).group()
        csv_file = "digit_recognizer_" + postfix + "_epochs" + ep + ".csv"
        pred_file = "prediction_" + postfix + "_epochs" + ep + ".csv"
        print("epochs=" + ep)
        
        # prediction
        prediction = model.predict_classes(test_data, verbose=0)
        output = pd.DataFrame({"ImageId" : np.arange(1, 28000+1), "Label":prediction})
        output.to_csv(os.path.join(data_dir, csv_file), index=False)
        
        # probability
        pred = model.predict_classes(train_data_x, verbose=0)
        pred_proba = model.predict_proba(train_data_x, verbose=0)

        pred_df = pd.DataFrame(pred, index=np.arange(train_data_len), columns=["Prediction"])
        pred_proba_df = pd.DataFrame(pred_proba, index=np.arange(train_data_len), columns=["p{}".format(i) for i in range(10)])

        output = pd.concat([pred_df, pred_proba_df], axis=1)
        output.to_csv(os.path.join(data_dir, pred_file), index=False)
        
print("Your submission was successfully saved!")


# ## Save history

# In[20]:


hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history_' + postfix + '.csv')
print("Your history was successfully saved!")


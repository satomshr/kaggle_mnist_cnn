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

datagen = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
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


# ## Create the convolutional base

# In[12]:


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

# In[13]:


model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


# ## Compile

# In[14]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ## Set file path of parameter data and callback

# In[15]:


model_dir = "./weights/"
data_dir = "./data/"

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

# In[16]:


# history = model.fit(train_data_x, train_data_y,
#                     validation_split=0.2,
#                     epochs=20,
#                     callbacks=[model_checkpoint_callback])

history = model.fit_generator(datagen.flow(X, y, batch_size=32),
                              steps_per_epoch=len(X)/32,
                              validation_data=(X_cv, y_cv),
                              epochs=60,
                              callbacks=[model_checkpoint_callback])

# history = model.fit(train_data_x, train_data_y, epochs=ep)


# ```
# Epoch 1/60
# 1050/1050 [==============================] - 15s 14ms/step - loss: 0.5664 - accuracy: 0.8135 - val_loss: 0.0921 - val_accuracy: 0.9733
# Epoch 2/60
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.1944 - accuracy: 0.9390 - val_loss: 0.0543 - val_accuracy: 0.9824
# Epoch 3/60
# 1050/1050 [==============================] - 16s 15ms/step - loss: 0.1417 - accuracy: 0.9555 - val_loss: 0.0475 - val_accuracy: 0.9852
# Epoch 4/60
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.1202 - accuracy: 0.9626 - val_loss: 0.0998 - val_accuracy: 0.9725
# Epoch 5/60
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.1053 - accuracy: 0.9679 - val_loss: 0.0679 - val_accuracy: 0.9807
# Epoch 6/60
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.0967 - accuracy: 0.9711 - val_loss: 0.0476 - val_accuracy: 0.9849
# Epoch 7/60
# 1050/1050 [==============================] - 15s 14ms/step - loss: 0.0926 - accuracy: 0.9717 - val_loss: 0.0417 - val_accuracy: 0.9900
# Epoch 8/60
# 1050/1050 [==============================] - 15s 14ms/step - loss: 0.0803 - accuracy: 0.9752 - val_loss: 0.0331 - val_accuracy: 0.9905
# Epoch 9/60
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0840 - accuracy: 0.9754 - val_loss: 0.0359 - val_accuracy: 0.9890
# Epoch 10/60
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0748 - accuracy: 0.9767 - val_loss: 0.0321 - val_accuracy: 0.9899
# Epoch 11/60
# 1050/1050 [==============================] - 15s 14ms/step - loss: 0.0722 - accuracy: 0.9782 - val_loss: 0.0331 - val_accuracy: 0.9895
# Epoch 12/60
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0665 - accuracy: 0.9799 - val_loss: 0.0345 - val_accuracy: 0.9899
# Epoch 13/60
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.0672 - accuracy: 0.9807 - val_loss: 0.0287 - val_accuracy: 0.9917
# Epoch 14/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0676 - accuracy: 0.9803 - val_loss: 0.0455 - val_accuracy: 0.9874
# Epoch 15/60
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.0649 - accuracy: 0.9807 - val_loss: 0.0299 - val_accuracy: 0.9917
# Epoch 16/60
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0669 - accuracy: 0.9799 - val_loss: 0.0395 - val_accuracy: 0.9888
# Epoch 17/60
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.0628 - accuracy: 0.9815 - val_loss: 0.0279 - val_accuracy: 0.9926
# Epoch 18/60
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0606 - accuracy: 0.9824 - val_loss: 0.0270 - val_accuracy: 0.9933
# Epoch 19/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0564 - accuracy: 0.9833 - val_loss: 0.0238 - val_accuracy: 0.9933
# Epoch 20/60
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.0590 - accuracy: 0.9826 - val_loss: 0.0371 - val_accuracy: 0.9911
# Epoch 21/60
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0587 - accuracy: 0.9828 - val_loss: 0.0314 - val_accuracy: 0.9911
# Epoch 22/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0564 - accuracy: 0.9837 - val_loss: 0.0261 - val_accuracy: 0.9931
# Epoch 23/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0541 - accuracy: 0.9845 - val_loss: 0.0302 - val_accuracy: 0.9925
# Epoch 24/60
# 1050/1050 [==============================] - 15s 14ms/step - loss: 0.0534 - accuracy: 0.9845 - val_loss: 0.0269 - val_accuracy: 0.9935
# Epoch 25/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0505 - accuracy: 0.9851 - val_loss: 0.0269 - val_accuracy: 0.9932
# Epoch 26/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0540 - accuracy: 0.9843 - val_loss: 0.0411 - val_accuracy: 0.9902
# Epoch 27/60
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0524 - accuracy: 0.9844 - val_loss: 0.0291 - val_accuracy: 0.9917
# Epoch 28/60
# 1050/1050 [==============================] - 12s 12ms/step - loss: 0.0528 - accuracy: 0.9847 - val_loss: 0.0393 - val_accuracy: 0.9899
# Epoch 29/60
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.0525 - accuracy: 0.9852 - val_loss: 0.0317 - val_accuracy: 0.9908
# Epoch 30/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0495 - accuracy: 0.9858 - val_loss: 0.0251 - val_accuracy: 0.9936
# Epoch 31/60
# 1050/1050 [==============================] - 12s 12ms/step - loss: 0.0462 - accuracy: 0.9863 - val_loss: 0.0333 - val_accuracy: 0.9924
# Epoch 32/60
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0507 - accuracy: 0.9847 - val_loss: 0.0311 - val_accuracy: 0.9936
# Epoch 33/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0489 - accuracy: 0.9859 - val_loss: 0.0281 - val_accuracy: 0.9926
# Epoch 34/60
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0466 - accuracy: 0.9864 - val_loss: 0.0419 - val_accuracy: 0.9883
# Epoch 35/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0462 - accuracy: 0.9864 - val_loss: 0.0310 - val_accuracy: 0.9925
# Epoch 36/60
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0459 - accuracy: 0.9867 - val_loss: 0.0344 - val_accuracy: 0.9921
# Epoch 37/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0468 - accuracy: 0.9859 - val_loss: 0.0305 - val_accuracy: 0.9930
# Epoch 38/60
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0489 - accuracy: 0.9858 - val_loss: 0.0306 - val_accuracy: 0.9918
# Epoch 39/60
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0451 - accuracy: 0.9864 - val_loss: 0.0261 - val_accuracy: 0.9940
# Epoch 40/60
# 1050/1050 [==============================] - 12s 12ms/step - loss: 0.0497 - accuracy: 0.9855 - val_loss: 0.0327 - val_accuracy: 0.9921
# Epoch 41/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0428 - accuracy: 0.9881 - val_loss: 0.0363 - val_accuracy: 0.9914
# Epoch 42/60
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0463 - accuracy: 0.9862 - val_loss: 0.0369 - val_accuracy: 0.9917
# Epoch 43/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0440 - accuracy: 0.9868 - val_loss: 0.0416 - val_accuracy: 0.9917
# Epoch 44/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0456 - accuracy: 0.9865 - val_loss: 0.0328 - val_accuracy: 0.9907
# Epoch 45/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0425 - accuracy: 0.9873 - val_loss: 0.0308 - val_accuracy: 0.9920
# Epoch 46/60
# 1050/1050 [==============================] - 12s 12ms/step - loss: 0.0449 - accuracy: 0.9866 - val_loss: 0.0316 - val_accuracy: 0.9924
# Epoch 47/60
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.0424 - accuracy: 0.9881 - val_loss: 0.0300 - val_accuracy: 0.9924
# Epoch 48/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0437 - accuracy: 0.9866 - val_loss: 0.0335 - val_accuracy: 0.9932
# Epoch 49/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0425 - accuracy: 0.9876 - val_loss: 0.0319 - val_accuracy: 0.9933
# Epoch 50/60
# 1050/1050 [==============================] - 12s 12ms/step - loss: 0.0441 - accuracy: 0.9876 - val_loss: 0.0263 - val_accuracy: 0.9945
# Epoch 51/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0423 - accuracy: 0.9883 - val_loss: 0.0238 - val_accuracy: 0.9935
# Epoch 52/60
# 1050/1050 [==============================] - 14s 13ms/step - loss: 0.0439 - accuracy: 0.9872 - val_loss: 0.0240 - val_accuracy: 0.9938
# Epoch 53/60
# 1050/1050 [==============================] - 12s 12ms/step - loss: 0.0428 - accuracy: 0.9872 - val_loss: 0.0268 - val_accuracy: 0.9945
# Epoch 54/60
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0449 - accuracy: 0.9878 - val_loss: 0.0251 - val_accuracy: 0.9942
# Epoch 55/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0442 - accuracy: 0.9878 - val_loss: 0.0302 - val_accuracy: 0.9943
# Epoch 56/60
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0420 - accuracy: 0.9880 - val_loss: 0.0351 - val_accuracy: 0.9932
# Epoch 57/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0427 - accuracy: 0.9883 - val_loss: 0.0400 - val_accuracy: 0.9917
# Epoch 58/60
# 1050/1050 [==============================] - 14s 14ms/step - loss: 0.0423 - accuracy: 0.9882 - val_loss: 0.0313 - val_accuracy: 0.9920
# Epoch 59/60
# 1050/1050 [==============================] - 13s 13ms/step - loss: 0.0403 - accuracy: 0.9886 - val_loss: 0.0370 - val_accuracy: 0.9914
# Epoch 60/60
# 1050/1050 [==============================] - 13s 12ms/step - loss: 0.0422 - accuracy: 0.9877 - val_loss: 0.0491 - val_accuracy: 0.9911
# ```

# In[17]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.95, 1])
plt.legend(loc='lower right')
plt.show()


# ## Prediction & Save probability for further study

# In[18]:


import re
postfix = "CNN1g"

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


# In[19]:


# output.to_csv('digit_recognizer_CNN1g_epochs{}.csv'.format(ep), index=False)
# print("Your submission was successfully saved!")


# ## Save probability for further study

# In[20]:


# pred = model.predict_classes(train_data_x, verbose=0)
# pred_proba = model.predict_proba(train_data_x, verbose=0)

# pred_df = pd.DataFrame(pred, index=np.arange(train_data_len), columns=["Prediction"])
# pred_proba_df = pd.DataFrame(pred_proba, index=np.arange(train_data_len), columns=["p{}".format(i) for i in range(10)])

# output = pd.concat([pred_df, pred_proba_df], axis=1)
# output.head()


# In[21]:


# output.to_csv("prediction_CNN1g_epochs{}.csv".format(ep), index=False)
# print("Your prediction was successfully saved!")

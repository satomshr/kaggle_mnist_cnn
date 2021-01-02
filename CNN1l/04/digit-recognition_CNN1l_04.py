#!/usr/bin/env python
# coding: utf-8

# # TensorFlow ; CNN
# - Create simple network according to TensorFlow tutorial ( https://www.tensorflow.org/tutorials/images/cnn?hl=ja )
# - Change kernel_size of first layer (from (3,3) to (5,5)), channels are increased
# - ImageDataGenerator is used for training data of each epochs
# - Learning rate is reduced using ReduceLROnPlateau
# - Tried BatchNormalization and Dropout, but not better than ReduceLROnPlateau (Combination of ReduceLROnPlateau, BatchNormalization, and Dropout might be better, but didn't tried it)
# - Results ; 0.99435 (epochs=33 when val_loss is smallest)

# In[ ]:


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

# In[ ]:


train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


train_data_len = len(train_data)
test_data_len = len(test_data)
print("Length of train_data ; {}".format(train_data_len))
print("Length of test_data ; {}".format(test_data_len))


# - Length of train_data ; 42000
# - Length of test_data ; 28000

# In[ ]:


train_data_y = train_data["label"]
train_data_x = train_data.drop(columns="label")
train_data_x.head()


# In[ ]:


print(type(train_data_x))
print(type(train_data_y))


# In[ ]:


train_data_x = train_data_x.astype('float64').values.reshape((train_data_len, 28, 28, 1))
test_data = test_data.astype('float64').values.reshape((test_data_len, 28, 28, 1))


# ## Set ImageDataGenerator & show examples

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.20,
                             height_shift_range=0.20,
                             shear_range=0.2,
                             zoom_range=0.2,
                             fill_mode='nearest')


# In[ ]:


from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(array_to_img(datagen.random_transform(train_data_x[i])), cmap='gray')
    plt.axis('off')
    
plt.show()


# ## Scaling and split data

# In[ ]:


train_data_x /= 255.0
test_data /= 255.0


# In[ ]:


from sklearn.model_selection import train_test_split
X, X_cv, y, y_cv = train_test_split(train_data_x, train_data_y, test_size=0.2, random_state=0, stratify=train_data_y)
# X, X_cv, y, y_cv = train_test_split(train_data_x, train_data_y, train_size=1024*33, random_state=0, stratify=train_data_y)

print("Train data size ; {}".format(len(X)))
print("Validation data size ; {}".format(len(X_cv)))

# There is a way to use 'validation_split' option in 'model.fit()'. But in this case, validation
# data is taken from the last part of the input data and not shuffled.
# So, here, I use 'sklearn.model_selection.train_test_split()'' to create validation data.


# ## Create the convolutional base

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()
# model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same', input_shape=(28, 28, 1)))
# model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same'))
# model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.summary()


# ## Add Dense layers on top

# In[ ]:


model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


# ## Optimizer

# In[ ]:


# from keras import optimizers
# adm_opt = optimizers.Adam(lr=0.004527, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# default ; lr=0.001
# https://keras.io/ja/optimizers/


# ## Compile

# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.compile(optimizer=adm_opt,
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


# ## Set file path of parameter data and callback

# In[ ]:


model_dir = "./weights_a/"
data_dir = "./data_a/"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

model_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(model_dir, "model-{epoch:02d}-{val_loss:.4f}.hdf5"),
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True)

reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.47,
                                       patience=5,
                                       min_lr=0.00005,
                                       verbose=1)

# default learning rate (lr) of adam is 0.001, so set min_lr = 0.0001.
# factor=0.47, so lr ; 0.001 -> 0.00047 -> 0.0002209 -> 0.000103823 -> 0.00004879681 -> 0.0000229345007 -> 0.00001077921532
# https://keras.io/ja/optimizers/
# https://keras.io/ja/callbacks/


# ## Fit and check

# In[ ]:


# history = model.fit(train_data_x, train_data_y, 
#                     validation_split=0.2,
#                     epochs=20,
#                     callbacks=[model_checkpoint_callback])

bs = 32 # batch_size
# cb = [model_checkpoint_callback]
cb = [model_checkpoint_callback, reduce_lr_callback]

history = model.fit_generator(datagen.flow(X, y, batch_size=bs),
                              steps_per_epoch=len(X)/bs,
                              validation_data=(X_cv, y_cv),
                              epochs=80,
                              callbacks=cb)


# In[ ]:


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.plot(history.history["accuracy"], label="accuracy")
ax1.plot(history.history["val_accuracy"], label="val_accuracy")
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_ylim([0.98, 1])
ax1.grid(True)
ax1.legend()

ax2.plot(history.history["loss"], label="loss")
ax2.plot(history.history["val_loss"], label="val_loss")
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_ylim([0, 0.1])
ax2.grid(True)
ax2.legend()
    
plt.show()


# ## Prediction & Save probability for further study

# In[ ]:


import re
postfix = "CNN1l"

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

# In[ ]:


hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history_' + postfix + '.csv', index=False)
print("Your history was successfully saved!")


#!/usr/bin/env python
# coding: utf-8

# check_prediction1.ipynb
# Check prediction and see what are good and what are wrong

# ## Read and check data
# - `train_data` ; train data
# - `pred_data` ; prediction by CNN (results of `pred_classes` and `pred_proba`)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data
train_data = pd.read_csv("../kaggle_mnist/data/train.csv")
pred_data = pd.read_csv("./prediction_CNN1e_epochs14.csv")

# Concat train_data and pred_data
df = pd.concat([train_data, pred_data], axis=1)

# check confusion matrix
from sklearn import metrics

co_mat = metrics.confusion_matrix(df["label"], df["Prediction"])
print(co_mat)

# [[4131    0    0    0    1    0    0    0    0    0]
#  [   0 4678    0    0    4    0    0    2    0    0]
#  [   0    1 4176    0    0    0    0    0    0    0]
#  [   0    0    1 4349    0    1    0    0    0    0]
#  [   0    0    0    0 4071    0    0    1    0    0]
#  [   0    0    0    3    0 3790    2    0    0    0]
#  [   2    0    0    0    1    2 4132    0    0    0]
#  [   0    2   11    0    1    0    0 4386    0    1]
#  [   0    0    0    0    0    0    0    0 4061    2]
#  [   0    0    0    1    8    1    0    0    0 4178]]

# Devide into matched and mismatched data
df_matched = df.query('label == Prediction')
df_mismatched = df.query('label != Prediction')

# Check matched data
# # Seperate data in each label
df_matched_ar = []

for i in range(10):
    tmp_ar = df_matched.query('label == {}'.format(i))
    df_matched_ar.append(tmp_ar.sort_values('p{}'.format(i), ascending=False))

# # high probability data
h_num = 10
w_num = 10

fig = plt.figure(figsize=(h_num, w_num))
fig.subplots_adjust(hspace=0.5)

for j in range(h_num):
    for i in range(w_num):
        ax = fig.add_subplot(h_num, w_num, i + j*w_num + 1, xticks=[], yticks=[], title=j)
        ax.imshow(df_matched_ar[j].iloc[i, 1:28*28+1].values.reshape((28, 28)), cmap='gray')

plt.show()

# # low probability data
h_num = 10
w_num = 12

fig = plt.figure(figsize=(h_num, w_num))
fig.subplots_adjust(hspace=0.5)

for j in range(h_num):
    for i in range(w_num):
        label_2nd = df_matched_ar[j].iloc[i, (-10):].nlargest(2).idxmin().replace('p', '')
        ax = fig.add_subplot(h_num, w_num, i + j*w_num + 1, xticks=[], yticks=[], title="{}, {}".format(j, label_2nd))
        ax.imshow(df_matched_ar[j].iloc[-(i+1), 1:28*28+1].values.reshape((28, 28)), cmap='gray')

plt.show()


# Check mismatched data
# # Seperate data in each label
df_mismatched_ar = []

for i in range(10):
    tmp_ar = df_mismatched.query('label == {}'.format(i))
    df_mismatched_ar.append(tmp_ar.sort_values('p{}'.format(i), ascending=False))

# # See each data
h_num = 10
w_num = 12

fig = plt.figure(figsize=(h_num, w_num))
fig.subplots_adjust(hspace=0.5)

for j in range(h_num):
    w_loop = w_num if w_num < len(df_mismatched_ar[j]) else len(df_mismatched_ar[j])
    for i in range(w_loop):
        pred_label = df_mismatched_ar[j].iloc[i, (-10):].idxmax().replace('p', '')
        ax = fig.add_subplot(h_num, w_num, i + j*w_num + 1, xticks=[], yticks=[], title="{}, {}".format(j, pred_label))
        ax.imshow(df_mismatched_ar[j].iloc[-(i+1), 1:28*28+1].values.reshape((28, 28)), cmap='gray')

plt.show()

for j in range(10):
    print("{} ; {} counts".format(j, len(df_mismatched_ar[j])))

# 0 ; 1 counts
# 1 ; 6 counts
# 2 ; 1 counts
# 3 ; 2 counts
# 4 ; 1 counts
# 5 ; 5 counts
# 6 ; 5 counts
# 7 ; 15 counts
# 8 ; 2 counts
# 9 ; 10 counts

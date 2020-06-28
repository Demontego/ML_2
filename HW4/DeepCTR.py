#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *


# In[2]:


data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data = pd.concat([data_train,data_test])
sparse_features = ['C' + str(i) for i in range(1, 16)]
dense_features = ['I' + str(i) for i in range(1, 3)]
data_train = []
data_test= []


# In[3]:


data[sparse_features] = data[sparse_features].fillna(-1, )
data[dense_features] = data[dense_features].fillna(0, )


# In[4]:


target = ['Label']
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])


# In[5]:


mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                          for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                          for feat in dense_features]
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(
    linear_feature_columns + dnn_feature_columns)

train, test = train_test_split(data, test_size=20317220, shuffle=False)

train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'


# In[6]:


data=[]
y=train[target].values
train=[]
test=[]


# In[8]:


model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
               task='binary',
               l2_reg_embedding=1e-4, device=device)
model.compile("adam", "binary_crossentropy",
              metrics=["binary_crossentropy"], )


# In[ ]:


model.fit(train_model_input, y,
          batch_size=2499146, epochs=20,  verbose=2)





pred = model.predict(test_model_input)




with open("deepCTR.csv", 'w') as f:
    f.write("Id,Click\n")
    for i,j in enumerate(pred):
        f.write(str(i+1)+","+str(j[0])+"\n")







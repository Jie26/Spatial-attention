#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import numpy as np


# In[3]:

# we get more than 1 sequence from each vehicle,in order to guaranteen the all test data is different(including no overlapping)
# we split the test every 100sequenece at a time

def datasplit(startpoint=0,random_sta=42):
    # startpoint [0,999] random_sta [0,100]
    #default 0/42
    x_1=np.load('6track_input_c.npy')
    y_1=np.load('6position_output.npy')
    x_2=np.load('4track_input_c.npy')
    y_2=np.load('4position_output.npy')
    x=np.concatenate((x_1,x_2),axis=0)
    y=np.concatenate((y_1,y_2),axis=0)
    x=x[:,:,:36]
    y=y[:,:,:2]

    x_test=np.zeros(shape=[32050,15,36])
    y_test=np.zeros(shape=[32050,26,2])

    j=startpoint
    k=0
    for i in range(290):
        j=j+1000
        x_test[k:k+100,:,:]=x[j:j+100,:,:]
        x=np.delete(x,np.arange(j,j+100),axis=0)
        y_test[k:k+100,:,:]=y[j:j+100,:,:]
        y=np.delete(y,np.arange(j,j+100),axis=0)
        j=j-100
        k=k+100
    x_test=x_test[:k,:,:]
    y_test=y_test[:k,:,:]
    
    #for validation we use train_test_split,the two datasets may have some sequence,which have some overlapping timestep
    x_train,  x_validation, y_train, y_validation= train_test_split(x, y, test_size=0.1,random_state = random_sta)
    
    np.save(r'./data_split/x_test.npy',x_test)
    np.save(r'./data_split/x_train.npy',x_train)
    np.save(r'./data_split/x_validation.npy',x_validation)
    np.save(r'./data_split/y_test.npy',y_test)
    np.save(r'./data_split/y_train.npy',y_train)
    np.save(r'./data_split/y_validation.npy',y_validation)
    print('data_split success')
    
    return 0
    
#datasplit() 


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn


# In[ ]:

def result_show():
    print('rmse reslt\n')
    true_tra=np.load(r'./result/true_tra.npy')
    lane_attention_pre=np.load(r'./result/lane_attn_predict_tra.npy')
    context_attention_pre=np.load(r'./result/context_attn_predict_tra.npy')
    soft_attention_pre=np.load(r'./result/soft_attn_predict_tra.npy')
    base_pre=np.load(r'./result/base_predict_tra.npy')

    a,b,c=lane_attention_pre.shape
    true_tra=true_tra.transpose(1,0,2)
    true_tra=true_tra[:,:b,:]

    true=torch.from_numpy(true_tra).float()
    lane=torch.from_numpy(lane_attention_pre).float()
    context=torch.from_numpy(context_attention_pre).float()
    soft=torch.from_numpy(soft_attention_pre).float()
    base=torch.from_numpy(base_pre).float()

    cre=nn.MSELoss()

    xloss=np.zeros(5)
    yloss=np.zeros(5)


    for i in range(5):
        xloss[i]=cre(lane[:5*(i+1),:,0],true[:5*(i+1),:,0])
        yloss[i]=cre(lane[:5*(i+1),:,1],true[:5*(i+1),:,1])
    loss=np.sqrt(xloss+yloss)
    longitudinal_loss=np.sqrt(xloss)
    lateral_loss=np.sqrt(yloss)
    print('lane attention\n')
    print('position loss as 1 seconds increment')
    print(loss)
    print('\n')
    print('longitudinal position loss')
    print(longitudinal_loss)
    print('\n')
    print('lateral position loss')
    print(lateral_loss)
    print('\n')
    print('----------------------------------------------------\n')
    print('\n')

    xloss=np.zeros(5)
    yloss=np.zeros(5)

    for i in range(5):
        xloss[i]=cre(context[:5*(i+1),:,0],true[:5*(i+1),:,0])
        yloss[i]=cre(context[:5*(i+1),:,1],true[:5*(i+1),:,1])
    loss=np.sqrt(xloss+yloss)
    longitudinal_loss=np.sqrt(xloss)
    lateral_loss=np.sqrt(yloss)
    print('context attention\n')
    print('position loss as 1 seconds increment')
    print(loss)
    print('\n')
    print('longitudinal position loss')
    print(longitudinal_loss)
    print('\n')
    print('lateral position loss')
    print(lateral_loss)
    print('\n')
    print('----------------------------------------------------\n')
    print('\n')

    xloss=np.zeros(5)
    yloss=np.zeros(5)

    for i in range(5):
        xloss[i]=cre(soft[:5*(i+1),:,0],true[:5*(i+1),:,0])
        yloss[i]=cre(soft[:5*(i+1),:,1],true[:5*(i+1),:,1])
    loss=np.sqrt(xloss+yloss)
    longitudinal_loss=np.sqrt(xloss)
    lateral_loss=np.sqrt(yloss)
    print('soft attention\n')
    print('position loss as 1 seconds increment')
    print(loss)
    print('\n')
    print('longitudinal position loss')
    print(longitudinal_loss)
    print('\n')
    print('lateral position loss\n')
    print(lateral_loss)
    print('\n')
    print('----------------------------------------------------\n')
    print('\n')


    xloss=np.zeros(5)
    yloss=np.zeros(5)

    for i in range(5):
        xloss[i]=cre(base[:5*(i+1),:,0],true[:5*(i+1),:,0])
        yloss[i]=cre(base[:5*(i+1),:,1],true[:5*(i+1),:,1])
    loss=np.sqrt(xloss+yloss)
    longitudinal_loss=np.sqrt(xloss)
    lateral_loss=np.sqrt(yloss)
    print('base line\n')
    print('position loss as 1 seconds increment')
    print(loss)
    print('\n')
    print('longitudinal position loss')
    
    print(longitudinal_loss)
    print('\n')
    print('lateral position loss')
    print(lateral_loss)


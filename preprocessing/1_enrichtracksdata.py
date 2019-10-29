#!/usr/bin/env python
# coding: utf-8

# # driving intention labeling
# keep lane 2/lane change to left 1/lane change to right 3

# In[5]:


import glob
import pandas as pd
import numpy as np
import os


# In[8]:


# read highD dataset
# tracks
#base_path1 = r'D:\TUM论文工作\第五周工作\数据\数据1'
base_path1 = r'/usr/stud/wangx0/storage/slurm/JieW/highd1/tracks'
files1 = os.listdir(base_path1)
files1.sort(key=lambda x: int(x.split('_')[0]))
dfs1=[]
for path1 in files1:
    full_path1 = os.path.join(base_path1, path1)
    dfs1.append(pd.read_csv(full_path1))
# tracks_Meta  
#base_path2 = r'D:\TUM论文工作\第五周工作\数据\数据2'
base_path2 = r'/usr/stud/wangx0/storage/slurm/JieW/highd1/tracksMeta'
files2 = os.listdir(base_path2)
files2.sort(key=lambda x: int(x.split('_')[0]))
dfs2=[]
for path2 in files2:
    full_path2 = os.path.join(base_path2, path2)
    dfs2.append(pd.read_csv(full_path2))


# In[53]:


# add drivingDirection and numLaneChanges to tracks.excel
numdata=60
for i,j in zip(range(numdata),range(numdata)):
    df1=dfs1[i]
    df2=dfs2[j]
    drmap1=df2.set_index('id').to_dict()['drivingDirection']
    df1['drivingDirection']=df1['id'].map(drmap1)
    drmap2=df2.set_index('id').to_dict()['numLaneChanges']
    df1['numLaneChanges']=df1['id'].map(drmap2)
    drmap3=df2.set_index('id').to_dict()['class']
    df1['class']=df1['id'].map(drmap3)
    drmap4=df2.set_index('id').to_dict()['numFrames']
    df1['numFrames']=df1['id'].map(drmap4)
    #df1.to_csv(r'D:\TUM论文工作\第五周工作\数据\数据4\%d_tracksnew.csv'%a[i])
    df1.to_csv(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/tracksenrich/%d_tracksenrich.csv'%(i+1))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import pandas as pd
import numpy as np
import os


# In[2]:


#read highD dataset
#tracks
base_path1 = r'/usr/stud/wangx0/storage/slurm/JieW/highd1/s1'
files1 = os.listdir(base_path1)
files1.sort(key=lambda x: int(x.split('_')[0]))
dfs1=[]
for path1 in files1:
    full_path1 = os.path.join(base_path1, path1)
    dfs1.append(pd.read_csv(full_path1))
#tracks_Meta  
base_path2 = r'/usr/stud/wangx0/storage/slurm/JieW/highd1/l1'
files2 = os.listdir(base_path2)
files2.sort(key=lambda x: int(x.split('_')[0]))
dfs2=[]
for path2 in files2:
    full_path2 = os.path.join(base_path2, path2)
    dfs2.append(pd.read_csv(full_path2))


# In[5]:


numdata=2
k=24
for i,j in zip(range(numdata),range(numdata)):
    df1=dfs1[i]
    df2=dfs2[j]
    df1['datataid']=k+1
    df1['label']=df2['label']
    df1.to_csv(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/trackfinal/%d_trackfinal.csv'%(k+1))
    k=k+1
    


# In[ ]:





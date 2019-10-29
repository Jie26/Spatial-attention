#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
import os


# In[ ]:


# for direction 2 lanenum_6
def carfill(line):
    if line.laneId==5:
        if line.precedingId==0:
            line.prexVelocity=line.xVelocity
            line.preyVelocity=line.yVelocity
            line.prex=line.x+200
            line.prey=line.y
        if line.followingId==0:
            line.folxVelocity=line.xVelocity
            line.folyVelocity=line.yVelocity
            line.folx=line.x-200
            line.foly=line.y
        if line.rightPrecedingId==0:
            line.rprxVelocity=line.xVelocity
            line.rpryVelocity=line.yVelocity
            line.rprx=line.x+200
            line.rpry=line.y
        if line.rightAlongsideId==0:
            line.ralxVelocity=line.xVelocity
            line.ralyVelocity=line.yVelocity
            line.ralx=line.x+150
            line.raly=line.y
        if line.rightFollowingId==0:
            line.rfoxVelocity=line.xVelocity
            line.rfoyVelocity=line.yVelocity
            line.rfox=line.x-200
            line.rfoy=line.y
    
    if line.laneId==6:
        if line.precedingId==0:
            line.prexVelocity=line.xVelocity
            line.preyVelocity=line.yVelocity
            line.prex=line.x+200
            line.prey=line.y
        if line.followingId==0:
            line.folxVelocity=line.xVelocity
            line.folyVelocity=line.yVelocity
            line.folx=line.x-200
            line.foly=line.y
        if line.leftPrecedingId==0:
            line.lprxVelocity=line.xVelocity
            line.lpryVelocity=line.yVelocity
            line.lprx=line.x+200
            line.lpry=line.y
        if line.leftAlongsideId==0:
            line.lalxVelocity=line.xVelocity
            line.lalyVelocity=line.yVelocity
            line.lalx=line.x+150
            line.laly=line.y
        if line.leftFollowingId==0:
            line.lfoxVelocity=line.xVelocity
            line.lfoyVelocity=line.yVelocity
            line.lfox=line.x-200
            line.lfoy=line.y


# In[ ]:


base_path1 = r'/usr/stud/wangx0/storage/slurm/JieW/highd1/non-exist4'
files1 = os.listdir(base_path1)
files1.sort(key=lambda x: int(x.split('_')[0]))
a=[1,2,3,15,16,17,18,19,20,21,22,23,24]
i=0
for path1 in files1:
    t1=time.process_time()
    full_path1 = os.path.join(base_path1, path1)
    df1=pd.read_csv(full_path1)
    del df1['Unnamed: 0']
    df1=df1.fillna(value=0)
    df1.apply(carfill,axis=1)
    t2=time.process_time()
    print(t2-t1)
    df1.to_csv(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/non-exist41/%d_track_nonexist.csv'%(a[i]))
    i=i+1


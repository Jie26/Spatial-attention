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
    if line.laneId==7:
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
    if line.laneId==8:
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


base_path1 = r'/usr/stud/wangx0/storage/slurm/JieW/highd1/l1'
files1 = os.listdir(base_path1)
files1.sort(key=lambda x: int(x.split('_')[0]))
b=[25,26]
a=[4,5,6,7,8,9,10,11,12,13,14,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]
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
    df1.to_csv(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/non-exist61/%d_track_nonexist.csv'%(b[i]))
    i=i+1
print('success')


# In[ ]:





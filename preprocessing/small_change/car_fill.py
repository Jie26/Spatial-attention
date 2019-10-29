#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import time


# In[2]:


def carfill6(line):
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
            line.rpry=line.y+5
        if line.rightAlongsideId==0:
            line.ralxVelocity=line.xVelocity
            line.ralyVelocity=line.yVelocity
            line.ralx=line.x+150
            line.raly=line.y+5
        if line.rightFollowingId==0:
            line.rfoxVelocity=line.xVelocity
            line.rfoyVelocity=line.yVelocity
            line.rfox=line.x-200
            line.rfoy=line.y+5
    return line.prexVelocity,line.preyVelocity,line.prex,line.prey,line.folxVelocity,line.folyVelocity,line.folx,line.foly,           line.rprxVelocity,line.rpryVelocity,line.rprx,line.rpry,line.ralxVelocity,line.ralyVelocity,line.ralx,line.raly,line.rfoxVelocity,           line.rfoyVelocity,line.rfox,line.rfoy
    


# In[3]:


def carfill7(line):
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
            line.rpry=line.y+5
        if line.rightAlongsideId==0:
            line.ralxVelocity=line.xVelocity
            line.ralyVelocity=line.yVelocity
            line.ralx=line.x+150
            line.raly=line.y+5
        if line.rightFollowingId==0:
            line.rfoxVelocity=line.xVelocity
            line.rfoyVelocity=line.yVelocity
            line.rfox=line.x-200
            line.rfoy=line.y+5
        if line.leftPrecedingId==0:
            line.lprxVelocity=line.xVelocity
            line.lpryVelocity=line.yVelocity
            line.lprx=line.x+200
            line.lpry=line.y-5
        if line.leftAlongsideId==0:
            line.lalxVelocity=line.xVelocity
            line.lalyVelocity=line.yVelocity
            line.lalx=line.x+150
            line.laly=line.y-5
        if line.leftFollowingId==0:
            line.lfoxVelocity=line.xVelocity
            line.lfoyVelocity=line.yVelocity
            line.lfox=line.x-200
            line.lfoy=line.y-5
    return line.prexVelocity,line.preyVelocity,line.prex,line.prey,line.folxVelocity,line.folyVelocity,line.folx,line.foly,        line.rprxVelocity,line.rpryVelocity,line.rprx,line.rpry,line.ralxVelocity,line.ralyVelocity,line.ralx,line.raly,line.rfoxVelocity,        line.rfoyVelocity,line.rfox,line.rfoy,line.lprxVelocity,line.lpryVelocity,line.lprx,line.lpry,line.lalxVelocity,line.lalyVelocity,        line.lalx,line.laly,line.lfoxVelocity,line.lfoyVelocity,line.lfox,line.lfoy


# In[4]:


def carfill8(line):
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
            line.lpry=line.y-5
        if line.leftAlongsideId==0:
            line.lalxVelocity=line.xVelocity
            line.lalyVelocity=line.yVelocity
            line.lalx=line.x+150
            line.laly=line.y-5
        if line.leftFollowingId==0:
            line.lfoxVelocity=line.xVelocity
            line.lfoyVelocity=line.yVelocity
            line.lfox=line.x-200
            line.lfoy=line.y-5
    return line.prexVelocity,line.preyVelocity,line.prex,line.prey,line.folxVelocity,line.folyVelocity,line.folx,line.foly,           line.lprxVelocity,line.lpryVelocity,line.lprx,line.lpry,line.lalxVelocity,line.lalyVelocity,           line.lalx,line.laly,line.lfoxVelocity,line.lfoyVelocity,line.lfox,line.lfoy


# In[ ]:


base_path1 = r'/usr/stud/wangx0/storage/slurm/JieW/highd1/non-exist6'
files1 = os.listdir(base_path1)
files1.sort(key=lambda x: int(x.split('_')[0]))
a=[4,5,6,7,8,9,10,11,12,13,14,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]
i=0


# In[ ]:


for path1 in files1:
    t1=time.process_time()
    full_path1 = os.path.join(base_path1, path1)
    df=pd.read_csv(full_path1)
    del df['Unnamed: 0']
    df=df.fillna(value=0)
    
    df1=df.apply(carfill6,axis=1)
    df2=pd.DataFrame(list(df1))
    df['prexVelocity']=df2[0]
    df['preyVelocity']=df2[1]
    df['prex']=df2[2]
    df['prey']=df2[3]

    df['folxVelocity']=df2[4]
    df['folyVelocity']=df2[5]
    df['folx']=df2[6]
    df['foly']=df2[7]

    df['rprxVelocity']=df2[8]
    df['rpryVelocity']=df2[9]
    df['rprx']=df2[10]
    df['rpry']=df2[11]

    df['ralxVelocity']=df2[12]
    df['ralyVelocity']=df2[13]
    df['ralx']=df2[14]
    df['raly']=df2[15]

    df['rfoxVelocity']=df2[16]
    df['rfoyVelocity']=df2[17]
    df['rfox']=df2[18]
    df['rfoy']=df2[19]
    
    
    df1=df.apply(carfill7,axis=1)
    df2=pd.DataFrame(list(df1))
    
    df['prexVelocity']=df2[0]
    df['preyVelocity']=df2[1]
    df['prex']=df2[2]
    df['prey']=df2[3]

    df['folxVelocity']=df2[4]
    df['folyVelocity']=df2[5]
    df['folx']=df2[6]
    df['foly']=df2[7]

    df['rprxVelocity']=df2[8]
    df['rpryVelocity']=df2[9]
    df['rprx']=df2[10]
    df['rpry']=df2[11]

    df['ralxVelocity']=df2[12]
    df['ralyVelocity']=df2[13]
    df['ralx']=df2[14]
    df['raly']=df2[15]

    df['rfoxVelocity']=df2[16]
    df['rfoyVelocity']=df2[17]
    df['rfox']=df2[18]
    df['rfoy']=df2[19]


    df['lprxVelocity']=df2[20]
    df['lpryVelocity']=df2[21]
    df['lprx']=df2[22]
    df['lpry']=df2[23]

    df['lalxVelocity']=df2[24]
    df['lalyVelocity']=df2[25]
    df['lalx']=df2[26]
    df['laly']=df2[27]

    df['lfoxVelocity']=df2[28]
    df['lfoyVelocity']=df2[29]
    df['lfox']=df2[30]
    df['lfoy']=df2[31]
    
    df1=df.apply(carfill8,axis=1)
    df2=pd.DataFrame(list(df1))
    
    df['prexVelocity']=df2[0]
    df['preyVelocity']=df2[1]
    df['prex']=df2[2]
    df['prey']=df2[3]

    df['folxVelocity']=df2[4]
    df['folyVelocity']=df2[5]
    df['folx']=df2[6]
    df['foly']=df2[7]

    df['lprxVelocity']=df2[8]
    df['lpryVelocity']=df2[9]
    df['lprx']=df2[10]
    df['lpry']=df2[11]

    df['lalxVelocity']=df2[12]
    df['lalyVelocity']=df2[13]
    df['lalx']=df2[14]
    df['laly']=df2[15]

    df['lfoxVelocity']=df2[16]
    df['lfoyVelocity']=df2[17]
    df['lfox']=df2[18]
    df['lfoy']=df2[19]
    
    
    t2=time.process_time()
    print(t2-t1)
    df.to_csv(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/7_non-exist61/%d_track_nonexist.csv'%(a[i]))
    i=i+1
print('success')


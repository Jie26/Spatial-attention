#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import pandas as pd
import numpy as np
import os
import time


# In[ ]:


#core function for labeling
def label(df1):
    #label for numchanges==1
    df1['label']=2
    d=df1[df1['numLaneChanges']==1]
    d1=d['id'].drop_duplicates()
    for i in d1:
        dfid=df1[df1['id']==i]
        #frame=abs(dfid['yVelocity']).argmax()
        frame=abs(dfid['yVelocity']).idxmax()
        m=0
        n=0
        while abs(df1.loc[frame-m,'yVelocity'])>gate_yveloctiy and df1.loc[frame-m,'id']== df1.loc[frame,'id']:
            m=m+1
            if frame-m<1:
                break
        while abs(df1.loc[frame+n,'yVelocity'])>gate_yveloctiy and df1.loc[frame+n,'id']== df1.loc[frame,'id']:
            n=n+1
        if df1.loc[frame,'drivingDirection']==1 and df1.loc[frame+n,'y']>df1.loc[frame-m,'y']:
            df1.loc[frame-m:frame+n,'label']=1
        if df1.loc[frame,'drivingDirection']==1 and df1.loc[frame+n,'y']<df1.loc[frame-m,'y']:
            df1.loc[frame-m:frame+n,'label']=3
        if df1.loc[frame,'drivingDirection']==2 and df1.loc[frame+n,'y']>df1.loc[frame-m,'y']:
            df1.loc[frame-m:frame+n,'label']=3
        if df1.loc[frame,'drivingDirection']==2 and df1.loc[frame+n,'y']<df1.loc[frame-m,'y']:
            df1.loc[frame-m:frame+n,'label']=1
    #label for numchanges>1
    f=df1[df1['numLaneChanges']>1]
    f1=f['id'].drop_duplicates()
    s=[]
    for j in f1:
        dfid=df1[df1['id']==j]
        for i in dfid.index:
            if i==dfid.index.max():
                break
            if dfid['laneId'][i]!=dfid['laneId'][i+1]:
                 s.append(i)
            for frame in s:
                m=0
                n=0
                while abs(df1.loc[frame-m,'yVelocity'])>gate_yveloctiy and df1.loc[frame-m,'id']== df1.loc[frame,'id']:
                    m=m+1
                while abs(df1.loc[frame+n,'yVelocity'])>gate_yveloctiy and df1.loc[frame+n,'id']== df1.loc[frame,'id']:
                    n=n+1
                if df1.loc[frame,'drivingDirection']==1 and df1.loc[frame+1,'y']>df1.loc[frame-1,'y']:
                    df1.loc[frame-m:frame+n,'label']=1
                if df1.loc[frame,'drivingDirection']==1 and df1.loc[frame+1,'y']<df1.loc[frame-1,'y']:
                    df1.loc[frame-m:frame+n,'label']=3
                if df1.loc[frame,'drivingDirection']==2 and df1.loc[frame+1,'y']>df1.loc[frame-1,'y']:
                    df1.loc[frame-m:frame+n,'label']=3
                if df1.loc[frame,'drivingDirection']==2 and df1.loc[frame+1,'y']<df1.loc[frame-1,'y']:
                    df1.loc[frame-m:frame+n,'label']=1


# In[ ]:


gate_yveloctiy=0.1
base_path1 = r'/usr/stud/wangx0/storage/slurm/JieW/highd1/tracksenrich1'
files1 = os.listdir(base_path1)
files1.sort(key=lambda x: int(x.split('_')[0]))
i=0
for path1 in files1:
    i=i+1
    t1=time.process_time()
    full_path1 = os.path.join(base_path1, path1)
    df4=pd.read_csv(full_path1)
    label(df4)
    df4.to_csv(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/trackslabel/%d_trackslabel.csv'%(i))
    
    t2=time.process_time()
    print(t2-t1)


# In[ ]:





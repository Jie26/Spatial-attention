#!/usr/bin/env python
# coding: utf-8

# In[1]

import pandas as pd
import numpy as np
import os
import time



def drivingscenereproduce(df1):
    num=df1.shape[0]
    for i in range(num):
        a=df1.at[i,'frame']
        b1=df1.at[i,'precedingId']
        b2=df1.at[i,'followingId']
        b3=df1.at[i,'leftPrecedingId']
        b4=df1.at[i,'leftAlongsideId']
        b5=df1.at[i, 'leftFollowingId']
        b6=df1.at[i, 'rightPrecedingId']
        b7=df1.at[i,'rightAlongsideId']
        b8=df1.at[i, 'rightFollowingId']
        if b1==0:
            pass
            #df1.at[i,['prex','prey','prexVelocity','preyVelocity','prexAcceleration','preyAcceleration']=0
        else:
            df2=df1[(df1['frame'] == a) & (df1['id'] == b1)]
       
            df2.index=[1]
            df1.at[i,'prex']=df2.at[1,'x']
            df1.at[i,'prey']=df2.at[1,'y']
            df1.at[i,'prexVelocity']=df2.at[1,'xVelocity']
            df1.at[i,'preyVelocity']=df2.at[1,'yVelocity']
            df1.at[i,'prexAcceleration']=df2.at[1,'xAcceleration']
            df1.at[i,'preyAcceleration']=df2.at[1,'yAcceleration']
        if b2==0:
            pass
            #df1.at[i,['prex','prey','prexVelocity','preyVelocity','prexAcceleration','preyAcceleration']=0
        else:
            df2=df1[(df1['frame'] == a) & (df1['id'] == b2)]
            
            df2.index=[1]
            df1.at[i,'folx']=df2.at[1,'x']
            df1.at[i,'foly']=df2.at[1,'y']
            df1.at[i,'folxVelocity']=df2.at[1,'xVelocity']
            df1.at[i,'folyVelocity']=df2.at[1,'yVelocity']
            df1.at[i,'folxAcceleration']=df2.at[1,'xAcceleration']
            df1.at[i,'folyAcceleration']=df2.at[1,'yAcceleration']   
        if b3==0:
            pass
            #df1.at[i,['prex','prey','prexVelocity','preyVelocity','prexAcceleration','preyAcceleration']=0
        else:
            df2=df1[(df1['frame'] == a) & (df1['id'] == b3)]
            df2.index=[1]
            df1.at[i,'lprx']=df2.at[1,'x']
            df1.at[i,'lpry']=df2.at[1,'y']
            df1.at[i,'lprxVelocity']=df2.at[1,'xVelocity']
            df1.at[i,'lpryVelocity']=df2.at[1,'yVelocity']
            df1.at[i,'lprxAcceleration']=df2.at[1,'xAcceleration']
            df1.at[i,'lpryAcceleration']=df2.at[1,'yAcceleration']  
        if b4==0:
            pass
            #df1.at[i,['prex','prey','prexVelocity','preyVelocity','prexAcceleration','preyAcceleration']=0
        else:
            df2=df1[(df1['frame'] == a) & (df1['id'] == b4)]
            df2.index=[1]
            df1.at[i,'lalx']=df2.at[1,'x']
            df1.at[i,'laly']=df2.at[1,'y']
            df1.at[i,'lalxVelocity']=df2.at[1,'xVelocity']
            df1.at[i,'lalyVelocity']=df2.at[1,'yVelocity']
            df1.at[i,'lalxAcceleration']=df2.at[1,'xAcceleration']
            df1.at[i,'lalyAcceleration']=df2.at[1,'yAcceleration']   
        if b5==0:
            pass
            #df1.at[i,['prex','prey','prexVelocity','preyVelocity','prexAcceleration','preyAcceleration']=0
        else:
            df2=df1[(df1['frame'] == a) & (df1['id'] == b5)]
            df2.index=[1]
            df1.at[i,'lfox']=df2.at[1,'x']
            df1.at[i,'lfoy']=df2.at[1,'y']
            df1.at[i,'lfoxVelocity']=df2.at[1,'xVelocity']
            df1.at[i,'lfoyVelocity']=df2.at[1,'yVelocity']
            df1.at[i,'lfoxAcceleration']=df2.at[1,'xAcceleration']
            df1.at[i,'lfoyAcceleration']=df2.at[1,'yAcceleration']   
        if b6==0:
            pass
            #df1.at[i,['prex','prey','prexVelocity','preyVelocity','prexAcceleration','preyAcceleration']=0
        else:
            df2=df1[(df1['frame'] == a) & (df1['id'] == b6)]
            df2.index=[1]
            df1.at[i,'rprx']=df2.at[1,'x']
            df1.at[i,'rpry']=df2.at[1,'y']
            df1.at[i,'rprxVelocity']=df2.at[1,'xVelocity']
            df1.at[i,'rpryVelocity']=df2.at[1,'yVelocity']
            df1.at[i,'rprxAcceleration']=df2.at[1,'xAcceleration']
            df1.at[i,'rpryAcceleration']=df2.at[1,'yAcceleration']
        if b7==0:
            pass
            #df1.at[i,['prex','prey','prexVelocity','preyVelocity','prexAcceleration','preyAcceleration']=0
        else:
            df2=df1[(df1['frame'] == a) & (df1['id'] == b7)]
            df2.index=[1]
            df1.at[i,'ralx']=df2.at[1,'x']
            df1.at[i,'raly']=df2.at[1,'y']
            df1.at[i,'ralxVelocity']=df2.at[1,'xVelocity']
            df1.at[i,'ralyVelocity']=df2.at[1,'yVelocity']
            df1.at[i,'ralxAcceleration']=df2.at[1,'xAcceleration']
            df1.at[i,'ralyAcceleration']=df2.at[1,'yAcceleration']
        if b8==0:
            pass
            #df1.at[i,['prex','prey','prexVelocity','preyVelocity','prexAcceleration','preyAcceleration']=0
        else:
            df2=df1[(df1['frame'] == a) & (df1['id'] == b8)]
            df2.index=[1]
            df1.at[i,'rfox']=df2.at[1,'x']
            df1.at[i,'rfoy']=df2.at[1,'y']
            df1.at[i,'rfoxVelocity']=df2.at[1,'xVelocity']
            df1.at[i,'rfoyVelocity']=df2.at[1,'yVelocity']
            df1.at[i,'rfoxAcceleration']=df2.at[1,'xAcceleration']
            df1.at[i,'rfoyAcceleration']=df2.at[1,'yAcceleration']
    
base_path = r'/usr/stud/wangx0/storage/slurm/JieW/highd1/tracksenrich/rich1'
files = os.listdir(base_path)
files.sort(key=lambda x: int(x.split('_')[0]))
i=0
for path in files:
    t1=time.process_time()
    full_path = os.path.join(base_path, path)
    df4=pd.read_csv(full_path)
    del df4['Unnamed: 0']
    drivingscenereproduce(df4)
    df4.to_csv(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/scenario/%d_trackscene.csv'%(i+1))
    t2=time.process_time()
    print('process time:',t2-t1)
    i=i+1




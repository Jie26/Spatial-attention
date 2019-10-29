#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import time


# In[ ]:


def nolanechange(df):
    global j
    global track_input
    global position_output
    global label_output
    
    df.index=np.arange(1,df.shape[0]+1)

    df=df[df.index%5==0]

    df=df[df['numFrames']>249]
    
    k=list(set(df.id.values))
    
    for i in k:
    
        dftemp=df[df['id']==i]
        dftemp.index=np.arange(1,dftemp.shape[0]+1)
        dftemp1=dftemp[['x', 'y', 'xVelocity','yVelocity',  'folx', 'foly', 'folxVelocity',
       'folyVelocity', 'ralx', 'raly','ralxVelocity', 'ralyVelocity',
       'rfox', 'rfoy', 'rfoxVelocity', 'rfoyVelocity', 'lalx', 'laly', 'lalxVelocity', 'lalyVelocity',
        'lfox', 'lfoy', 'lfoxVelocity','lfoyVelocity', 'lprx', 'lpry','lprxVelocity', 'lpryVelocity', 
       'prex', 'prey', 'prexVelocity', 'preyVelocity', 'rprx', 'rpry', 'rprxVelocity', 'rpryVelocity','id','datataid']]
        dftemp2=dftemp[['x', 'y','id','datataid']]
        dftemp3=dftemp[['label','id','datataid']]
        #print(dftemp1.size)
        track_input[j,:,:]=dftemp1.iloc[10:25].values
        position_output[j,:,:]=dftemp2.iloc[24:50].values
        label_output[j,:,:]=dftemp3.iloc[24:50].values
        j=j+1
        num=dftemp.shape[0]
        if num>=55:
            track_input[j,:,:]=dftemp1.iloc[15:30].values
            position_output[j,:,:]=dftemp2.iloc[29:55].values
            label_output[j,:,:]=dftemp3.iloc[29:55].values
            j=j+1
        if num>=60:
            track_input[j,:,:]=dftemp1.iloc[20:35].values
            position_output[j,:,:]=dftemp2.iloc[34:60].values
            label_output[j,:,:]=dftemp3.iloc[34:60].values
            j=j+1
        if num>=65:
            track_input[j,:,:]=dftemp1.iloc[25:40].values
            position_output[j,:,:]=dftemp2.iloc[39:65].values
            label_output[j,:,:]=dftemp3.iloc[39:65].values
            j=j+1
        if num>=70:
            track_input[j,:,:]=dftemp1.iloc[30:45].values
            position_output[j,:,:]=dftemp2.iloc[44:70].values
            label_output[j,:,:]=dftemp3.iloc[44:70].values
            j=j+1
        if num>=75:
            track_input[j,:,:]=dftemp1.iloc[35:50].values
            position_output[j,:,:]=dftemp2.iloc[49:75].values
            label_output[j,:,:]=dftemp3.iloc[49:75].values
            j=j+1
        if num>=80:
            track_input[j,:,:]=dftemp1.iloc[40:55].values
            position_output[j,:,:]=dftemp2.iloc[54:80].values
            label_output[j,:,:]=dftemp3.iloc[54:80].values
            j=j+1
        if num>=85:
            track_input[j,:,:]=dftemp1.iloc[45:60].values
            position_output[j,:,:]=dftemp2.iloc[59:85].values
            label_output[j,:,:]=dftemp3.iloc[59:85].values
            j=j+1
        if num>=90:
            track_input[j,:,:]=dftemp1.iloc[50:65].values
            position_output[j,:,:]=dftemp2.iloc[64:90].values
            label_output[j,:,:]=dftemp3.iloc[64:90].values
            j=j+1
        if num>=95:
            track_input[j,:,:]=dftemp1.iloc[55:70].values
            position_output[j,:,:]=dftemp2.iloc[69:95].values
            label_output[j,:,:]=dftemp3.iloc[69:95].values
            j=j+1
        if num>=100:
            track_input[j,:,:]=dftemp1.iloc[60:75].values
            position_output[j,:,:]=dftemp2.iloc[74:100].values
            label_output[j,:,:]=dftemp3.iloc[74:100].values
            j=j+1
        if num>=105:
            track_input[j,:,:]=dftemp1.iloc[65:80].values
            position_output[j,:,:]=dftemp2.iloc[79:105].values
            label_output[j,:,:]=dftemp3.iloc[79:105].values
            j=j+1


# In[ ]:


def lanechange(df):
    global j
    global track_input
    global position_output
    global label_output
    df.index=np.arange(1,df.shape[0]+1)

    for mode in range(5):
       
        dfmode=df[df.index%5==mode]

        dfmode=dfmode[dfmode['numFrames']>249]

        k=list(set(dfmode.id.values))
       
        for i in k:

            dftemp=dfmode[dfmode['id']==i]
        
            dftemp.index=np.arange(1,dftemp.shape[0]+1)
        
            dftemp1=dftemp[['x', 'y', 'xVelocity','yVelocity',  'folx', 'foly', 'folxVelocity',
       'folyVelocity', 'ralx', 'raly','ralxVelocity', 'ralyVelocity',
       'rfox', 'rfoy', 'rfoxVelocity', 'rfoyVelocity', 'lalx', 'laly', 'lalxVelocity', 'lalyVelocity',
        'lfox', 'lfoy', 'lfoxVelocity','lfoyVelocity', 'lprx', 'lpry','lprxVelocity', 'lpryVelocity', 
       'prex', 'prey', 'prexVelocity', 'preyVelocity', 'rprx', 'rpry', 'rprxVelocity', 'rpryVelocity','id','datataid']]
            dftemp2=dftemp[['x', 'y','id','datataid']]
            dftemp3=dftemp[['label','id','datataid']]
            track_input[j,:,:]=dftemp1.iloc[10:25].values
            position_output[j,:,:]=dftemp2.iloc[24:50].values
            label_output[j,:,:]=dftemp3.iloc[24:50].values
            j=j+1
            num=dftemp.shape[0]
            if num>=55:
                track_input[j,:,:]=dftemp1.iloc[15:30].values
                position_output[j,:,:]=dftemp2.iloc[29:55].values
                label_output[j,:,:]=dftemp3.iloc[29:55].values
                j=j+1
            if num>=60:
                track_input[j,:,:]=dftemp1.iloc[20:35].values
                position_output[j,:,:]=dftemp2.iloc[34:60].values
                label_output[j,:,:]=dftemp3.iloc[34:60].values
                j=j+1
            if num>=65:
                track_input[j,:,:]=dftemp1.iloc[25:40].values
                position_output[j,:,:]=dftemp2.iloc[39:65].values
                label_output[j,:,:]=dftemp3.iloc[39:65].values
                j=j+1
            if num>=70:
                track_input[j,:,:]=dftemp1.iloc[30:45].values
                position_output[j,:,:]=dftemp2.iloc[44:70].values
                label_output[j,:,:]=dftemp3.iloc[44:70].values
                j=j+1
            if num>=75:
                track_input[j,:,:]=dftemp1.iloc[35:50].values
                position_output[j,:,:]=dftemp2.iloc[49:75].values
                label_output[j,:,:]=dftemp3.iloc[49:75].values
                j=j+1
            if num>=80:
                track_input[j,:,:]=dftemp1.iloc[40:55].values
                position_output[j,:,:]=dftemp2.iloc[54:80].values
                label_output[j,:,:]=dftemp3.iloc[54:80].values
                j=j+1
            if num>=85:
                track_input[j,:,:]=dftemp1.iloc[45:60].values
                position_output[j,:,:]=dftemp2.iloc[59:85].values
                label_output[j,:,:]=dftemp3.iloc[59:85].values
                j=j+1
            if num>=90:
                track_input[j,:,:]=dftemp1.iloc[50:65].values
                position_output[j,:,:]=dftemp2.iloc[64:90].values
                label_output[j,:,:]=dftemp3.iloc[64:90].values
                j=j+1
            if num>=95:
                track_input[j,:,:]=dftemp1.iloc[55:70].values
                position_output[j,:,:]=dftemp2.iloc[69:95].values
                label_output[j,:,:]=dftemp3.iloc[69:95].values
                j=j+1
            if num>=100:
                track_input[j,:,:]=dftemp1.iloc[60:75].values
                position_output[j,:,:]=dftemp2.iloc[74:100].values
                label_output[j,:,:]=dftemp3.iloc[74:100].values
                j=j+1
            if num>=105:
                track_input[j,:,:]=dftemp1.iloc[65:80].values
                position_output[j,:,:]=dftemp2.iloc[79:105].values
                label_output[j,:,:]=dftemp3.iloc[79:105].values
                j=j+1


# In[ ]:


base_path1 = r'/usr/stud/wangx0/storage/slurm/JieW/highd1/non-exist41'
# base_path1 = r'D:\TUM论文工作\第十周工作\0612\6'
files1 = os.listdir(base_path1)
files1.sort(key=lambda x: int(x.split('_')[0]))
j=0
shape1=[3000000,15,38]
shape2=[3000000,26,4]
shape3=[3000000,26,3]
track_input=np.zeros(shape=shape1)
position_output=np.zeros(shape=shape2)
label_output=np.zeros(shape=shape3)


# In[ ]:


for path1 in files1:
      
    t1=time.process_time()
    full_path1 = os.path.join(base_path1, path1)
    df1=pd.read_csv(full_path1)
    del df1['Unnamed: 0']
    df2=df1[df1.drivingDirection==2]
    df3=df2[df2.numLaneChanges==0]
    df4=df2[df2.numLaneChanges>0]
    nolanechange(df3)
    lanechange(df4)
    t2=time.process_time()
    print(t2-t1)
print('j:',j)
track_input=track_input[:j,:,:]
position_output=position_output[:j,:,:]
label_output=label_output[:j,:,:]
np.save(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/data_sqe4/4track_input.npy',track_input)     
np.save(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/data_sqe4/4position_output.npy',position_output)
np.save(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/data_sqe4/4label_output.npy',label_output)
    


# In[ ]:





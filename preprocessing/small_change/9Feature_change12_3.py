#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


x_1=np.load(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/data_sqe5/6track_input.npy')
x_2=np.load(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/data_sqe6/4track_input.npy')


# In[2]:


# x_1=np.load(r'D:\TUM论文工作\data_sqe1\6track_input.npy')
# x_2=np.load(r'D:\TUM论文工作\data_sqe2\4track_input.npy')


# In[5]:


a,b,c=x_1.shape


# In[9]:


x_1=x_1.reshape(a*b,c)


# In[11]:


x=np.zeros([a*b,c])


# In[15]:


x[:,0:4]=x_1[:,0:4]


# In[ ]:


j=4


# In[ ]:


for i in range(8):
    x[:,j]=x_1[:,j]-x_1[:,0]
    j=j+1
    x[:,j]=x_1[:,j]-x_1[:,1]
    j=j+1
    x[:,j]=x_1[:,j]-x_1[:,2]
    j=j+1
    x[:,j]=x_1[:,j]-x_1[:,3]
    j=j+1
x[:,36:38]=x_1[:,36:38]


# In[ ]:


x=x.reshape(a,b,c)


# In[ ]:


np.save(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/data_sqe5/6track_input_c.npy',x)


# In[ ]:


a,b,c=x_2.shape
x_2=x_2.reshape(a*b,c)
x=np.zeros([a*b,c])
x[:,0:4]=x_2[:,0:4]
j=4
for i in range(8):
    x[:,j]=x_2[:,j]-x_2[:,0]
    j=j+1
    x[:,j]=x_2[:,j]-x_2[:,1]
    j=j+1
    x[:,j]=x_2[:,j]-x_2[:,2]
    j=j+1
    x[:,j]=x_2[:,j]-x_2[:,3]
    j=j+1
x[:,36:38]=x_2[:,36:38]
x=x.reshape(a,b,c)
np.save(r'/usr/stud/wangx0/storage/slurm/JieW/highd1/data_sqe6/4track_input_c.npy',x)


# In[ ]:





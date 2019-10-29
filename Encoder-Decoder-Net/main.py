#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from lane_attention import lane_attn
from context_attention import context_attn
from baseline import base_structure
from soft_attention import soft_attn
from data_split import datasplit
from RMSE_compare import result_show

# In[ ]:


datasplit(0,42)
print('---------------------------------------------------------')
base_structure()
print('---------------------------------------------------------')
soft_attn()
print('---------------------------------------------------------')
context_attn()
print('---------------------------------------------------------')
lane_attn()
print('---------------------------------------------------------')
result_show()

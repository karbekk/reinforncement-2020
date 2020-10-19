#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys


# In[2]:


sys.setrecursionlimit(10000)


# In[4]:


class trans(object):
    def __init__(self,N):
        self.N = N
    def startState(self):
        return 1
    def isend(self,state):
        return state == self.N
    
    def action(self,state):
        result = []
        if state+1<=self.N:
            result.append('walk')
        if state*2<=self.N:
            result.append('tran')   
        return result
    
    def succprob(self,state,action):
        result = []
        if action=='walk':
            result.append((state+1, 1., -1.))
        elif action =='tran':
            result.append((state*2, 0.5, -2.))
            result.append((state, 0.5, -2.))
        return result 
    def dis(self):
        return 1.
    def states(self):
        return range(1, self.N+1)
    
    


# In[5]:


mdp = trans(N=10)


# In[7]:


print(mdp.action(3))


# In[8]:


print(mdp.succprob(3,'walk'))


# In[ ]:





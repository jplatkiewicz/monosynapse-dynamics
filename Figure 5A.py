
# coding: utf-8

# # Biophysics code

# In[2]:


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from ccg import correlograms

def compthta(Rtemp,Ttemp,delta):
    Sobs = len(np.intersect1d(Rtemp,Ttemp))
    mxSpk = np.max(np.append(Rtemp,Ttemp))
    bEdges = np.arange(0,mxSpk,delta)
    bo = histc(Rtemp, bEdges) 
    refCounts = np.append(bo[0],0)
    w = np.floor(Ttemp/delta)
    w = w[w!=np.max(w)]
    Nr = refCounts[w.astype('int')]
    Nr = Nr[Nr!=0]/delta
    naive = Sobs - np.sum(Nr)
    thtahat = naive/(1-((1/len(Nr))*np.sum(Nr)))
    return thtahat

def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return [r, map_to_bins]

#------------------------------------------------------------------------------
# Load spike data
#------------------------------------------------------------------------------

train_ref = np.load('train_ref_static.npy')     # Collection of reference spike trains
train_targ = np.load('train_targ_static.npy')   # Collection of target spike trains
train_ref0 = np.load('train_ref0_static.npy')   # Collection of reference spike trains (no synapse)
train_targ0 = np.load('train_targ0_static.npy') # Collection of target spike trains (no synapse)
params = np.load('parameters.npy')
weight_value = np.load('weights.npy')
Ntrial = params[0]                              # Number of trials
duration = params[1]                            # Trial duration in (ms)
period = params[2]                              # Nonstationarity timescale in (ms)
Fs = params[3]
Nphase = params[4]
phase = duration/Nphase
gm = params[5]
g0 = params[6]

print(params)

train = np.append(train_ref,train_targ)
cell = np.int64(np.append(np.zeros(len(train_ref)),np.ones(len(train_targ))))

#------------------------------------------------------------------------------
# Analyze spike data
#------------------------------------------------------------------------------

# Measure the distribution of synchrony count before injection
synch_width = 1.*5
#--WITH SYNAPSE--
Tref = synch_width*np.floor(train_ref/synch_width)
lmax = 0.#lag[np.argmax(Craw[0,1])]
x = (train_targ-lmax)*(np.sign(train_targ-lmax)+1)/2.
x = x[np.nonzero(x)]
Ttarg = synch_width*np.floor(train_targ/synch_width)
Tsynch = np.array(list(set(Tref) & set(Ttarg)))
synch_count = np.bincount(np.int64(np.floor(Tsynch/(Ntrial*phase))),minlength=Nphase.astype('int'))
#--WITHOUT SYNAPSE--
Tref0 = synch_width*np.floor(train_ref0/synch_width)
lmax0 = 0.#lag[np.argmax(Craw0[0,1])]
x = (train_targ0-lmax0)*(np.sign(train_targ0-lmax0)+1)/2.
x = x[np.nonzero(x)]
Ttarg0 = synch_width*np.floor(x/synch_width)
Tsynch0 = np.array(list(set(Tref0) & set(Ttarg0)))
synch_count0 = np.bincount(np.int64(np.floor(Tsynch0/(Ntrial*phase))),minlength=Nphase.astype('int'))

# Check the optimal synchrony window 
#Nsynch = 100 
#synch_width_range = np.linspace(.1,10,Nsynch)
#s = np.zeros(Nsynch)
#s0 = np.zeros(Nsynch)
#lmax = 0.#lag[np.argmax(Craw[0,1])]
##print(latency/ms,lmax)
#lmax0 = 0.#lag[np.argmax(Craw0[0,1])]
#for k in range(Nsynch):
#    synch_width = synch_width_range[k]
#    #--WITH SYNAPSE--
#    Tref = synch_width*np.floor(train_ref/synch_width)
#    x = (train_targ-lmax)*(np.sign(train_targ-lmax)+1)/2.
#    x = x[np.nonzero(x)]
#    Ttarg = synch_width*np.floor(x/synch_width)
#    Tsynch = np.array(list(set(Tref) & set(Ttarg)))
#    synch_count = np.bincount(np.int64(np.floor(Tsynch/(Ntrial*phase))),minlength=Nphase)
#    s[k] = np.mean(synch_count)
#    #--WITHOUT SYNAPSE--
#    Tref0 = synch_width*np.floor(train_ref0/synch_width)
#    x = (train_targ0-lmax0)*(np.sign(train_targ0-lmax0)+1)/2.
#    x = x[np.nonzero(x)]
#    Ttarg0 = synch_width*np.floor(x/synch_width)
#    Tsynch0 = np.array(list(set(Tref0) & set(Ttarg0)))
#    synch_count0 = np.bincount(np.int64(np.floor(Tsynch0/(Ntrial*phase))),minlength=Nphase)
#    s0[k] = np.mean(synch_count0)
#plt.figure()
##plt.plot(synch_width_range,(np.amax(Craw[0,1])-np.amax(Craw0[0,1]))*np.ones(Nsynch),'--r')
#plt.plot(synch_width_range,s-s0,'--k')
#plt.figure()
##plt.plot(synch_width_range,np.amax(Craw[0,1])*np.ones(Nsynch),'--r')
##plt.plot(synch_width_range,np.amax(Craw0[0,1])*np.ones(Nsynch),'--g')
#plt.plot(synch_width_range,s,'o-k')
#plt.plot(synch_width_range,s0,'o-c')
#plt.show()    
#exit()

#print(synch_count,synch_count0)
#print(synch_count-synch_count0)

# Excess synchrony count unbiased estimation
delta = period
Ndelta = int(Ntrial*duration/delta)
count_ref = np.bincount(np.int64(np.floor(train_ref/delta)),minlength=Ndelta)
count_targ = np.bincount(np.int64(np.floor(train_targ/delta)),minlength=Ndelta)
count_synch = np.bincount(np.int64(np.floor(Tsynch/delta)),minlength=Ndelta)
Ndelta_phase = int(Ntrial*phase/delta)
RS_prod = np.sum(np.reshape(count_ref*count_synch,(Nphase.astype('int'),Ndelta_phase)),axis=1)
alpha = RS_prod/(delta*synch_count)  
RT_prod = np.sum(np.reshape(count_ref*count_targ,(Nphase.astype('int'),Ndelta_phase)),axis=1)
#alphaN = alpha[~np.isnan(alpha)]
#synch_countN = synch_count[~np.isnan(alpha)]
#RT_prodN = RT_prod[~np.isnan(alpha)]

# ==================Zach revisions =======================
#estimate = (delta*synch_count-RT_prod)/(delta*synch_count-RS_prod)*synch_count
estimate = []; synchOn = []
indT = np.digitize(train_targ%duration,np.arange(0,12000,1000)) - 1
indR = np.digitize(train_ref%duration,np.arange(0,12000,1000)) - 1
for k in range(int(len(np.unique(indT)))):
    temp = compthta(np.round(train_ref[indR==k]),np.round(train_targ[indT==k]),delta)
    estimate = np.append(estimate,temp)
    temp = len(np.intersect1d(np.round(train_ref[indR==k]),np.round(train_targ[indT==k])))
    synchOn = np.append(synchOn,temp)
# ==================Zach revisions =======================

synchOff = []
indT = np.digitize(train_targ0%duration,np.arange(0,12000,1000)) - 1
indR = np.digitize(train_ref0%duration,np.arange(0,12000,1000)) - 1
for k in range(int(len(np.unique(indT)))):
    temp = len(np.intersect1d(np.round(train_ref0[indR==k]),np.round(train_targ0[indT==k])))
    synchOff = np.append(synchOff,temp)

    

# Evaluate the true injected synchrony count by comparing conditions "synapse on" and "synapse off"
injected_true = (synch_count-synch_count0)#*1./synch_count
print(injected_true)

# Check the result
x = synchOn-synchOff
y = estimate
P = np.polyfit(x, y, 1)
xx = np.linspace(np.min(x),np.max(x),1e3)
yy = P[0]*xx + P[1]

FigEst = plt.figure(figsize=(7,5))
colG = np.array([128,129,132])/255
plt.plot(xx,yy,color=colG,linestyle='--')
plt.plot(synchOn-synchOff,estimate,'ok',color='k',markersize=12)
plt.xlabel('theta')
plt.ylabel('theta hat')
plt.box('off')

FigCon = plt.figure(figsize=(7,5))
y = g0*weight_value*(0-(-50))/120.*10**9
plt.plot(y,estimate,color='k',linewidth=.7)
plt.plot(y,estimate,'ok',color='k',markersize=12)
plt.xlabel('Normalized peak conductance')
plt.ylabel('theta hat')
plt.box('off')
plt.show()


# In[ ]:





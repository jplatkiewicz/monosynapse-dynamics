# coding: utf-8

'''
(1) Load a collection of spike train pairs.
(2) Inject synchronous spikes onto each pair of spike trains at random times.
(3) Estimate the amount of injected synchronies.
'''

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Define Functions
#------------------------------------------------------------------------------

def compthta(Rtemp,Ttemp,delta):
    Sobs = len(np.intersect1d(Rtemp,Ttemp))
    mxSpk = np.max(np.append(Rtemp,Ttemp))
    bEdges = np.arange(0,mxSpk,10)
    bo = histc(Rtemp, bEdges) 
    refCounts = np.append(bo[0],0)
    w = np.floor(Ttemp/delta)
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



def GenerateInject(train,Ntrial,duration,synch_width,inject_count):
    Nwidth = int(duration/synch_width)
    allwidths = np.arange(int(Ntrial*duration/synch_width))
    include_index = np.int64(np.floor(train0/synch_width))
    include_idx = list(set(include_index)) 
    mask = np.zeros(allwidths.shape,dtype=bool)
    mask[include_idx] = True
    wheretoinject = synch_width*allwidths[~mask]
    alreadythere = synch_width*allwidths[mask]
    widths = np.append(wheretoinject,alreadythere)
    tags = np.append(np.zeros(len(wheretoinject)),np.ones(len(alreadythere)))
    ind_sort = np.argsort(widths)
    widths = widths[ind_sort]
    tags = tags[ind_sort]
    widths = widths[:Ntrial*Nwidth]
    tags = tags[:Ntrial*Nwidth]
    widths = np.reshape(widths,(Ntrial,Nwidth))
    tags = np.reshape(tags,(Ntrial,Nwidth))
    ind_perm = np.int64(np.transpose(np.random.permutation(np.mgrid[:Nwidth,:Ntrial][0]))) 
    widths = widths[np.arange(np.shape(widths)[0])[:,np.newaxis],ind_perm]
    tags = tags[np.arange(np.shape(tags)[0])[:,np.newaxis],ind_perm]
    ind_sort = np.argsort(tags,axis=1)
    widths = widths[np.arange(np.shape(widths)[0])[:,np.newaxis],ind_sort]
    tags = tags[np.arange(np.shape(tags)[0])[:,np.newaxis],ind_sort]
    train_inject = np.ravel(widths[:,:inject_count])
    
    return train_inject

#------------------------------------------------------------------------------
# Load spike data
#------------------------------------------------------------------------------

train_ref0 = np.load('train_ref_range.npy')   # Collection of reference spike trains
train_targ0 = np.load('train_targ_range.npy') # Collection of target spike trains
params = np.load('parameters_range.npy')      
Ntrial = int(params[0])                       # Number of trials
duration = params[1]                     # Trial duration in (ms)
period = params[2]                       # Nonstationarity timescale in (ms)
Fs = params[3]

print(params)

train0 = np.append(train_ref0,train_targ0)
cell0 = np.int64(np.append(np.zeros(len(train_ref0)),np.ones(len(train_targ0))))

#------------------------------------------------------------------------------
# Inject synchronous spikes
#------------------------------------------------------------------------------

synch_width = 1.                                   # Width of synchrony window in (ms)
Tref0 = synch_width*np.floor(train_ref0/synch_width)  # Discretize reference trains
Ttarg0 = synch_width*np.floor(train_targ0/synch_width)# Discretize target trains
Tsynch0 = np.array(list(set(Tref0) & set(Ttarg0)))    # Compute reference-target synchronies' time
print(np.int64(np.floor(Tsynch0/duration)))
synch_count0 = np.bincount(np.int64(np.floor(Tsynch0/duration)),minlength=Ntrial) # Count number of synchronies per trial 

inject_count = int(np.amax(synch_count0)/3.)           # Amount of injected synchronies
print("# injected spikes/trial: ",inject_count)    
Nwidth = int(duration/synch_width)                 # Number of synchrony windows in
train_inject = GenerateInject(train0, Ntrial, duration, synch_width, inject_count)
train_ref = np.sort(np.append(train_ref0,train_inject))
train_targ = np.sort(np.append(train_targ0,train_inject)) 
train = np.append(train_ref,train_targ)
cell = np.int64(np.append(np.zeros(len(train_ref)),np.ones(len(train_targ))))

# Synchrony cout distribution after injection
synch_width = 1.
Tref = synch_width*np.floor(train_ref/synch_width)
Ttarg = synch_width*np.floor(train_targ/synch_width)
Tsynch = np.array(list(set(Tref) & set(Ttarg)))
synch_count = np.bincount(np.int64(np.floor(Tsynch/duration)),minlength=Ntrial)

# Predict the amount of injected synchrony
interval = period
Ninterval = int(duration/interval)
count_ref = np.bincount(np.int64(np.floor(train_ref/interval)),minlength=Ninterval*Ntrial)
count_targ = np.bincount(np.int64(np.floor(train_targ/interval)),minlength=Ninterval*Ntrial)
count_synch = np.bincount(np.int64(np.floor(Tsynch/interval)),minlength=Ninterval*Ntrial)
RS_prod = np.sum(np.reshape(count_ref*count_synch,(Ntrial,Ninterval)),axis=1)
RT_prod = np.sum(np.reshape(count_ref*count_targ,(Ntrial,Ninterval)),axis=1) 
injection_unbiased = (interval*synch_count-RT_prod)/(interval*synch_count-RS_prod)*synch_count
injection_naive = synch_count-RT_prod/interval

# Represent the estimated injected synchrony distribution
def compthta(Rtemp,Ttemp,delta):
    Sobs = len(np.intersect1d(Rtemp,Ttemp))
    mxSpk = np.max(np.append(Rtemp,Ttemp))
    bEdges = np.arange(0,mxSpk,10)
    bo = histc(Rtemp, bEdges) 
    refCounts = bo[0]
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


injection_unbiased = []
Dur = duration.astype('int')
S = []
for k in range(1000):
    TT = train_targ[((0 + k*Dur) < train_targ)*(train_targ < (Dur + k*Dur))]
    RR = train_ref[((0 + k*Dur) < train_ref)*(train_ref < (Dur + k*Dur))]
    injection_unbiased = np.append(injection_unbiased,compthta(np.round(RR),np.round(TT),interval))
    
FigTh = plt.figure()
plt.xlabel('Injected synchrony estimate',fontsize=18)
plt.ylabel('Normalized count',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
bins = np.arange(np.amin(injection_naive),np.amax(injection_naive),1)
count,base = np.histogram(injection_naive,bins=bins,density=1)
X = base[:-1]
plt.bar(X,count,width=.5,align='center',color='c',edgecolor='c')
plt.plot(np.mean(injection_naive)*np.ones(2),[0,np.amax(count)],'--c')
bins = np.arange(np.amin(injection_unbiased),np.amax(injection_unbiased),1)
count,base = np.histogram(injection_unbiased,bins=bins,density=1)
X = base[:-1]
plt.bar(X,count,width=.25,align='center',color='None',edgecolor='k',linewidth=2)
plt.plot(np.mean(injection_unbiased)*np.ones(2),[0,np.amax(count)],'--k')
plt.plot(inject_count*np.ones(2),[0,np.amax(count)],'--r')
plt.show()



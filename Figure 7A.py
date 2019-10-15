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
    if delta < 1e6:
        Sobs = len(np.intersect1d(Rtemp,Ttemp))
        mxSpk = np.max(np.append(Rtemp,Ttemp))
        bEdges = np.arange(0,mxSpk,delta)
        bo = histc(Rtemp, bEdges) 
        refCounts = np.append(bo[0],0)
        w = np.floor(Ttemp/delta)
        Nr = refCounts[w.astype('int')]
        Nr = Nr[Nr!=0]/delta
        naive = Sobs - np.sum(Nr)
        thtahat = naive/(1-((1/len(Nr))*np.sum(Nr)))
    else:
        Sobs = len(np.intersect1d(Rtemp,Ttemp))
        mxSpk = np.max(np.append(Rtemp,Ttemp))
        Nr = np.ones(len(Ttemp))*(len(Rtemp))
        Nr = Nr/mxSpk
        naive = Sobs - np.sum(Nr)
        thtahat = naive/(1-((1/len(Nr))*np.sum(Nr)))
    return thtahat

def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return [r, map_to_bins]

def GenerateInject(train,duration,synch_width,inject_count):
    Nwidth = int(duration/synch_width)
    allwidths = np.arange(Nwidth)
    include_index = np.int64(np.floor(train/synch_width))
    include_idx = list(set(include_index)) 
    mask = np.zeros(allwidths.shape,dtype=bool)
    mask[include_idx] = True
    wheretoinject = synch_width*allwidths[~mask]
    alreadythere = synch_width*allwidths[mask]
    widths = np.append(wheretoinject,alreadythere)
    tags = np.int64(np.append(np.zeros(len(wheretoinject)), np.ones(len(alreadythere))))
    ind_sort = np.argsort(widths)
    widths = widths[ind_sort]
    tags = tags[ind_sort]
    ind_perm = np.int64(np.random.permutation(widths))
    widths = widths[ind_perm]
    tags = tags[ind_perm]
    ind_sort = np.argsort(tags)
    widths = widths[ind_sort]
    tags = tags[ind_sort]    
    train_inject = np.ravel(widths[:inject_count])
    
    return train_inject

#------------------------------------------------------------------------------
# Load spike data
#------------------------------------------------------------------------------

train_ref0 = np.load('train_ref_range.npy')   # Collection of reference spike trains
train_targ0 = np.load('train_targ_range.npy') # Collection of target spike trains
params = np.load('parameters_range.npy')
Ntrial = params[0]                       # Number of trials
duration = params[1]                     # Trial duration in (ms)
period = params[2]                       # Nonstationarity timescale in (ms)
Fs = params[3]

print(params)

#------------------------------------------------------------------------------
# Inject synchronous spikes
#------------------------------------------------------------------------------

synch_width = 1.                                   # Width of synchrony window in (ms)
nb_value = 10
inject_range = np.int64(np.linspace(0,1000,nb_value))
duration_new = int(Ntrial*duration/nb_value)
Delta_range = np.array([period/4.,period/2.,period,period*2,period*4,10*period,duration_new])
nb_Delta = len(Delta_range)
nb_ref0 = np.bincount(np.int64(np.floor(train_ref0/duration_new)))   
nb_targ0 = np.bincount(np.int64(np.floor(train_targ0/duration_new)))  
i_ref0 = np.append(np.zeros(1),np.cumsum(nb_ref0))
i_targ0 = np.append(np.zeros(1),np.cumsum(nb_targ0))
injection_unbiased = np.zeros((nb_Delta,nb_value))
for j in range(nb_Delta):
    interval = Delta_range[j]
    Ninterval = int(duration_new/interval)
    i_ref0 = i_ref0.astype('int')
    i_targ0 = i_targ0.astype('int')
    for k in range(nb_value):
        Tref0 = train_ref0[i_ref0[k]:i_ref0[k+1]]-k*duration_new
        Ttarg0 = train_targ0[i_targ0[k]:i_targ0[k+1]]-k*duration_new
        T0 = np.append(Tref0,Ttarg0)
        G0 = np.int64(np.append(np.zeros(len(Tref0)), np.ones(len(Ttarg0))))
        Tref0_s = synch_width*np.floor(Tref0/synch_width)  # Discretize reference trains
        Ttarg0_s = synch_width*np.floor(Ttarg0/synch_width)# Discretize target trains
        Tsynch0 = np.array(list(set(Tref0_s) & set(Ttarg0_s)))    # Compute reference-target synchronies' time
        synch_count0 = len(Tsynch0) # Count number of synchronies per trial    
        inject_count = inject_range[k]           # Amount of injected synchronies
        Tinj = GenerateInject(T0,duration_new,synch_width,inject_count)
        Tref = np.sort(np.append(Tref0,Tinj))
        Ttarg = np.sort(np.append(Ttarg0,Tinj))
        T = np.append(Tref,Ttarg)
        G = np.int64(np.append(np.zeros(len(Tref)), np.ones(len(Ttarg))))

        # Synchrony cout distribution after injection
        Tref_s = synch_width*np.floor(Tref/synch_width)
        Ttarg_s = synch_width*np.floor(Ttarg/synch_width)
        # Check if multiple spikes at the same time
        _,n = np.unique(Tref_s,return_counts=True)
        indr = np.nonzero(n-np.ones(len(n)))[0]
        _,n = np.unique(Ttarg_s,return_counts=True)
        indt = len(np.nonzero(n-np.ones(len(n)))[0])
        Tsynch = np.array(list(set(Tref_s) & set(Ttarg_s)))
        synch_count = len(Tsynch)

        # Predict the amount of injected synchrony
        count_ref = np.bincount(np.int64(np.floor(Tref/interval)), minlength=Ninterval)
        count_targ = np.bincount(np.int64(np.floor(Ttarg/interval)), minlength=Ninterval)
        count_synch = np.bincount(np.int64(np.floor(Tsynch/interval)), minlength=Ninterval)
        RS_prod = np.sum(count_ref*count_synch)
        RT_prod = np.sum(count_ref*count_targ) 
        
        injection_unbiased[j,k] = compthta(np.round(Tref_s),np.round(Ttarg_s),interval)
    
# Represent the estimated injected synchrony distribution
cm = plt.get_cmap('gist_rainbow')
FigTh = plt.figure(figsize=(7,6))
ax = FigTh.add_subplot(111)
ax.set_color_cycle([cm(1.*i/nb_Delta) for i in range(nb_Delta)])
plt.xlabel('True Injected Synchrony Count',fontsize=18)
plt.ylabel('Estimate',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.plot(inject_range,inject_range,'-k')
plt.box('off')

for j in range(nb_Delta):
    ax.plot(inject_range,injection_unbiased[j,:],'o--',markeredgecolor='None')
FigLeg = plt.figure()
ax = FigLeg.add_subplot(111)
ax.set_color_cycle([cm(1.*i/nb_Delta) for i in range(nb_Delta)])
plt.show()



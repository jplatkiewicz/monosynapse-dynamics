'''
(1) Load a collection of spike train pairs.
(2) Inject synchronous spikes onto each pair of spike trains at random times.
(3) Plot the resulting CCG.
'''

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from ccg import correlograms

#------------------------------------------------------------------------------
# Define Functions
#------------------------------------------------------------------------------

def GenerateInject(train,duration,synch_width,inject_count):
    Nwidth = int(duration/synch_width)
    allwidths = np.arange(Nwidth)
    include_index = np.int64(np.floor(train/synch_width))
    include_idx = list(set(include_index)) 
    mask = np.zeros(allwidths.shape,dtype=bool)
    mask[include_idx] = True
    wheretoinject = synch_width*allwidths[~mask]
    #-- Prevent the injected spikes to be too close
    print('a', np.amin(np.diff(wheretoinject)), np.mean(np.diff(wheretoinject)), np.amax(np.diff(wheretoinject)))
    wheretoinject = wheretoinject[::500]
    print('b',np.amin(np.diff(wheretoinject)), np.mean(np.diff(wheretoinject)),np.amax(np.diff(wheretoinject)))
    #ind_keep = np.where(np.diff(wheretoinject) > 30)[0]
    #wheretoinject = wheretoinject[ind_keep]
    #print('c', np.amin(np.diff(wheretoinject)), np.mean(np.diff(wheretoinject)), np.amax(np.diff(wheretoinject)))
    #--
    alreadythere = synch_width*allwidths[mask]
    widths = np.append(wheretoinject,alreadythere)
    tags = np.int64(np.append(np.zeros(len(wheretoinject)), np.ones(len(alreadythere))))
    ind_sort = np.argsort(widths)
    widths = widths[ind_sort]
    tags = tags[ind_sort]
    #-- Prevent the injected spikes to be too close
    #print('1',np.amin(np.diff(widths)), np.mean(np.diff(widths)),np.amax(np.diff(widths)))
    #widths = widths[::1000]
    #print('2',np.amin(np.diff(widths)), np.mean(np.diff(widths)),np.amax(np.diff(widths)))
    #ind_keep = np.where(np.diff(widths) > 30)[0]
    #widths = widths[ind_keep]
    #print('3',np.amin(np.diff(widths)), np.mean(np.diff(widths)),np.amax(np.diff(widths)))
    #--
    ind_perm = np.int64(np.random.permutation(len(widths)))
    widths = widths[ind_perm]
    tags = tags[ind_perm]
    ind_sort = np.argsort(tags)
    widths = widths[ind_sort]
    tags = tags[ind_sort]    
    train_inject = np.ravel(widths[:inject_count])
    
    return train_inject

def PoissonProcess(intensity,duration,N):
    count = np.random.poisson(lam=intensity,size=N)
    u = np.random.uniform(0,1,size=np.sum(count))
    c = np.int64(np.append(np.zeros(1),np.cumsum(count)))
    T = []
    tag = []
    ind_pos = np.nonzero(count)[0]
    for i in ind_pos:
        T.extend(duration*np.sort(u[c[i]:c[i+1]]))
        tag.extend(i*np.ones(count[i]))
    T = np.asarray(T)
    tag = np.asarray(tag)
                 
    return T,tag

#------------------------------------------------------------------------------
# Load spike data
#------------------------------------------------------------------------------

train_ref0 = np.load('train_ref_range.npy')   # Collection of reference spike trains
train_targ0 = np.load('train_targ_range.npy') # Collection of target spike trains
params = np.load('parameters_range.npy')          
Ntrial = int(params[0])                  # Number of trials
duration = params[1]                     # Trial duration in (ms)
period = params[2]                       # Nonstationarity timescale in (ms)
Fs = params[3]

print(params)

#------------------------------------------------------------------------------
# Inject synchronous spikes
#------------------------------------------------------------------------------

synch_width = 1.                           # Width of synchrony window in (ms)
nb_value = 10
inject_range = np.int64(np.linspace(0,1000,nb_value))
print(inject_range)
duration_new = int(Ntrial*duration/nb_value)
print(duration_new)

Nwidth = int(duration/synch_width)         # Number of synchrony windows in 
interval = period
Ninterval = int(duration_new/interval)
nb_ref0 = np.bincount(np.int64(np.floor(train_ref0/duration_new)))   
nb_targ0 = np.bincount(np.int64(np.floor(train_targ0/duration_new)))  
i_ref0 = np.int64(np.append(np.zeros(1),np.cumsum(nb_ref0)))
i_targ0 = np.int64(np.append(np.zeros(1),np.cumsum(nb_targ0)))
k = 1
Tref0 = train_ref0[i_ref0[k]:i_ref0[k+1]]-k*duration_new
Ttarg0 = train_targ0[i_targ0[k]:i_targ0[k+1]]-k*duration_new
T0 = np.append(Tref0,Ttarg0)
G0 = np.int64(np.append(np.zeros(len(Tref0)), np.ones(len(Ttarg0))))
Tref0_s = synch_width*np.floor(Tref0/synch_width)  # Discretize reference trains
Ttarg0_s = synch_width*np.floor(Ttarg0/synch_width)# Discretize target trains
Tsynch0 = np.array(list(set(Tref0_s) & set(Ttarg0_s)))    # Compute reference-target synchronies' time
synch_count0 = len(Tsynch0) # Count number of synchronies per trial    
#s[k] = synch_count0
inject_count = inject_range[k]           # Amount of injected synchronies
print(k,inject_count,len(Tref0),len(Ttarg0),synch_count0)
Tinj = GenerateInject(T0,duration_new,synch_width,inject_count)
Tref = np.sort(np.append(Tref0,Tinj))
#-For figure
Ttarg = np.sort(np.append(Ttarg0,Tinj+1.5))
T = np.append(Tref,Ttarg)
G = np.int64(np.append(np.zeros(len(Tref)), np.ones(len(Ttarg))))

# Check whether the injection has been well performed
lagmax = 50.
bine = .1
ind_sort = np.argsort(T0)
st = T0[ind_sort]*.001
sc = G0[ind_sort]
C0 = correlograms(st,sc,sample_rate=Fs,bin_size=bine*10/1000.,window_size=lagmax/1000.)    
lag0 = (np.arange(len(C0[0,1]))-len(C0[0,1])/2)*bine*10
ind_sort = np.argsort(T)
st = T[ind_sort]*.001
sc = G[ind_sort]
C = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(C[0,1]))-len(C[0,1])/2)*bine
FigCCGi = plt.figure()
plt.xlim(-lagmax/2.,lagmax/2.)
plt.title('Cross-correlogram',fontsize=18)
plt.xlabel('Time lag  (ms)',fontsize=18)
plt.ylabel('Firing rate (Hz)',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.bar(lag, C[0,1]/(len(Tref)*bine*.001), bine, align='center', color='k', edgecolor='k', linewidth=2)
#plt.bar(lag, C0[0,1]/(len(Tref)*bine*.001), bine, align='center', color='None', edgecolor='k', linewidth=2)
#plt.plot(lag,C[0,1]/(len(Tref)*bine*.001),'.-k')
plt.plot(lag0,C0[0,1]/(len(Tref0)*bine*10*.001),'--c')
FigCCGi.savefig('Fig_CCG-injected.eps')

plt.show()
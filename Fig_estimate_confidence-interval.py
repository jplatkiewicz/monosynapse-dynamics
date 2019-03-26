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
from ccg import correlograms
import time
#from itertools import permutations

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
    train_inject = widths[:inject_count].flatten()
    
    return train_inject

#------------------------------------------------------------------------------
# Load spike data
#------------------------------------------------------------------------------

#train_ref0 = np.load('train_ref.npy')   
#train_targ0 = np.load('train_targ.npy') 
#params = np.load('parameters.npy')  
train_ref0 = np.load('train_ref_range.npy')   # Collection of reference spike trains
train_targ0 = np.load('train_targ_range.npy') # Collection of target spike trains
params = np.load('parameters_range.npy')          
#train_ref0 = np.load('train_ref_test.npy')   
#train_targ0 = np.load('train_targ_test.npy') 
#params = np.load('parameters_test.npy')          
Ntrial = params[0]                       # Number of trials
duration = params[1]                     # Trial duration in (ms)
period = params[2]                       # Nonstationarity timescale in (ms)
Fs = params[3]

print(params)

#------------------------------------------------------------------------------
# Inject synchronous spikes
#------------------------------------------------------------------------------

synch_width = 1.                                   # Width of synchrony window in (ms)
inject_count = 50

# Jitter a number of times the hypothetical background spikes
Nexper = 100#0
Nsurr = 110#0
duration_new = Ntrial*duration/Nexper
print(duration_new)
interval = period
Ninterval = int(duration_new/interval)
nb_ref0 = np.bincount(np.int64(np.floor(train_ref0/duration_new)))   
nb_targ0 = np.bincount(np.int64(np.floor(train_targ0/duration_new)))  
i_ref0 = np.int64(np.append(np.zeros(1),np.cumsum(nb_ref0)))
i_targ0 = np.int64(np.append(np.zeros(1),np.cumsum(nb_targ0)))
alpha = 5.
CI_low = np.zeros(Nexper)
CI_up = np.zeros(Nexper)
inject_estim = np.zeros(Nexper)
s = np.zeros(Nexper)
for k in range(Nexper):
    Tref0 = train_ref0[i_ref0[k]:i_ref0[k+1]]-k*duration_new
    Ttarg0 = train_targ0[i_targ0[k]:i_targ0[k+1]]-k*duration_new  
    T0 = np.append(Tref0,Ttarg0)
    Tref0_s = synch_width*np.floor(Tref0/synch_width)  
    Ttarg0_s = synch_width*np.floor(Ttarg0/synch_width)
    Tsynch0 = np.array(list(set(Tref0_s) & set(Ttarg0_s)))    
    synch_count0 = len(Tsynch0)
    print(k,synch_count0)
    Tinject = GenerateInject(T0, duration_new, synch_width, inject_count)
    train_ref = np.sort(np.append(Tref0,Tinject))
    train_targ = np.sort(np.append(Ttarg0,Tinject)) 
    # Synchrony cout distribution after injection
    Tref = synch_width*np.floor(train_ref/synch_width)
    Ttarg = synch_width*np.floor(train_targ/synch_width)
    train_synch = np.array(list(set(Tref) & set(Ttarg)))
    synch_count = len(train_synch)
    s[k] = np.amin(synch_count)
    Nsynch = len(train_synch)    
    start_time = time.time()
    #ind_synchR  = dict((value, idx) for idx,value in enumerate(Tref))
    #[ind_synchR[x] for x in train_synch]
    #ind_synch = dict((value, idx) for idx,value in enumerate(Ttarg))
    #[ind_synch[x] for x in train_synch]    
    #print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    ind_synchR = np.zeros(len(train_synch),dtype=np.int64)
    ind_synch = np.zeros(len(train_synch),dtype=np.int64)
    for j,t in enumerate(train_synch):
        ind_synchR[j] = np.where(Tref-t == 0)[0]
        ind_synch[j] = np.where(Ttarg-t == 0)[0]
    print("--- %s seconds ---" % (time.time() - start_time))    
    mask = np.zeros(len(train_ref),dtype=bool)
    mask[ind_synchR] = True
    Tref_nosynch = train_ref[~mask]
    mask = np.zeros(len(train_targ),dtype=bool)
    mask[ind_synch] = True
    Ttarg_nosynch = train_targ[~mask]    
    # Predict the amount of injected synchrony
    count_ref = np.bincount(np.int64(np.floor(train_ref/interval)), minlength=Ninterval)
    count_targ = np.bincount(np.int64(np.floor(train_targ/interval)), minlength=Ninterval)
    count_synch = np.bincount(np.int64(np.floor(train_synch/interval)), minlength=Ninterval)
    RS_prod = np.sum(count_ref*count_synch)
    RT_prod = np.sum(count_ref*count_targ) 
    theta_hat = int((interval*synch_count-RT_prod)/(interval*synch_count-RS_prod)*synch_count)
    inject_estim[k] = theta_hat
    print("theta",theta_hat)    
    # Permute for each surrogate the synchrony train, independently from one another
    ind_synch = np.tile(np.arange(Nsynch,dtype=int),(Nsurr,1))
    ind_synch = ind_synch.flatten()
    tag = np.sort(np.tile(np.arange(Nsynch),Nsurr))
    ind_synch = np.vstack((ind_synch,tag))    
    ind_permut = np.random.permutation(ind_synch.transpose()).transpose()
    ind_sort = np.argsort(ind_permut[1,:])
    ind_permut = ind_permut[0,ind_sort].reshape(Nsurr,Nsynch) + np.arange(0,Nsurr*Nsynch,Nsynch).reshape(Nsurr,1)
    # Split the synchrony train into frozen spikes and spikes that will be resampled
    ind_freeze = ind_permut[:,:theta_hat].flatten()
    ind_NOfreeze = ind_permut[:,theta_hat:].flatten()
    train_synch_NOTfrozen = np.tile(train_synch,Nsurr)[ind_NOfreeze]
    train_synch_frozen = np.tile(train_synch,Nsurr)[ind_freeze]
    Ttarg_backg = np.hstack(( np.tile(Ttarg_nosynch,(Nsurr,1)), train_synch_NOTfrozen.reshape(Nsurr,Nsynch-theta_hat) ))
    # Jitter the target train
    train_jitt = interval*np.floor(Ttarg_backg/interval) + np.random.uniform(0,interval,size=(Nsurr,Ttarg_backg.shape[1]))
    train_jitt = np.hstack(( train_jitt, train_synch_frozen.reshape(Nsurr,theta_hat) ))
    # Compute the injected synchrony estimate using our formula
    Tjitt = synch_width*np.floor(train_jitt/synch_width)
    Tjitt = Tjitt+duration_new*np.arange(Nsurr).reshape(Nsurr,1)
    Tref_surr = np.tile(Tref,Nsurr) + duration_new*np.arange(Nsurr).reshape(Nsurr,1)
    Tref_surr = Tref_surr.flatten()
    Tjitt = Tjitt.flatten()
    Tsynch_obs = np.array(list(set(Tref_surr) & set(Tjitt)))
    synch_obs = np.bincount(np.int64(np.floor(Tsynch_obs/duration_new)), minlength=Nsurr)
    count_ref = np.bincount(np.int64(np.floor(Tref/interval)), minlength=Ninterval)
    count_ref = np.tile(count_ref,Nsurr)
    #count_ref1 = np.bincount(np.int64(np.floor(Tref_surr/interval)), minlength=Nsurr*Ninterval)
    #print(count_ref.shape,count_ref1.shape,len(np.nonzero(count_ref)[0]),len(np.nonzero(count_ref1)[0]),np.array_equal(count_ref,count_ref1),len(np.nonzero(count_ref-count_ref1)[0]))
    count_jitt = np.bincount(np.int64(np.floor(Tjitt/interval)), minlength=Ninterval*Nsurr)
    count_synch = np.bincount(np.int64(np.floor(Tsynch_obs/interval)), minlength=Ninterval*Nsurr)
    RS_prod = np.sum(np.reshape(count_ref*count_synch,(Nsurr,Ninterval)), axis=1)
    RT_prod = np.sum(np.reshape(count_ref*count_jitt,(Nsurr,Ninterval)), axis=1) 
    injection_surrogate = np.int64((interval*synch_obs-RT_prod)/(interval*synch_obs-RS_prod)*synch_obs)  
    # Determine the bounds of the 1-alpha confidence interval
    CI_low[k] = np.percentile(injection_surrogate,alpha/2.)
    CI_up[k] = np.percentile(injection_surrogate,100-alpha/2.)     

ind_hit = np.where((inject_count-CI_low >= 0)*(CI_up-inject_count >= 0))[0]    
print('Proba of hit: ',len(ind_hit)*1./Nexper)    
   
# Represent the estimated injected synchrony distribution
FigCI = plt.figure()
#figure,ax = plt.subplots(1,1)
plt.xlabel('True Injected Count - Lower Bound',fontsize=18)
plt.ylabel('Upper Bound - True Injected Count',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#ax.set_xlim(xmin = 0)
#plt.xlim(xmin=0)
#plt.ylim(ymin=0)
plt.scatter(inject_count-CI_low,CI_up-inject_count,s=12,c='k',marker='o')
plt.plot(np.zeros(2),[np.amin(CI_up)-inject_count,np.amax(CI_up)-inject_count],'--k')
plt.plot([inject_count-np.amin(CI_low),inject_count-np.amax(CI_low)],np.zeros(2),'--k')
FigEst = plt.figure()
plt.xlabel('Confidence Interval Center',fontsize=18)
plt.ylabel('Normalized Count',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
x = (CI_up+CI_low)/2.-inject_count
print(np.amin(-30./s),np.amin(x/s),np.amin(x))
bins = np.arange(np.amin(x), np.amax(x)+1, 1)
count,base = np.histogram(x,bins=bins,density=1)
X = base[:-1]
plt.bar(X,count,width=.5,align='center',color='k',edgecolor='k')
plt.plot(np.mean(x)*np.ones(2),[0,np.amax(count)],'--k',linewidth=2)
x = inject_estim-inject_count
print(np.amax(20./s),np.amax(x/s),np.amax(x))
bins = np.arange(np.amin(x), np.amax(x)+1, 1)
count,base = np.histogram(x,bins=bins,density=1)
X = base[:-1]
plt.bar(X,count,width=.25,align='center',color='None',edgecolor='b')
plt.plot(np.mean(x)*np.ones(2),[0,np.amax(count)],'--b',linewidth=2)
plt.plot(np.zeros(2),[0,np.amax(count)],'--r')
FigCIlen = plt.figure()
plt.xlabel('Confidence Interval Length',fontsize=18)
plt.ylabel('Normalized Count',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
x = CI_up-CI_low
bins = np.arange(np.amin(x), np.amax(x)+1, 1)
count,base = np.histogram(x,bins=bins,density=1)
X = base[:-1]
plt.bar(X,count,width=.5,align='center',color='k',edgecolor='k')


#plt.plot(np.mean(injection_surrogate)*np.ones(2),[0,np.amax(count)],'--k')
#plt.plot(CI_low*np.ones(2),[0,np.amax(count)],'--c',linewidth=2)
#plt.plot(CI_up*np.ones(2),[0,np.amax(count)],'--c',linewidth=2)
#plt.plot(inject_count*np.ones(2),[0,np.amax(count)],'--r')

#xbins = np.arange(np.amin(theta-CI_low), np.amax(theta-CI_low)+1, 1)
#ybins = np.arange(np.amin(CI_up-theta), np.amax(CI_up-theta)+1, 1)
#count,_,_ = np.histogram2d(theta-CI_low,CI_up-theta,bins=(xbins,ybins),normed=True)
#print(count)
#im = plt.imshow(count, interpolation='nearest', origin='low',
#                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],cmap='jet')

#plt.plot(np.zeros(2),[0,CI_up-CI_low],'--k',linewidth=2)
#plt.plot((CI_up-CI_low)/2.*np.ones(2),[0,CI_up-CI_low],'--r',linewidth=2)
#plt.plot((CI_up-CI_low)*np.ones(2),[0,CI_up-CI_low],'--k',linewidth=2)
#plt.plot([0,CI_up-CI_low],np.zeros(2),'--k',linewidth=2)
#plt.plot([0,CI_up-CI_low],(CI_up-CI_low)/2.*np.ones(2),'--r',linewidth=2)
#plt.plot([0,CI_up-CI_low],(CI_up-CI_low)*np.ones(2),'--k',linewidth=2)

#FigTh = plt.figure()
#plt.xlabel('Injected synchrony estimate',fontsize=18)
#plt.ylabel('Normalized count',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#bins = np.arange(np.amin(injection_surrogate), np.amax(injection_surrogate)+1, 10)
#count,base = np.histogram(injection_surrogate,bins=bins,density=1)
#X = base[:-1]
#X1 = X[~np.isnan(count)]
#count1 = count[~np.isnan(count)]
#plt.bar(X1,count1,width=5,align='center',color='k',edgecolor='k')
#plt.plot(np.mean(injection_surrogate)*np.ones(2),[0,np.amax(count)],'--k')
#plt.plot(CI_low*np.ones(2),[0,np.amax(count)],'--c',linewidth=2)
#plt.plot(CI_up*np.ones(2),[0,np.amax(count)],'--c',linewidth=2)
#plt.plot(inject_count*np.ones(2),[0,np.amax(count)],'--r')
plt.show()

# Represent the estimated injected synchrony distribution
#FigTh.savefig('unbiased-vs-naive-estimate-distribution.eps')


# coding: utf-8

# In[1]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from ccg import correlograms

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

params = np.load('parameters_1.npy')        # Generative simulation parameters
Ntrial = int(params[0])                        # Number of pairs
duration = params[1]                      # Trial duration in (ms)
period = params[2]                 # Nonstationarity timescale in (ms)
Fs = params[3]                            # Sampling frequency

print(params)

train_ref = np.load('train_ref_1.npy')    # 
train_targ = np.load('train_targ_1.npy')  # Same for target trains

print(len(train_ref),len(train_ref)/Ntrial*1.)

# Basic Check: Homogeneous Poisson
check = 0
if check == 1:
    train_ref,label_ref = PoissonProcess(len(train_ref)/Ntrial*1., duration, Ntrial)
    train_targ,label_targ = PoissonProcess(len(train_targ)/Ntrial*1., duration, Ntrial)
    train_ref = label_ref*duration+train_ref
    train_targ = label_targ*duration+train_targ

# Inject spikes (simultaneously in both trains)
inject_count = 50
print("# injected spikes: ",inject_count)
Ninjectedtrial = int(Ntrial/2.)
train_inject = duration*np.random.uniform(0,1,size=(Ninjectedtrial,inject_count)) + np.reshape(np.arange(Ninjectedtrial)*duration,(Ninjectedtrial,1))
train_inject = train_inject.flatten()
train_ref = np.sort(np.append(train_ref,train_inject))
train_targ = np.sort(np.append(train_targ,train_inject)) 
train = np.append(train_ref,train_targ)
cell = np.int64(np.append(np.zeros(len(train_ref)),np.ones(len(train_targ))))

# -- Compute the probability of detection
synch_width = 2.5
lagmax = period
bine = .1
nb_ref = np.bincount(np.int64(np.floor(train_ref/duration)),minlength=Ntrial)
nb_targ = np.bincount(np.int64(np.floor(train_targ/duration)),minlength=Ntrial)
G = np.array([])
for k in range(Ntrial):
    G = np.append(G,np.tile(2*k,nb_ref[k]))
for k in range(Ntrial): 
    G = np.append(G,np.tile(2*k+1,nb_targ[k])) 
ind_sort = np.argsort(G)
T = train[ind_sort]
G = np.int64(G[ind_sort])
n = np.bincount(G)
ind = np.int64(np.append(np.zeros(1),np.cumsum(n)))
ind_sort = np.argsort(T)
st = T[ind_sort]*.001
sc = G[ind_sort]
Craw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(Craw[0,1]))-len(Craw[0,1])/2)*bine
ind_synch = np.where(abs(lag) <= synch_width/2.)[0]

#----Compute the ROC curve
Delta_range = np.array([4.1,5.1, 6.1])#np.linspace(period*.97,period*1.11,3) # np.array([period/2.,period,period*2.])
print(Delta_range)
nb_Delta = len(Delta_range)
cm = plt.get_cmap('gist_rainbow')
FigROC = plt.figure()
ax = FigROC.add_subplot(111)
ax.set_color_cycle([cm(1.*i/nb_Delta) for i in range(nb_Delta)])
plt.title('ROC curve',fontsize=18)
plt.xlabel('False positive rate',fontsize=18)
plt.ylabel('True positive rate',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(-.1,1.1)
plt.ylim(-.1,1.1)
plt.plot([0,1],[0,1],'--b')
plt.plot(.5*np.ones(2),[0,1],'--b')
plt.plot([0,1],.5*np.ones(2),'--b')

Nsurr = 11*9#0
nb_injection = Ninjectedtrial
nb_noinjection = Ntrial-Ninjectedtrial
synch_count = np.zeros(Ntrial)
synch_jitt = np.zeros((Ntrial,Nsurr))
for interval in Delta_range:
    print('--*--',interval,'--*--')
    for k in range(Ntrial):
        print(k)
        synch_count[k] = np.sum(Craw[2*k,2*k+1][ind_synch])
        Tref = T[ind[2*k]:ind[2*k+1]]-k*duration
        Ttarg = T[ind[2*k+1]:ind[2*(k+1)]]-k*duration
        # Compute synchrony based on jitter null
        Tjitt = np.tile(Ttarg,(Nsurr,1))
        Tjitt = interval*np.floor(Tjitt/interval)+np.random.uniform(0,interval,size=(Nsurr,len(Ttarg)))
        Tjitt = np.reshape(Tjitt,Nsurr*len(Ttarg))
        Tjitt = np.append(Tref,Tjitt)
        Tjitt = bine*np.floor(Tjitt/bine)
        Gjitt = np.sort(np.tile(np.arange(1,Nsurr+1,1),len(Ttarg)))
        Gjitt = np.append(np.zeros(len(Tref)),Gjitt)
        ind_sort = np.argsort(Tjitt)
        st = Tjitt[ind_sort]*.001
        sc = np.int64(Gjitt[ind_sort])
        C = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
        synch_jitt[k,:] = np.sum(C[0,1:,ind_synch],axis=0)
    synch_count = np.reshape(synch_count,(Ntrial,1))
    observed_synchrony = np.tile(synch_count,(1,Nsurr))
    nb_test = 1000
    threshold_range = np.reshape(np.linspace(0,1.,nb_test),(1,nb_test))
    th = np.tile(threshold_range,(nb_injection,1))
    # -- Jitter
    comparison = np.sign(np.sign(synch_jitt-observed_synchrony)+1)
    pvalue = (1+np.sum(comparison,axis=1))/(Nsurr+1.)
    pvalue_inj = pvalue[:nb_injection]
    pvalue_noinj = pvalue[nb_injection:]
    #----Compute the detection probabilities for each value of the threshold
    pval = np.tile(np.reshape(pvalue_inj,(nb_injection,1)),(1,nb_test))
    probatruepositive_jitt = np.sum(np.sign(np.sign(th-pval)+1),axis=0)*1./nb_injection
    pval = np.tile(np.reshape(pvalue_noinj,(nb_injection,1)),(1,nb_test))
    probafalsepositive_jitt = np.sum(np.sign(np.sign(th-pval)+1),axis=0)*1./nb_injection
    #----Compute the detection probabilities for each value of the threshold
    pval = np.tile(np.reshape(pvalue_inj,(nb_injection,1)),(1,nb_test))
    pval = np.tile(np.reshape(pvalue_noinj,(nb_injection,1)),(1,nb_test))

    ax.plot(threshold_range[0,:],probatruepositive_jitt,'.-')

    print(len(threshold_range),len(probatruepositive_jitt),np.shape(threshold_range))

FigLeg = plt.figure()
ax = FigLeg.add_subplot(111)
ax.set_color_cycle([cm(1.*i/nb_Delta) for i in range(nb_Delta)])
for i in range(nb_Delta):
    print(Delta_range[i])
    ax.plot(i*np.ones(2), np.ones(2), 'o',markeredgecolor='none',markersize=18)
plt.show()


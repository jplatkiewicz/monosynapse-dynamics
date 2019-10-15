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

params = np.load('parameters_var.npy')        # Generative simulation parameters
Ntrial = int(params[0])                        # Number of pairs
duration = params[1]                      # Trial duration in (ms)
period = params[2]                 # Nonstationarity timescale in (ms)
Fs = params[3]                            # Sampling frequency

print(params)

train_ref = np.load('train_ref_var.npy')
train_targ = np.load('train_targ_var.npy')  # Same for target trains

print(len(train_ref),len(train_ref)/Ntrial*1.)

# Basic Check: Homogeneous Poisson
check = 0
if check == 1:
    train_ref,label_ref = PoissonProcess(len(train_ref)/Ntrial*1., duration, Ntrial)
    train_targ,label_targ = PoissonProcess(len(train_targ)/Ntrial*1., duration, Ntrial)
    train_ref = label_ref*duration+train_ref
    train_targ = label_targ*duration+train_targ

train = np.append(train_ref,train_targ)
cell = np.int64(np.append(np.zeros(len(train_ref)),np.ones(len(train_targ))))

# Compute the CCG of the spike trains
synch_width = 5.
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
lagmax = period
bine = .1
ind_sort = np.argsort(T)
st = T[ind_sort]*.001
sc = G[ind_sort]
Craw = correlograms(st, sc, sample_rate=Fs, bin_size=bine/1000., window_size=lagmax/1000.)
lag = (np.arange(len(Craw[0,1]))-len(Craw[0,1])/2)*bine
ind_synch = np.where(abs(lag) <= synch_width/2.)[0]
synch_count = np.zeros(Ntrial)
for k in range(Ntrial):
    synch_count[k] = np.sum(Craw[2*k,2*k+1][ind_synch])
i_mean = np.argmin(abs(synch_count-np.mean(synch_count)))    
     
Nperiod = int(duration/period)
n = np.bincount(G)
ind = np.int64(np.append(np.zeros(1),np.cumsum(n)))
count_targ = np.int64(np.bincount(np.int64(np.floor(train_targ/period)), minlength=Nperiod*Ntrial))
count_targ = np.reshape(count_targ,(Ntrial,Nperiod))
Nsurr = 1100
Tref = T[ind[2*i_mean]:ind[2*i_mean+1]]-i_mean*duration
Ttarg = T[ind[2*i_mean+1]:ind[2*(i_mean+1)]]-i_mean*duration
# Compute synchrony based on jitter null
Tjitt = np.tile(Ttarg,(Nsurr,1))
Tjitt = period*np.floor(Tjitt/period) + np.random.uniform(0, period, size=(Nsurr,len(Ttarg)))
Tjitt = np.reshape(Tjitt,Nsurr*len(Ttarg))
Tjitt = np.append(Tref,Tjitt)
Gjitt = np.sort(np.tile(np.arange(1,Nsurr+1,1),len(Ttarg)))
Gjitt = np.append(np.zeros(len(Tref)),Gjitt)
ind_sort = np.argsort(Tjitt)
st = Tjitt[ind_sort]*.001
sc = np.int64(Gjitt[ind_sort])
C = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
synch_jitt = np.sum(C[0,1:,ind_synch],axis=0)
# Compute synchrony based on bootstrap null
c_targ = count_targ[i_mean,:]
ind_spike = np.nonzero(c_targ)[0]
Tboot = []
Gboot = []
for j in ind_spike:
    tr,tag = PoissonProcess(c_targ[j],period,Nsurr)
    tr = tr+j*period
    tag = tag+1
    Tboot.extend(tr)
    Gboot.extend(tag)      
Tboot = np.asarray(Tboot)
Gboot = np.asarray(Gboot)
Tboot = np.append(Tref,Tboot)
Gboot = np.append(np.zeros(len(Tref)),Gboot)
ind_sort = np.argsort(Tboot)
st = Tboot[ind_sort]*.001
sc = np.int64(Gboot[ind_sort])
C = correlograms(st, sc,sample_rate=Fs, bin_size=bine/1000., window_size=lagmax/1000.)
synch_boot = np.sum(C[0,1:,ind_synch],axis=0)

FigSynch = plt.figure()
width = .5
bins = np.int64(np.linspace(np.amin(synch_boot),np.amax(synch_boot)+1,100))
count,base = np.histogram(synch_boot,bins=bins,density=1)
X = base[:-1]
X1 = X[~np.isnan(count)]
count1 = count[~np.isnan(count)]
plt.bar(X1,count1,width,align='center',color='c',edgecolor='c')
bins = np.int64(np.linspace(np.amin(synch_jitt),np.amax(synch_jitt)+1,100))
count,base = np.histogram(synch_jitt,bins=bins,density=1)
X = base[:-1]
X1 = X[~np.isnan(count)]
count1 = count[~np.isnan(count)]
plt.bar(X1,count1,width/2.,align='center',color='None',edgecolor='k',linewidth=2)
plt.plot(np.ones(2)*np.mean(synch_boot),[0,np.amax(count1)],'--c')
plt.plot(np.ones(2)*np.mean(synch_jitt),[0,np.amax(count1)],'--k')
plt.plot(np.ones(2)*synch_count[i_mean],[0,np.amax(count1)],'--r')
plt.show()

FigSynch.savefig('synchrony_distribution.eps')
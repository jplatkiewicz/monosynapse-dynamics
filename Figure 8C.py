# coding: utf-8

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def compthta(Rtemp,Ttemp,delta):
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

params = np.load('parameters.npy')
weight_value = np.load('weights.npy')
phase = duration/Nphase
Nphase = int(Nphase)

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
lmax = 0.
x = (train_targ-lmax)*(np.sign(train_targ-lmax)+1)/2.
x = x[np.nonzero(x)]
Ttarg = synch_width*np.floor(train_targ/synch_width)
Tsynch = np.array(list(set(Tref) & set(Ttarg)))
synch_count = np.bincount(np.int64(np.floor(Tsynch/(Ntrial*phase))),minlength=Nphase)
#--WITHOUT SYNAPSE--
Tref0 = synch_width*np.floor(train_ref0/synch_width)
lmax0 = 0.
x = (train_targ0-lmax0)*(np.sign(train_targ0-lmax0)+1)/2.
x = x[np.nonzero(x)]
Ttarg0 = synch_width*np.floor(x/synch_width)
Tsynch0 = np.array(list(set(Tref0) & set(Ttarg0)))
synch_count0 = np.bincount(np.int64(np.floor(Tsynch0/(Ntrial*phase))),minlength=Nphase)

# Excess synchrony count unbiased estimation
delta = period
Ndelta = int(Ntrial*duration/delta)
count_ref = np.bincount(np.int64(np.floor(train_ref/delta)),minlength=Ndelta)
count_targ = np.bincount(np.int64(np.floor(train_targ/delta)),minlength=Ndelta)
count_synch = np.bincount(np.int64(np.floor(Tsynch/delta)),minlength=Ndelta)
Ndelta_phase = int(Ntrial*phase/delta)
RS_prod = np.sum(np.reshape(count_ref*count_synch,(Nphase,Ndelta_phase)),axis=1)
alpha = RS_prod/(delta*synch_count)  
RT_prod = np.sum(np.reshape(count_ref*count_targ,(Nphase,Ndelta_phase)),axis=1)
estimate = (delta*synch_count-RT_prod)/(delta*synch_count-RS_prod)*synch_count

# Evaluate the true injected synchrony count by comparing conditions "synapse on" and "synapse off"
injected_true = (synch_count-synch_count0)

estimate = []; synchOn = []
indT = np.digitize(train_targ%duration,np.arange(0,12000,1000)) - 1
indR = np.digitize(train_ref%duration,np.arange(0,12000,1000)) - 1
for k in range(int(len(np.unique(indT)))):
    temp = compthta(np.round(train_ref[indR==k]),np.round(train_targ[indT==k]),delta)
    estimate = np.append(estimate,temp)
    temp = len(np.intersect1d(np.round(train_ref[indR==k]),np.round(train_targ[indT==k])))
    synchOn = np.append(synchOn,temp)

synchOff = []
indT = np.digitize(train_targ0%duration,np.arange(0,12000,1000)) - 1
indR = np.digitize(train_ref0%duration,np.arange(0,12000,1000)) - 1
for k in range(int(len(np.unique(indT)))):
    temp = len(np.intersect1d(np.round(train_ref0[indR==k]),np.round(train_targ0[indT==k])))
    synchOff = np.append(synchOff,temp)

# Check the result
plt.figure()
plt.xlabel('Normalized Conductance')
plt.ylabel('Estimate Rate (Hz)')
x = g0/gm*weight_value
y = estimate/(Ntrial*phase*.001)
ind = np.argsort(x)
x = x[ind]
y = y[ind]
plt.plot(x,y,'o-k',linewidth=.25,markersize=16)
FigPredic = plt.figure()
plt.xlabel('Normalized Conductance')
plt.ylabel('Normalized Estimate')
x = g0/gm*weight_value
y = estimate*1./synch_count
ind = np.argsort(x)
x = x[ind]
y = y[ind]
plt.plot(x,y,'o-k')
plt.figure()
plt.xlabel('Estimate Rate (Hz)')
plt.ylabel('PSP (mV)')
x = estimate/(Ntrial*phase*.001)
y = g0*weight_value*(0-(-50))/120.*10**9
gradient,intercept,r_value,p_value,std_err = stats.linregress(x,y)
print("Relationship between PSP and theta_hat; slope:",gradient,"intercept",intercept)
print("Linear fit of theta_hat-gsyn; R:",r_value,"p-value",p_value)
plt.plot(x,y,'ok')
plt.plot(x,gradient*x+intercept,'-r')
plt.figure()
plt.xlabel('Normalized Estimate')
plt.ylabel('PSP (mV)')
x = estimate*1./synch_count
y = g0*weight_value*(0-(-50))/120.*10**9
gradient,intercept,r_value,p_value,std_err = stats.linregress(x,y)
print("Relationship between PSP and theta_hat; slope:",gradient,"intercept",intercept)
print("Linear fit of theta_hat-gsyn; R:",r_value,"p-value",p_value)
plt.plot(x,y,'ok')
plt.plot(x,gradient*x+intercept,'-r')
FigCheck = plt.figure()
plt.xlabel('Normalized True')
plt.ylabel('Normalized Estimate')
x = injected_true*1./synch_count
y = estimate*1./synch_count
gradient,intercept,r_value,p_value,std_err = stats.linregress(x,y)
print("Linear fit of theta_hat-theta; R:",r_value,"p-value",p_value)
mM = np.array([np.amin(x),np.amax(x)])
plt.plot(mM,gradient*mM+intercept,'--r',linewidth=.5)
plt.plot(x,y,'ok',markersize=16)
plt.show()

FigPredic.savefig('Fig_prediction-conductance.eps')
FigCheck.savefig('Fig_prediction-injected-on-off.eps')

plt.plot(synchOn-synchOff,estimate)
plt.plot(np.array([0,2e3]),np.array([0,2e3]))

y = g0*weight_value*(0-(-50))/120.*10**9
plt.plot(y,estimate)

x = synchOn-synchOff
y = estimate
P = np.polyfit(x, y, 1)
xx = np.linspace(np.min(x),np.max(x),1e3)
yy = P[0]*xx + P[1]

FigEst = plt.figure(figsize=(7,5))
colG = np.array([128,129,132])/255
plt.plot(xx,yy,color=colG,linestyle='--')
plt.plot(synchOn-synchOff,estimate,'ok',color='k',markersize=12)
plt.xlim([0,2000])
plt.ylim([0,2000])
plt.xlabel('theta')
plt.ylabel('theta hat')
plt.box('off')
plt.show()
FigEst.savefig('Fig_zach_Fig8C1.eps')
FigEst.savefig('Fig_zach_Fig8C1.tif')


FigCon = plt.figure(figsize=(7,5))
y = g0*weight_value*(0-(-50))/120.*10**9
plt.plot(y,estimate,color='k',linewidth=.7)
plt.plot(y,estimate,'ok',color='k',markersize=12)
plt.xlabel('Normalized peak conductance')
plt.ylabel('theta hat')
plt.box('off')
plt.show()
FigCon.savefig('Fig_zach_Fig8C2.eps')
FigCon.savefig('Fig_zach_Fig8C2.tif')


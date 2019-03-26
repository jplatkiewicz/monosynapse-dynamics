import numpy as np
from scipy import stats
#get_ipython().magic('matplotlib inline')
from ccg import correlograms
import matplotlib.pyplot as plt


params = np.load('parameters_dynamic.npy')
train_ref = np.load('train_ref_dynamic.npy')
train_targ = np.load('train_targ_dynamic.npy')

# Simulation parameters
Ntrial = params[0]
duration = params[1]    # duration of the trial
time_step = params[2]
Fs = 1./(time_step*.001)
period = params[3]

# Excess synchrony count unbiased estimation USING A SLIDING WINDOW
synch_width = 1.*5
delta = period
Ndelta = int(Ntrial*duration/delta)
window_width = Ntrial*duration/50.-0*delta #5.#10. #10*period
Ninterval_window = int(window_width/delta)
Nwindow = Ndelta-Ninterval_window
print("# sliding windows:", Nwindow)
Ndt_window = int(window_width/time_step)
estimate = np.array([])
true = np.array([])
s = np.array([])
x = np.array([])
Tr = synch_width*np.unique(np.floor(train_ref/synch_width))
Tt = synch_width*np.unique(np.floor(train_targ/synch_width))
train_synch = np.array(list(set(Tr) & set(Tt)))
i_old = 0
iA = 0
iB = 0
j_old = 0
jA = 0
jB = 0
t0 = 0
while t0 <= Ntrial*duration-window_width:
    #--Reference train
    Tref = np.array([])
    iA = i_old
    if iA < len(train_ref) and train_ref[iA] < t0:
        while iA < len(train_ref) and train_ref[iA] < t0:
            iA += 1
        i_old = iA-1
    if iA < len(train_ref) and train_ref[iA] < t0+window_width:
        iB = iA
        while iB < len(train_ref) and train_ref[iB] < t0+window_width: 
            iB += 1
        Tref = train_ref[iA:iB] 
    #--Target train
    Ttarg = np.array([])
    jA = j_old
    if jA < len(train_targ) and train_targ[jA] < t0:
        while jA < len(train_targ) and train_targ[jA] < t0:
            jA += 1
        j_old = jA-1
    if jA < len(train_targ) and train_targ[jA] < t0+window_width:
        jB = jA
        while jB < len(train_targ) and train_targ[jB] < t0+window_width: 
            jB += 1
        Ttarg = train_targ[jA:jB] 
    #--Compute injected synchrony count estimate
    if len(Tref) >= 1 and len(Ttarg) >= 1:
        Tref = synch_width*np.unique(np.floor((Tref-t0)/synch_width))
        Ttarg = synch_width*np.unique(np.floor((Ttarg-t0)/synch_width))
        Tsynch = np.array(list(set(Tref) & set(Ttarg)))
        count_ref = np.bincount(np.int64(np.floor(Tref/delta)),minlength=Ninterval_window)
        count_targ = np.bincount(np.int64(np.floor(Ttarg/delta)),minlength=Ninterval_window)
        count_synch = np.bincount(np.int64(np.floor(Tsynch/delta)),minlength=Ninterval_window)
        RS_prod = np.sum(count_ref*count_synch)
        RT_prod = np.sum(count_ref*count_targ)
        synch_count = len(Tsynch)
        estimate = np.append(estimate,(delta*synch_count-RT_prod)/(delta*synch_count-RS_prod)*synch_count)
        s = np.append(s,synch_count)
        x = np.append(x,np.sum(count_targ))
    else:
        estimate = np.append(estimate,0)
        s = np.append(s,synch_count)
        x = np.append(x,np.sum(count_targ))
    t0 += delta
    #
    #lagmax = 100.                  
    #bine = 1.                    
    #train = np.append(Tref,Ttarg)
    #cell = np.int64(np.append(np.zeros(len(Tref)),np.ones(len(Ttarg))))
    #ind_sort = np.argsort(train)
    #st = train[ind_sort]*.001
    #sc = cell[ind_sort]
    #C = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
    #lag = (np.arange(len(C[0,1]))-len(C[0,1])/2)*bine
    #train0 = np.append(Tref0,Ttarg0)
    #cell0 = np.int64(np.append(np.zeros(len(Tref0)),np.ones(len(Ttarg0))))    
    #ind_sort = np.argsort(train0)
    #st = train0[ind_sort]*.001
    #sc = cell0[ind_sort]
    #C0 = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
    #def close_event():
    #    plt.close()
    #fig = plt.figure()
    #timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
    #timer.add_callback(close_event)
    #plt.plot(lag,C[0,1]/(len(train_ref)*bine*.001),'.-k')
    #plt.plot(lag,C[0,1]/(len(train_ref)*bine*.001),'.-c')
    #timer.start()
    #plt.show()    
    
# Check the result
time_win = np.arange(0,len(estimate),1)*(Ntrial*duration-window_width)/(len(estimate)-1.)+window_width/2.
#estimate_PSP = (estimate/s*24.475-22.302)
plt.figure()
plt.plot(time_win/1000.,x,'-k')
plt.figure()
plt.xlabel('Time (s)',fontsize=18)
plt.ylabel('Estimate',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.plot(time_win,,'o-k')
plt.plot(time_win/1000.,estimate/(Ntrial*duration*.001),'-k')
FigTracking = plt.figure()
plt.xlabel('Time (s)',fontsize=18)
plt.ylabel('PSP Estimate',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.plot(time_win,,'o-k')
plt.plot(time_win/1000.,2.83*estimate/(Ntrial*duration*.001)-.927,'-k')
plt.figure()
plt.plot(time_win/1000.,estimate*1./s,'-k')
plt.figure()
plt.plot(time_win/1000.,24.475*estimate*1./s-22.302,'-k')
plt.figure()
plt.plot(time_win/1000.,s,'-m')
plt.plot(time_win/1000.,estimate,'-b')
plt.show()

FigTracking.savefig('Fig_tracking-conductance.eps')
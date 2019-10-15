from brian2 import *
from ccg import correlograms


# Simulation parameters
Ntrial = 5000
duration = 1000.    # duration of the trial
time_step = 0.1            #-in (ms)
defaultclock.dt = time_step*ms  
Fs = 1/(time_step*.001)    #-in (Hz)

print("# trials:",Ntrial,"Trial duration: ",duration,"(ms)")

# Biophysical neuron parameters
#--Neuron parameters
cm = 250*pF               # membrane capacitance
gm = 25*nS                # membrane conductance
tau = cm/gm               # membrane time constant
El = -70*mV               # resting potential
Vr = El+10*mV             # reset value
refractory_period = 0*ms  # refractory period

# Adaptive threshold parameters
# -- Pyramidal cell
tauTh_pyr = 7*ms
Vi_pyr = -60*mV
ka_pyr = 5*mV
ki_pyr = ka_pyr/1.
alpha_pyr = ka_pyr/ki_pyr
Vt_pyr = Vi_pyr+5*mV
# -- Interneuron
tauTh_int = 1*ms
Vi_int = -60*mV
ka_int = 5*mV
ki_int = ka_int/.75
alpha_int = ka_int/ki_int
Vt_int = Vi_int+3*mV

# Biophysical background input parameters
tauI = 10*ms
muI = -50*mV
sigmaI_pyr = 6*mvolt
sigmaI = 2*mvolt 
xmin = muI-5*mV
xmax = muI+5*mV
period = 10.

# Neuron model: Leaky Integrate-and-Fire
#-----
#-Integrate-and-fire neuron model 
#-----
eqs_ref = Equations('''
dV/dt = (-V+mu+sigmaI_pyr*I)/tau : volt 
dtheta/dt = (-theta+Vt_pyr+alpha_pyr*(V-Vi_pyr)*int(V>=Vi_pyr))/tauTh_pyr : volt  
dI/dt = -I/tauI+(2/tauI)**.5*xi : 1
mu : volt 
''')
eqs_targ = Equations('''
dV/dt = (-V+mu+sigmaI*I)/tau : volt 
dtheta/dt = (-theta+Vt_int+alpha_int*(V-Vi_int)*int(V>=Vi_int))/tauTh_int : volt  
weight : 1
dI/dt = -I/tauI+(2/tauI)**.5*xi : 1
mu : volt (linked)
''')
#-----Specify the model
reference = NeuronGroup(Ntrial,model=eqs_ref,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
target = NeuronGroup(Ntrial,model=eqs_targ,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
reference.run_regularly('''mu = xmin+(xmax-xmin)*rand()''',dt=period*ms)
target.mu = linked_var(reference,'mu')
#-----Parameter initialization
reference.V = (xmax-xmin)*rand()+xmin
reference.theta = Vt_pyr+alpha_pyr*(reference.V-Vi_pyr)*np.sign(reference.V-Vi_pyr)
reference.I = 2*rand()-1
target.V = (xmax-xmin)*rand()+xmin
target.theta = Vt_int+alpha_int*(target.V-Vi_int)*np.sign(target.V-Vi_int)
target.I = 2*rand()-1
#-----Record variables
Sref = SpikeMonitor(reference)
Starg = SpikeMonitor(target) 
Mref = StateMonitor(reference,'V',record=True) 
Mtarg = StateMonitor(target,'V',record=True)

run(duration*ms)

# Representing the basic recorded variables
FigVm = figure()
xlabel('Time (ms)')
ylabel('Potential (mV)')
title('Comodulated pyramidal-interneuron LIFs')
plot(Mref.t/ms,Mref.V[0]/mV,'k')
plot(Mtarg.t/ms,Mtarg.V[0]/mV,'b')

# Show the correlation function for Vm
lagmax = 25.
Vref = reshape(Mref.V/mV,Ntrial*len(Mref.t))
Vtarg = reshape(Mtarg.V/mV,Ntrial*len(Mtarg.t))
m_max = int(lagmax/time_step)
n = len(Vtarg)
Vref_avg = mean(Vref)
Vtarg_avg = mean(Vtarg)    
CCF = zeros(2*m_max)
for k in range(2*m_max):
    m = k-m_max
    CCF[k] = sum((Vtarg[abs(m):]-Vtarg_avg)*(Vref[:n-abs(m)]-Vref_avg))/n
lag = arange(-m_max,m_max)*time_step
gamma_ref = sum((Vref-Vref_avg)*(Vref-Vref_avg))/n
gamma_targ = sum((Vtarg-Vtarg_avg)*(Vtarg-Vtarg_avg))/n
normalization = sqrt(gamma_ref*gamma_targ)
CCF = CCF/normalization
FigCCF = figure()
xlabel('Time lag (ms)')
ylabel('Potential (mV)')
title('Vm CCF')
xlim(-lagmax,lagmax)
plot(lag,CCF,'-k')

# Define the target train
train_ref = sort(Sref.i*duration+Sref.t/ms)
train_targ = sort(Starg.i*duration+Starg.t/ms)
train = append(train_ref,train_targ)
cell = int64(append(zeros(len(train_ref)),ones(len(train_targ))))

# Basic firing parameters
print("[Reference] # spikes: ",len(train_ref),
      "Average firing rate",len(train_ref)/(Ntrial*duration*.001),"(Hz)",
      "CV",std(diff(train_ref))/mean(diff(train_ref)))
print("[Target] # spikes: ",len(train_targ),
      "Average firing rate",len(train_targ)/(Ntrial*duration*.001),"(Hz)",
      "CV",std(diff(train_targ))/mean(diff(train_targ)))

# Compute the CCG of the spike trains
lagmax = 50.
bine = .1
ind_sort = np.argsort(train)
st = train[ind_sort]*.001
sc = cell[ind_sort]
Craw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(Craw[0,1]))-len(Craw[0,1])/2)*bine
FigCCG = figure()
title('Cross-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Firing rate (Hz)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
xlim(-lagmax/2.,lagmax/2.)
bar(lag,Craw[0,1]/(len(train_ref)*bine*.001),bine,align='center',color='k',edgecolor='k')

# Show the postsynaptic Vm STA
def STA(T,Vm,lag_max,duration,time_step):
    #--cut the part of the spike train that cannot be used for the STA    
    i = 0
    while T[i] < lag_max:
        i += 1
    i0 = i
    i = len(T)-1
    while T[i] > duration-lag_max:
        i -= 1
    i1 = i
    T = T[i0:i1]
    dt = time_step
    nb_spike = int(len(T))
    nb_point = int(2*lag_max/dt)
    sample = np.zeros((nb_spike,nb_point))
    for i in range(nb_spike):
        istart = np.int64(round((T[i]-lag_max)/dt))
        istop = np.int64(round((T[i]+lag_max)/dt)) 
        sample[i,:] = Vm[istart:istop]
    average = np.mean(sample,axis=0)

    return average

lagmax = 25.
period = Ntrial*duration
Vsta = STA(train_ref,Vtarg,lagmax,period,time_step)
FigSTA = figure()
title('Spike-triggered average of Postsynaptic $V_m$',fontsize=18)
xlabel('Time Lag (ms)',fontsize=18)
ylabel('Potential (mV)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
xlim(-lagmax,lagmax)
lag = linspace(-lagmax,lagmax,len(Vsta))
plot(lag,Vsta,'-k')
plot(zeros(2),[min(Vsta),max(Vsta)],'--k',linewidth=2)
show()

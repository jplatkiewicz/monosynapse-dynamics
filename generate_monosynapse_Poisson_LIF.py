from brian2 import *
#from scipy import special,stats,optimize,interpolate#signal,signal,integrate
import numpy as np
#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
#from tools import STA#,STA_bin
from ccg import correlograms


# Simulation parameters
Ntrial = 1000*8
duration = 5000.         # duration of the trial
time_step = 0.1            #-in (ms)
defaultclock.dt = time_step*ms  
Fs = 1/(time_step*.001)    #-in (Hz)          

print('# trials: ',Ntrial)

# Biophysical parameters
#--Neuron parameters
cm = 250*pF               # membrane capacitance
gm = 25*nS                # membrane conductance
tau = cm/gm               # membrane time constant
El = -70*mV               # resting potential
Vr = El+10*mV             # reset value
refractory_period = 0*ms  # refractory period

# Adaptive threshold parameters
tau_th = 1*ms
Vi = -60*mV
ka = 5*mV
ki = ka/.75
alpha = ka/ki
Vt = Vi+3*mV
print("Baseline threshold voltage: ",Vt/mV,"(mV)")

# Background synaptic input parameters
tauI = 10*ms              # input time constant
mu = -50*mV
sigmaI = 2.*mvolt

# Monosynaptic parameters
tauS = 3*ms                # synaptic conductance time constant
Esyn = 0*mV# muI-5*mV             # synapse reversal potential (mu+20*mV)
PSC = 50*pA               # post-synaptic current ammplitude: (high background noise: 100 pA | low noise regime: 15 pA)
g0 = PSC/(Esyn-mu)
latency = 0*ms

# Specify the neuron model: leaky integrate-and-fire
#-----
#-Integrate-and-fire neuron model with monosynapse
eqs = Equations('''
#-Potential
dV/dt = (-V+muI+sigmaI*I-g0/gm*gsyn*(V-Esyn))/tau : volt (unless refractory)
muI : volt
#-Threshold
dtheta/dt = (-theta+Vt+alpha*(V-Vi)*int(V>=Vi))/tau_th : volt (unless refractory)
#(-theta+Vt+ka*log(1+exp((V-Vi)/ki)))/tau_th : volt (unless refractory)
#-Individual Backgound Input
dI/dt = -I/tauI+(2/tauI)**.5*xi : 1
#-Monosynaptic input
dgsyn/dt = -gsyn/tauS : 1
''')

#-----Model setting
reference = PoissonGroup(Ntrial,30*Hz)
target = NeuronGroup(Ntrial,model=eqs,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
target_isol = NeuronGroup(Ntrial,model=eqs,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
#----Parameters initialization
target.gsyn = 0
for k in range(8):
    i0 = int(k*Ntrial/8.)
    i1 = int((k+1)*Ntrial/8.)
    val = -50*mV+.75*k*mV
    target[i0:i1].muI = val
    target[i0:i1].V = val
    target[i0:i1].theta = Vt+alpha*(val-Vi)*int(val>=Vi)
target_isol.gsyn = 0
for k in range(8):
    i0 = int(k*Ntrial/8.)
    i1 = int((k+1)*Ntrial/8.)
    val = -50*mV+.75*k*mV
    target_isol[i0:i1].muI = val
    target_isol[i0:i1].V = val
    target_isol[i0:i1].theta = Vt+alpha*(val-Vi)*int(val>=Vi)
#--Synaptic connection
synaptic = Synapses(reference,target,
             on_pre='''
             gsyn += 1
             ''')
synaptic.connect(i=arange(Ntrial),j=arange(Ntrial))
synaptic.delay = latency
#-----
#-Record variables
Sref = SpikeMonitor(reference)
Starg = SpikeMonitor(target)
Starg_isol = SpikeMonitor(target_isol)

run(duration*ms)

G = np.sort(Starg_isol.i)
Ngroup = int(Ntrial/8.)
nbr = np.bincount(np.int64(np.floor(G/Ngroup)))
rate = nbr/(Ngroup*duration*.001)
print('Firing rates: ',rate,'in (Hz)')

train_ref = sort(Sref.i*duration+Sref.t/ms)
train_targ = sort(Starg.i*duration+Starg.t/ms)

save('parameters.npy',np.array([Ntrial,duration,Fs]))
save('train_ref.npy',train_ref)
save('train_targ.npy',train_targ)
from brian2 import *
#from scipy import special,stats,optimize,interpolate#signal,signal,integrate
import numpy as np
#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
#from tools import STA#,STA_bin
from ccg import correlograms


# Simulation parameters
Ntrial = 1000*10
duration = 500.         # duration of the trial
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
muI = -50*mV
sigmaI = 2.*mvolt

# Monosynaptic parameters
tauS = 3*ms                # synaptic conductance time constant
Esyn = 0*mV# muI-5*mV             # synapse reversal potential (mu+20*mV)
PSC = 50*pA               # post-synaptic current ammplitude: (high background noise: 100 pA | low noise regime: 15 pA)
g0 = PSC/(Esyn-muI)
latency = 1.5*ms

# Stimulus parameters
t0 = 475*ms

# Specify the neuron model: leaky integrate-and-fire
#-----
#-Integrate-and-fire neuron model with monosynapse
eqs = Equations('''
#-Potential
dV/dt = (-V+muI+sigmaI*I-g0/gm*gsyn*(V-Esyn))/tau : volt (unless refractory)
#-Threshold
dtheta/dt = (-theta+Vt+alpha*(V-Vi)*int(V>=Vi))/tau_th : volt (unless refractory)
#(-theta+Vt+ka*log(1+exp((V-Vi)/ki)))/tau_th : volt (unless refractory)
#-Individual Backgound Input
I : 1 (linked)
#-Monosynaptic input
dgsyn/dt = -gsyn/tauS : 1
''')
eqs_input = Equations(''' 
dx/dt = -x/tauI+(2/tauI)**.5*xi : 1
''')
#-----Model setting
cell_inj = arange(Ntrial) 
spiketime_inj = t0*ones(Ntrial)
stimulus = SpikeGeneratorGroup(Ntrial,cell_inj,spiketime_inj)
target = NeuronGroup(Ntrial,model=eqs,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
target_no = NeuronGroup(Ntrial,model=eqs,threshold='V>theta+1000*mV',reset='V=Vr',refractory=refractory_period,method='euler')
background = NeuronGroup(Ntrial,model=eqs_input,threshold='x>1000',reset='x=0',refractory=refractory_period,method='euler')
target.I = linked_var(background,'x')
target_no.I = linked_var(background,'x')
#----Parameters initialization
target.V = muI
target.gsyn = 0
target.theta = Vt
target_no.V = target.V
target_no.gsyn = 0
target_no.theta = Vt
background.x = 2*rand(Ntrial)-1
#--Synaptic connection
synaptic = Synapses(stimulus,target,
             on_pre='''
             gsyn += 1
             ''')
synaptic.connect(i=arange(Ntrial),j=arange(Ntrial))
synaptic.delay = latency
synaptic_no = Synapses(stimulus,target_no,
             on_pre='''
             gsyn += 1
             ''')
synaptic_no.connect(i=arange(Ntrial),j=arange(Ntrial))
synaptic_no.delay = latency
#-----
#-Record variables
Starg = SpikeMonitor(target)
Mtarg = StateMonitor(target_no,('V'),record=True) 

run(duration*ms)

# Compute the PSTH
bin_size = .1
lagmax = 50
spiketime = sort(Starg.t/ms)
spiketime = int64(floor(spiketime/bin_size))
count = bincount(spiketime,minlength=int(duration/bin_size))/(Ntrial*bin_size*.001)
nb = int64(round(lagmax/bin_size))
PSTH = count[-nb:]/(Ntrial*bin_size*.001)

# Compute the spike-triggered average of Vm
PSP = mean(Mtarg.V/mV,0)
nbPSP = int64(round(lagmax/time_step))
PSP = PSP[-nbPSP:]

# Represent the PSTH
PSTHgraph = figure()
title('PSTH',fontsize=18)
xlabel('Time lag (ms)',fontsize=18)
ylabel('Firing Rate (Hz)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
xlim(-lagmax/2.,lagmax/2.)
lag = (arange(len(PSTH))-len(PSTH)/2)*bin_size
bar(lag,PSTH,bin_size,align='center',color='k',edgecolor='k')
STAgraph = figure()
title('Average PSP - Use Noisy Trace',fontsize=18)
xlabel('Time lag (ms)',fontsize=18)
ylabel('Potential (mV)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
xlim(-lagmax/2.,lagmax/2.)
nb_point = len(PSP)
lag_Vm = linspace(-lagmax/2.,lagmax/2.,nb_point)
plot(lag_Vm,PSP,'k')
show()

PSTHgraph.savefig('Fig_PSTH.eps')#,transparent=True)
STAgraph.savefig('Fig_STA.eps')
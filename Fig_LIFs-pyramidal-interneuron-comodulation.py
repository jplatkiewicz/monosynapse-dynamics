from brian2 import *
from scipy import stats
from ccg import correlograms


# Simulation parameters
Ntrial = 5000
duration = 1000.    # duration of the trial
time_step = 0.01            #-in (ms)
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

# Monosynapse parameters
tauS = 3*ms                # synaptic conductance time constant
Esyn = 0*mV                # synapse reversal potential (mu+20*mV)
PSC = 30*pA                # post-synaptic current ammplitude
g0 = PSC/(Esyn-muI)
print("Synaptic peak conductance: ",g0/nsiemens,"(siemens)")
latency = 1.*ms

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
Mref = StateMonitor(reference,('V','theta'),record=0) 
Mtarg = StateMonitor(target,('V','theta'),record=0)
Mmu = StateMonitor(target,'mu',dt=period*ms,record=True)

run(duration*ms)

# Representing the basic recorded variables
FigVm = figure()
xlabel('Time (ms)')
ylabel('Potential (mV)')
title('Comodulated pyramidal-interneuron LIFs')
plot(Mref.t/ms,Mref.V[0]/mV,'k')
plot(Mtarg.t/ms,Mtarg.V[0]/mV,'b')
#show()

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

# Show the effect of modulating the signal onto the synchrony count
synch_width = 1.
Tref = synch_width*floor(train_ref/synch_width)
Ttarg = synch_width*floor((train_targ-latency/ms)/synch_width)
Tsynch = array(list(set(Tref) & set(Ttarg)))
synch_count = bincount(int64(floor(Tsynch/period)),minlength=int(Ntrial*duration/period))
amplitude = reshape(Mmu.mu/mV,Ntrial*len(Mmu.t))
FigMu = figure()
plot(amplitude,synch_count,'.k')
show()

FigVm.savefig('Fig-pyr-int-Vm-traces.eps')
FigCCG.savefig('Fig-pyr-int-CCG.eps')
FigMu.savefig('Fig-pyr-int-mu-synch.eps')
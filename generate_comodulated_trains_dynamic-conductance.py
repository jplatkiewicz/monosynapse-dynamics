from brian2 import *
from ccg import correlograms


# Simulation parameters
Ntrial = 1000
duration = 1000.   # duration of the trial
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

# Monosynapse parameters
tauS = 3*ms                # synaptic conductance time constant
Esyn = 0*mV                # synapse reversal potential (mu+20*mV)
PSC = 60*pA*2                # post-synaptic current ammplitude
print("PSP amplitude: ",PSC/(120.*pA),"(mV)")
g0 = PSC/(Esyn-muI)
print("Synaptic peak conductance: ",g0/nsiemens,"(nSiemens)")
latency = 0*ms             # physiological value: 1.5 ms

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
dV/dt = (-V+mu+sigmaI*I-weight*g0/gm*gsyn*(V-Esyn))/tau : volt  
dtheta/dt = (-theta+Vt_int+alpha_int*(V-Vi_int)*int(V>=Vi_int))/tauTh_int : volt 
dgsyn/dt = -gsyn/tauS : 1
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
target.gsyn = 0
target.I = 2*rand()-1
target.weight[:int(Ntrial/2.)] = .5
target.weight[int(Ntrial/2.):] = 1.
#--Synaptic connection
synaptic = Synapses(reference,target,
             on_pre='''
             gsyn += 1
             ''')
synaptic.connect(i=arange(Ntrial),j=arange(Ntrial))
synaptic.delay = latency
#-----Record variables
Sref = SpikeMonitor(reference)
Starg = SpikeMonitor(target)
Msyn = StateMonitor(target,('gsyn','weight'),record=True)

run(duration*ms)

# Representing the basic recorded variables
nbpoint = len(Msyn.t)
time = np.arange(0,Ntrial*duration,time_step)
strength = reshape(Msyn.weight*Msyn.gsyn,Ntrial*nbpoint) 
FigGsyn = figure()
plot(time,strength,'k')

# Define the target train
train_ref = sort(Sref.i*duration+Sref.t/ms)
train_targ = sort(Starg.i*duration+Starg.t/ms)
train = append(train_ref,train_targ)
cell = int64(append(zeros(len(train_ref)),ones(len(train_targ))))

ind_ref = np.where(train_ref <= Ntrial*duration/2.)[0]
ind_targ = np.where(train_targ <= Ntrial*duration/2.)[0]

save('parameters_dynamic.npy',np.array([Ntrial,duration,time_step,period]))
save('train_ref_dynamic.npy',train_ref)
save('train_targ_dynamic.npy',train_targ)
FigGsyn.savefig('Figure_gsyn-trace.eps')

# Compute the CCG of the spike trains
lagmax = 50.
bine = 1.
ind_sort = np.argsort(train)
st = train[ind_sort]*.001
sc = cell[ind_sort]
Craw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(Craw[0,1]))-len(Craw[0,1])/2)*bine
FigCCG = figure()
xlim(-lagmax/2.,lagmax/2.)
title('Cross-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Firing rate (Hz)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
plot(lag,Craw[0,1]/(len(train_ref)*bine*.001),'.-k')
show()
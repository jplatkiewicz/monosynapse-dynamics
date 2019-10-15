from brian2 import *
from ccg import correlograms


# Simulation parameters
Ntrial = 1000*10
duration = 1000.*10    # duration of the trial
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
PSC = 25*pA                # post-synaptic current ammplitude
g0 = PSC/(Esyn-muI)
print("Synaptic peak conductance: ",g0/nsiemens,"(siemens)")
latency = 0*ms             # physiological value: 1.5 ms
Nphase = 10.
phase = duration/Nphase
wmin = .5
wmax = 5
weight_value = np.random.permutation(linspace(wmin,wmax,Nphase))
print(g0/gm*weight_value)
# Conversion synaptic conductance to PSP amplitude
FigPSP = figure()
xlabel('Normalized Conductance',fontsize=18)
ylabel('PSP Amplitude (mV)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
plot(g0/gm*weight_value,PSC*weight_value/(120.*pA),'ok')
figure()
plot(PSC*weight_value,PSC*weight_value/(120.*pA),'ok')

# Neuron model: Leaky Integrate-and-Fire
#-----
#-Integrate-and-fire neuron model with monosynapse
eqs_ref = Equations('''
dV/dt = (-V+mu+sigmaI_pyr*I)/tau : volt  
dtheta/dt = (-theta+Vt_pyr+alpha_pyr*(V-Vi_pyr)*int(V>=Vi_pyr))/tauTh_pyr : volt  
I : 1 (linked)
mu : volt
''')
eqs_ref0 = Equations('''
dV/dt = (-V+mu+sigmaI_pyr*I)/tau : volt  
dtheta/dt = (-theta+Vt_pyr+alpha_pyr*(V-Vi_pyr)*int(V>=Vi_pyr))/tauTh_pyr : volt  
I : 1 (linked)
mu : volt (linked)
''')
eqs_refnoise = Equations('''
dx/dt = -x/tauI+(2/tauI)**.5*xi : 1
''') 
eqs_targ = Equations('''
dV/dt = (-V+mu+sigmaI*I-g0/gm*gsyn*(V-Esyn))/tau : volt 
dtheta/dt = (-theta+Vt_int+alpha_int*(V-Vi_int)*int(V>=Vi_int))/tauTh_int : volt  
I : 1 (linked)
mu : volt (linked)
#-Monosynaptic input
dgsyn/dt = -gsyn/tauS : 1
''')
eqs_targnoise = Equations('''
dx/dt = -x/tauI+(2/tauI)**.5*xi : 1
''')
#----
#--
#-----Specify the synapse-on model 
#--
#----
reference = NeuronGroup(Ntrial,model=eqs_ref,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
target = NeuronGroup(Ntrial,model=eqs_targ,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
reference.run_regularly('''mu = xmin+(xmax-xmin)*rand()''',dt=period*ms)
target.mu = linked_var(reference,'mu')
ref_noise = NeuronGroup(Ntrial,model=eqs_refnoise,threshold='x>10**6',reset='x=0',method='euler')
targ_noise = NeuronGroup(Ntrial,model=eqs_targnoise,threshold='x>10**6',reset='x=0',method='euler')
reference.I = linked_var(ref_noise,'x')
target.I = linked_var(targ_noise,'x')
#-----Parameter initialization
reference.V = (xmax-xmin)*rand()+xmin
target.V = (xmax-xmin)*rand()+xmin
reference.theta = Vt_pyr+alpha_pyr*(reference.V-Vi_pyr)*np.sign(reference.V-Vi_pyr)
target.theta = Vt_int+alpha_int*(target.V-Vi_int)*np.sign(target.V-Vi_int)
target.gsyn = 0
ref_noise.x = 2*rand(Ntrial)-1
targ_noise.x = 2*rand(Ntrial)-1
#--Synaptic connection
weight = TimedArray(weight_value,dt=phase*ms)
synaptic = Synapses(reference,target,
             '''w = weight(t) : 1''',
             on_pre='''
             gsyn += w
             ''')
synaptic.connect(i=arange(Ntrial),j=arange(Ntrial))
synaptic.delay = latency
#----
#--
#----Specify the synpase-off model
#--
#----
reference0 = NeuronGroup(Ntrial,model=eqs_ref0,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
target0 = NeuronGroup(Ntrial,model=eqs_targ,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
reference0.mu = linked_var(reference,'mu')
target0.mu = linked_var(reference,'mu')
reference0.I = linked_var(ref_noise,'x')
target0.I = linked_var(targ_noise,'x')
#-----Parameter initialization
reference0.V = reference.V
target0.V = target.V
reference0.theta = reference.theta
target0.theta = target.theta
target0.gsyn = 0
#-----Record variables
Sref = SpikeMonitor(reference)
Starg = SpikeMonitor(target)
Msyn = StateMonitor(synaptic,'w',record=0)
Sref0 = SpikeMonitor(reference0)
Starg0 = SpikeMonitor(target0)

run(duration*ms)

# Representing the basic recorded variables

FigW = figure()
xlabel('Time (ms)')
ylabel('Synaptic weight')
title('Target cell')
plot(Msyn.t/ms,Msyn.w[0],'k')

# Basic firing parameters
print("WITH SYNAPSE")
print("# spikes/trial - Reference train: ",mean(Sref.count),std(Sref.count))
print("# spikes/trial - Target train: ",mean(Starg.count),std(Starg.count))
print('Average firing rate - Reference train: ',sum(Sref.count)/(Ntrial*duration*.001))
print('Average firing rate - Target train: ',sum(Starg.count)/(Ntrial*duration*.001))
print("WITHOUT SYNAPSE")
print("# spikes/trial - Reference train: ",mean(Sref.count),std(Sref.count))
print("# spikes/trial - Target train: ",mean(Starg.count),std(Starg.count))
print('Average firing rate - Reference train: ',sum(Sref.count)/(Ntrial*duration*.001))
print('Average firing rate - Target train: ',sum(Starg.count)/(Ntrial*duration*.001))


# Organize the spike times into two long spike trains

#--WITH SYNAPSE--
train_ref = sort(Sref.t/ms+floor(Sref.t/(ms*phase))*(-1+Ntrial)*phase+Sref.i*phase)
train_targ = sort(Starg.t/ms+floor(Starg.t/(ms*phase))*(-1+Ntrial)*phase+Starg.i*phase)
train = append(train_ref,train_targ)
cell = int64(append(zeros(len(train_ref)),ones(len(train_targ))))
#--WITHOUT SYNAPSE--
train_ref0 = sort(Sref0.t/ms+floor(Sref0.t/(ms*phase))*(-1+Ntrial)*phase+Sref0.i*phase)
train_targ0 = sort(Starg0.t/ms+floor(Starg0.t/(ms*phase))*(-1+Ntrial)*phase+Starg0.i*phase)
train0 = append(train_ref0,train_targ0)
cell0 = int64(append(zeros(len(train_ref0)),ones(len(train_targ0))))

save('parameters.npy',np.array([Ntrial,duration,period,Fs,Nphase,gm,g0]))
save('weights.npy',weight_value)
save('train_ref_static.npy',train_ref)
save('train_targ_static.npy',train_targ)
save('train_ref0_static.npy',train_ref0)
save('train_targ0_static.npy',train_targ0)

# Compute the CCG between the two neurons
lagmax = 100.                   #- in (ms)
bine = 1.                       #- in (ms)
#--WITH SYNAPSE--
ind_sort = np.argsort(train)
st = train[ind_sort]*.001
sc = cell[ind_sort]
Craw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(Craw[0,1]))-len(Craw[0,1])/2)*bine
#--WITHOUT SYNAPSE--
ind_sort = np.argsort(train0)
st = train0[ind_sort]*.001
sc = cell0[ind_sort]
Craw0 = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)

# Represent the ACGs and the CCG
FigACG = figure()
title('Auto-correlograms',fontsize=18)
xlim(-lagmax/2.,lagmax/2.)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Raw count',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
plot(lag,Craw[0,0],'.-k')
plot(lag,Craw[1,1],'.-b')
plot(lag,Craw0[1,1],'.-c')
FigCCG = figure()
xlim(-lagmax/2.,lagmax/2.)
title('Cross-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Raw count',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
plot(lag,Craw[0,1]/(len(train_ref)*bine*.001),'.-k')
show()
from brian2 import *
from scipy import stats
from ccg import correlograms


# Simulation parameters
Ntrial = 1000*5#*10*5
duration = 1000.#*100#*5#    # duration of the trial
time_step = 0.1            #-in (ms)
defaultclock.dt = time_step*ms  
Fs = 1/(time_step*.001)    #-in (Hz)

print("# trials:",Ntrial,"Trial duration: ",duration,"(ms)")

# Biophysical neuron parameters
#--Neuron parameters
cm = 250*pF               # membrane capacitance
gm = 25*nS                # membrane conductance
tau = cm/gm#/10           # membrane time constant
#print("membrane time constant: ",tau/ms,"(ms)")
El = -70*mV               # resting potential
#Vt = El+20*mV             # spike threshold
#print("Threshold voltage: ",Vt/mV,"(mV)")
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
mu_0 = muI
sigma_0 = (xmax-xmin)/2.
tau_0 = period*ms

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
mu : volt (linked)
''')
eqs_input = Equations('''
xx = mu_0 + sigma_0*x : volt
dx/dt = (-x+y)/tau_0 : 1
dy/dt = -y/tau_0+(2/tau_0)**.5*xi : 1
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
# reference.run_regularly('''mu = xmin+(xmax-xmin)*rand()''',dt=period*ms)
# target.mu = linked_var(reference,'mu')
signal = NeuronGroup(Ntrial,model=eqs_input,threshold='x>10**6',reset='x=0',refractory=0*ms,method='euler') 
reference.mu = linked_var(signal, 'xx')
target.mu = linked_var(signal, 'xx')
#-----Parameter initialization
reference.V = (xmax-xmin)*rand()+xmin
reference.theta = Vt_pyr+alpha_pyr*(reference.V-Vi_pyr)*np.sign(reference.V-Vi_pyr)
reference.I = 2*rand()-1
signal.x = numpy.random.randn(Ntrial)
signal.y = numpy.random.randn(Ntrial)
target.V = (xmax-xmin)*rand()+xmin
target.theta = Vt_int+alpha_int*(target.V-Vi_int)*np.sign(target.V-Vi_int)
target.I = 2*rand()-1
target.gsyn = 0
target.weight[:int(Ntrial/2.)] = 1.
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
Mref = StateMonitor(reference,('V','theta', 'mu'),record=0) #('V','I','mu'),record=True) 
Mtarg = StateMonitor(target,('V','theta','mu'),record=0) #('V','I'),record=True) 
#MrefI = StateMonitor(reference,('I'),record=0) 
#MtargI = StateMonitor(target,('I'),record=0) 

run(duration*ms)

print("# spikes/trial - Reference train: ",mean(Sref.count),std(Sref.count))
print("# spikes/trial - Target train: ",mean(Starg.count),std(Starg.count))
print('Average firing rate - Reference train: ',sum(Sref.count)/(Ntrial*duration*.001))
print('Average firing rate - Target train: ',sum(Starg.count)/(Ntrial*duration*.001))

# Representing the basic recorded variables
figure()
#xlabel('Time (ms)')
#ylabel('Potential (mV)')
#title('Membrane')
#xlim([500,1000])
#plot(M.t/ms,Vt*ones(n)/mV,'--r')
plot(Mref.t/ms,Mref.theta[0]/mV,'r')
plot(Mref.t/ms,Mref.V[0]/mV,'k')
figure()
#xlabel('Time (ms)')
#ylabel('Potential (mV)')
#title('Membrane')
#xlim([500,1000])
#plot(M.t/ms,Vt*ones(n)/mV,'--r')
plot(Mtarg.t/ms,Mtarg.theta[0]/mV,'r')
plot(Mref.t/ms,Mref.V[0]/mV,'b')
plot(Mtarg.t/ms,Mtarg.V[0]/mV,'k')
#FigIref = figure()
#xlabel('Time (ms)')
#ylabel('Potential (mV)')
#title('Input')
#xlim([500,1000])
#plot(Mref.t/ms,MrefI.I[0],'k')
#FigItarg = figure()
#xlabel('Time (ms)')
#ylabel('Potential (mV)')
#title('Input')
#xlim([500,1000])
#plot(Mtarg.t/ms,MtargI.I[0],'k')
#show()

# Define the target train
train_ref = sort(Sref.i*duration+Sref.t/ms)
train_targ = sort(Starg.i*duration+Starg.t/ms)
train = append(train_ref,train_targ)
cell = int64(append(zeros(len(train_ref)),ones(len(train_targ))))

print('CV - Reference train: ',std(diff(train_ref))/mean(diff(train_ref)))
print('CV - Target train: ',std(diff(train_targ))/mean(diff(train_targ)))

# Compute the CCG of the spike trains
lagmax = 50.
bine = .1
ind_sort = np.argsort(train)
st = train[ind_sort]*.001
sc = cell[ind_sort]
Craw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(Craw[0,1]))-len(Craw[0,1])/2)*bine
FigACGpre = figure()
xlim(-lagmax/2.,lagmax/2.)
title('Auto-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Spike Transmission ',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
xlim(-lagmax/2.,lagmax/2.)
bar(lag,Craw[0,0]/(len(train_ref)*bine*.001),color='k')
FigACGpost = figure()
xlim(-lagmax/2.,lagmax/2.)
title('Auto-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Firing rate (Hz)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
bar(lag,Craw[1,1]/(len(train_targ)*bine*.001),color='k')
FigCCG = figure()
title('Cross-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Firing rate (Hz)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
xlim(-lagmax/2.,lagmax/2.)
bar(lag,Craw[0,1]/(len(train_ref)*bine*.001),bine,align='center',color='k',edgecolor='k')
FigCCGzoom = figure()
title('Zoom',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Spike Transmission Probability',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
xlim(0,5)
bar(lag,Craw[0,1]/(len(train_ref)*1.),bine,align='center',color='k',edgecolor='k')
show()

FigCCG.savefig('Fig_CCG.eps')
FigCCGzoom.savefig('Fig_CCG_zoom.eps')
#save('train_ref.npy',train_ref)
#save('train_targ.npy',train_targ)
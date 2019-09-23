from brian2 import *
from scipy import stats
from ccg import correlograms

# Simulation parameters
Ntrial = 20#1000
duration = 100000.*5#    # duration of the trial
time_step = 0.1            #-in (ms)
defaultclock.dt = time_step*ms  
Fs = 1/(time_step*.001)    #-in (Hz)

print("# trials:",Ntrial,"Trial duration: ",duration,"(ms)")

# Biophysical neuron parameters
#--Neuron parameters
cm = 250*pF               # membrane capacitance
gm = 25*nS                # membrane conductance
tau = cm/gm               # membrane time constant
#print("membrane time constant: ",tau/ms,"(ms)")
El = -70*mV               # resting potential
Vt = El+20*mV             # spike threshold
#print("Threshold voltage: ",Vt/mV,"(mV)")
Vr = El+10*mV             # reset value
refractory_period = 0*ms  # refractory period

# Biophysical background input parameters
muI = Vt-10*mV
sigma = 5*mvolt
xmin = muI-5*mV #7.5
xmax = muI+5*mV #7.5
period = 5.    #5

# Neuron model: Leaky Integrate-and-Fire
#-----
#-Integrate-and-fire neuron model 
#-----
eqs_ref = Equations('''
dV/dt = (-V+mu+sigma*tau**.5*xi)/tau : volt 
mu : volt 
''')
eqs_targ = Equations('''
dV/dt = (-V+mu+sigma*tau**.5*xi)/tau : volt 
mu : volt (linked)
''')
#-----Specify the model
reference = NeuronGroup(Ntrial,model=eqs_ref,threshold='V>Vt',reset='V=Vr',refractory=refractory_period,method='euler')
target = NeuronGroup(Ntrial,model=eqs_targ,threshold='V>Vt',reset='V=Vr',refractory=refractory_period,method='euler')
reference.run_regularly('''mu = xmin+(xmax-xmin)*rand()''',dt=period*ms)
target.mu = linked_var(reference,'mu')
#-----Parameter initialization
reference.V = (xmax-xmin)*rand()+xmin
target.V = (xmax-xmin)*rand()+xmin
#-----Record variables
Sref = SpikeMonitor(reference)
Starg = SpikeMonitor(target) 

run(duration*ms)

print("# spikes/trial - Reference train: ",mean(Sref.count),std(Sref.count))
print("# spikes/trial - Target train: ",mean(Starg.count),std(Starg.count))
print('Average firing rate - Reference train: ',sum(Sref.count)/(Ntrial*duration*.001))
print('Average firing rate - Target train: ',sum(Starg.count)/(Ntrial*duration*.001))

# Define the target train
train_ref = sort(Sref.i*duration+Sref.t/ms)
train_targ = sort(Starg.i*duration+Starg.t/ms)
train = append(train_ref,train_targ)
cell = int64(append(zeros(len(train_ref)),ones(len(train_targ))))

print('CV - Reference train: ',std(diff(train_ref))/mean(diff(train_ref)))
print('CV - Target train: ',std(diff(train_targ))/mean(diff(train_targ)))

# Compute the CCG of the spike trains
lagmax = 50.
bine = 1.
ind_sort = np.argsort(train)
st = train[ind_sort]*.001
sc = cell[ind_sort]
Craw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(Craw[0,1]))-len(Craw[0,1])/2)*bine
FigACG = figure()
xlim(-lagmax/2.,lagmax/2.)
title('Auto-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Firing rate (Hz)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
bar(lag,Craw[0,0]/(len(train_ref)*bine*.001),bine,align='center',color='k',edgecolor='k')
FigCCG = figure()
title('Cross-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Firing rate (Hz)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
xlim(-lagmax/2.,lagmax/2.)
bar(lag,Craw[0,1]/(len(train_ref)*bine*.001),bine,align='center',color='k',edgecolor='k')
show()

save('parameters_var.npy',np.array([Ntrial,duration,period,Fs]))
save('train_ref_var.npy',train_ref)
save('train_targ_var.npy',train_targ)
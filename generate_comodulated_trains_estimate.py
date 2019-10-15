'''
Generate a pair of spike trains from two neuron models, over mutliple trials.
The model comprised two leaky integrate-and-fire neurons 
that are NOT monosynaptically connected but are driven by a common nonstationary signal.
In addition, they both receive independent background inputs.
The nonstationary signal and background inputs are different from one trial to another.
'''

from brian2 import *
from ccg import correlograms

# Define simulation parameters 
Ntrial = 100                    # Number of trials
duration = 100000.              # Duration of a trial in (ms)
time_step = 0.1                 
defaultclock.dt = time_step*ms   # Time step of equations integration 
Fs = 1/(time_step*.001)          #-in (Hz)

print("# trials: ",Ntrial,"Trial duration: ",duration,"(ms)")

# Define biophysical parameters 
cm = 250*pF                     # Membrane capacitance
gm = 25*nS                      # Membrane conductance
tau = cm/gm                     # Membrane time constant
El = -70*mV                     # Resting potential
Vt = El+20*mV                   # Spike threshold
Vr = El+10*mV                   # Reset voltage
refractory_period = 0*ms        # Refractory period

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

print("background input time constant: ",tauI/ms,"(ms)","Input average amplitude: ",muI/mV,"(mV)","Input amplitude range:",.1*floor((xmax-xmin)/mV/.1),"(mV)","Input standard-deviation",sigmaI/mV,"(mV)","Interval duration: ",period,"(ms)")

# Neuron model: Leaky Integrate-and-Fire
#-----
#-Integrate-and-fire neuron model with monosynapse
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
dV/dt = (-V+mu+sigmaI*I)/tau : volt 
dtheta/dt = (-theta+Vt_int+alpha_int*(V-Vi_int)*int(V>=Vi_int))/tauTh_int : volt  
weight : 1
dI/dt = -I/tauI+(2/tauI)**.5*xi : 1
mu : volt (linked)
''')
#-----Specify the model
reference = NeuronGroup(Ntrial,model=eqs_ref,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
target = NeuronGroup(Ntrial,model=eqs_targ,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
signal = NeuronGroup(Ntrial,model=eqs_input,threshold='x>10**6',reset='x=0',refractory=0*ms,method='euler') 
reference.mu = linked_var(signal, 'xx')
target.mu = linked_var(signal, 'xx')
#-----Parameter initialization
reference.V = (xmax-xmin)*rand()+xmin
reference.theta = Vt_pyr+alpha_pyr*(reference.V-Vi_pyr)*np.sign(reference.V-Vi_pyr)
reference.I = 2*rand()-1
target.V = (xmax-xmin)*rand()+xmin
target.theta = Vt_int+alpha_int*(target.V-Vi_int)*np.sign(target.V-Vi_int)
target.I = 2*rand()-1
signal.x = numpy.random.randn(Ntrial)
signal.y = numpy.random.randn(Ntrial)
#-----Record variables
Sref = SpikeMonitor(reference)
Starg = SpikeMonitor(target)

run(duration*ms)

# Organize the spike times into two long spike trains
train_ref = unique(Sref.i*duration+Sref.t/ms)
train_targ = unique(Starg.i*duration+Starg.t/ms)
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
plt.show()

save('parameters.npy',np.array([Ntrial,duration,period,Fs]))
save('train_ref.npy',train_ref)
save('train_targ.npy',train_targ)
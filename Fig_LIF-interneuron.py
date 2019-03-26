from brian2 import *
from scipy import stats,signal,optimize
from pandas import Series
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
tau = cm/gm           # membrane time constant
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

# Biophysical background input parameters
tauI = 10*ms
muI = -50*mV
sigmaI = 2*mvolt 
xmin = muI-5*mV
xmax = muI+5*mV
period = 10.
print("Mean input: ",muI/mV,"(mV)")
mu_0 = muI
sigma_0 = (xmax-xmin)/4.
tau_0 = period*ms

# Neuron model: Leaky Integrate-and-Fire
#-----
#-Integrate-and-fire neuron model 
#-----
eqs = Equations('''
#-Potential
dV/dt = (-V+mu+sigmaI*I)/tau : volt 
dtheta/dt = (-theta+Vt+alpha*(V-Vi)*int(V>=Vi))/tau_th : volt  
dI/dt = -I/tauI+(2/tauI)**.5*xi : 1
mu : volt (linked) 
''')
eqs_input = Equations('''
xx = mu_0 + sigma_0*x : volt
dx/dt = (-x+y)/tau_0 : 1
dy/dt = -y/tau_0+(2/tau_0)**.5*xi : 1
''')
#-----Specify the model
neuron = NeuronGroup(Ntrial,model=eqs,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
# neuron.run_regularly('''mu = xmin+(xmax-xmin)*rand()''',dt=period*ms)
signal = NeuronGroup(Ntrial,model=eqs_input,threshold='x>10**6',reset='x=0',refractory=0*ms,method='euler') 
neuron.mu = linked_var(signal, 'xx')
#-----Parameter initialization
neuron.V = (xmax-xmin)*rand(Ntrial)+xmin
neuron.I = 2*rand()-1
neuron.theta = Vt+alpha*(neuron.V-Vi)*np.sign(neuron.V-Vi)
signal.x = numpy.random.randn(Ntrial)
signal.y = numpy.random.randn(Ntrial)
#-----Record variables
S = SpikeMonitor(neuron)
M = StateMonitor(neuron,('V','theta', 'mu'),record=0)  
Mmu = StateMonitor(neuron,'mu',dt=period*ms,record=True)

run(duration*ms)

# Representing the basic recorded variables
FigVm = figure()
xlabel('Time (ms)')
ylabel('Potential (mV)')
title('Membrane')
#xlim([500,1000])
plot(M.t/ms,M.theta[0]/mV,'-r')
plot(M.t/ms,M.V[0]/mV,'k')
#show()

# Define the target train
train = sort(S.i*duration+S.t/ms)

# Basic firing parameters
print("# spikes: ",len(train),
      "Average firing rate",len(train)/(Ntrial*duration*.001),"(Hz)",
      "CV",std(diff(train))/mean(diff(train)))

# Compute the ACG of the spike train
lagmax = 50.
bine = .1
ind_sort = np.argsort(train)
st = train[ind_sort]*.001
cell = int64(zeros(len(train)))
sc = cell[ind_sort]
Araw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,window_size=lagmax/1000.)
lag = (np.arange(len(Araw[0,0]))-len(Araw[0,0])/2)*bine
FigACG = figure()
xlim(-lagmax/2.,lagmax/2.)
title('Auto-correlogram',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Firing rate (Hz)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
bar(lag,Araw[0,0]/(len(train)*bine*.001),bine,align='center',color='k',edgecolor='k')
show()

# Show the effect of modulating the signal onto the firing rate
rate = bincount(int64(floor(train/period)),minlength=int(Ntrial*duration/period))#/(period*.001)
amplitude = reshape(Mmu.mu/mV,Ntrial*len(Mmu.t))
FigMu = figure()
plot(amplitude,rate,'.k')
x = []
y = []
for k in numpy.unique(rate):
	x.append(numpy.mean(amplitude[rate == k]))
	y.append(k)
plt.plot(x, y , 'o-r')
show()

FigVm.savefig('Fig-int-Vm-trace.eps')
FigACG.savefig('Fig-int-ACG.eps')
FigMu.savefig('Fig-int-mu-count.eps')
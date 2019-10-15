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
tau = cm/gm           # membrane time constant
El = -70*mV               # resting potential
Vr = El+10*mV             # reset value
refractory_period = 0*ms  # refractory period

# Adaptive threshold parameters
tau_th = 7*ms
Vi = -60*mV
ka = 5*mV
ki = ka/1.
alpha = ka/ki
Vt = Vi+5*mV

# Biophysical background input parameters
tauI = 10*ms
muI = -50*mV
sigmaI = 6*mvolt 
xmin = muI-5*mV
xmax = muI+5*mV
period = 10.
print("Mean input: ",muI/mV,"(mV)")

# Neuron model: Leaky Integrate-and-Fire
#-----
#-Integrate-and-fire neuron model 
#-----
eqs = Equations('''
#-Potential
dV/dt = (-V+mu+sigmaI*I)/tau : volt 
dtheta/dt = (-theta+Vt+alpha*(V-Vi)*int(V>=Vi))/tau_th : volt  
dI/dt = -I/tauI+(2/tauI)**.5*xi : 1
mu : volt 
''')
#-----Specify the model
neuron = NeuronGroup(Ntrial,model=eqs,threshold='V>theta',reset='V=Vr',refractory=refractory_period,method='euler')
neuron.run_regularly('''mu = xmin+(xmax-xmin)*rand()''',dt=period*ms)
#-----Parameter initialization
neuron.V = (xmax-xmin)*rand(Ntrial)+xmin
neuron.I = 2*rand()-1
neuron.theta = Vt+alpha*(neuron.V-Vi)*np.sign(neuron.V-Vi)
#-----Record variables
S = SpikeMonitor(neuron)
Mth = StateMonitor(neuron,'theta',record=0)  
M = StateMonitor(neuron,'V',record=True)  

run(duration*ms)

# Representing the basic recorded variables
FigVm = figure()
xlabel('Time (ms)')
ylabel('Potential (mV)')
title('Membrane')
plot(M.t/ms,Mth.theta[0]/mV,'-r')
plot(M.t/ms,M.V[0]/mV,'k')

# Show the auto-correlation function for Vm
lagmax = 25.
V = reshape(M.V/mV,Ntrial*len(M.t))
m_max = int(lagmax/time_step)
n = len(V)
V_avg = mean(V)
ACF = zeros(2*m_max)
for k in range(2*m_max):
    m = k-m_max
    ACF[k] = sum((V[abs(m):]-V_avg)*(V[:n-abs(m)]-V_avg))/n
lag = arange(-m_max,m_max)*time_step
gamma = sum((V-V_avg)**2)/n
normalization = sqrt(gamma**2)
ACF = ACF/normalization
FigACF = figure()
xlabel('Time lag (ms)')
ylabel('Potential (mV)')
title('Pyramidal Vm ACF')
xlim(-lagmax,lagmax)
plot(lag,ACF,'-k')

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
title('Auto-correlogram of Pyramidal Cell',fontsize=18)
xlabel('Time lag  (ms)',fontsize=18)
ylabel('Firing rate (Hz)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
bar(lag,Araw[0,0]/(len(train)*bine*.001),bine,align='center',color='k',edgecolor='k')

# Show Vm STA
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
Vsta = STA(train,V,lagmax,period,time_step)
FigSTA = figure()
title('Spike-triggered average of Pyramidal $V_m$',fontsize=18)
xlabel('Time Lag (ms)',fontsize=18)
ylabel('Potential (mV)',fontsize=18)
xticks(fontsize=18)
yticks(fontsize=18)
xlim(-lagmax,lagmax)
lag = linspace(-lagmax,lagmax,len(Vsta))
plot(lag,Vsta,'-k')
plot(zeros(2),[min(Vsta),max(Vsta)],'--k',linewidth=2)
show()

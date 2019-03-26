'''
Python command lines for generating
all the figures in the manuscript entitled,
Can we infer fast monosynaptic dynamics from spike times in vivo?
'''

#-----------
# Figure 1
#----------- 
#”-- Panel A: Pyramidal Neuron's Firing Behavior 
python Fig_LIF-pyramidal.py
#”-- Panel A: Pyramidal Neuron's Membrane Potential Dynamics
python Fig_LIF-pyramidal-Vm.py
#”-- Panel A: Interneuron's Firing Behavior 
python Fig_LIF-interneuron.py
#”-- Panel A: Interneuron's Membrane Potential Dynamics
python Fig_LIF-interneuron-Vm.py

#-- Panel B: Pyramidal Cell - Interneuron's Co-Firing Behavior 
python Fig_LIFs-pyramidal-interneuron-comodulation.py
#”-- Panel B: Pyramidal Cell - Interneuron's Membrane Potential Dynamics
python Fig_LIFs-pyramidal-interneuron-comodulation-Vm.py

#-----------
# Figure 2
#----------- 
#”-- Panel A: Pyramidal Cell Receiving  Interneuron's Spikes via an Instantaneous Synapse
python generate_comodulated_trains.py
python Fig_instantaneous_monosynapse_CCG.py
#”-- Panel B: Pyramidal Cell Receiving Spikes v]ia a Synaptic Conductance
python Fig_ultraprecise_monosynapse_PSTH.py
#-- Panel C: Pyramidal Cell Receiving  Interneuron's Spikes via a Synaptic Conductance
python Fig_ultraprecise_monosynapse_CCG.py

#-----------
# Figure 3
#----------- 
#”-- Panel A: Estimating the amount of injected synchrony and characterizing its variability
python generate_comodulated_trains_estimate.py
python Fig_estimate_variability.py
#”-- Panel B: Estimating the amount of injected synchrony for a wide range of values
python Fig_estimate_range.py
#”-- Panel C: Bounding the amount of injected synchrony with a confidence interval
python Fig_estimate_confidence-interval.py

#-----------
# Figure 4
#----------- 
#”-- Panel A: Inferring a fixed monosynaptic conductance from spike times
python generate_comodulated_trains_static-conductance.py
python Fig_estimate_static-conductance.py
#”-- Panel B: Inferring the time course of a monosynaptic conductance from spike times
python generate_comodulated_trains_dynamic-conductance.py
python Fig_estimate_dynamic-conductance.py

#-----------
# Figure 5
#----------- 
#”-- Panel A: Comparing the detection of a synapse with a bootstrap and the jitter methods
python generate_comodulated_trains_with-injection.py
python Fig_compare_ROC.py
#”-- Panel B: Comparing the null with a bootstrap and the jitter methods
python generate_comodulated_trains_synchrony-distribution.py
python Fig_compare_distribution.py

#-----------
# Figure 6
#----------- 
#”-- Panel A: Assessing the influence of the jitter timescale on the injected synchrony estimation
python generate_comodulated_trains_with-injection.py
python Fig_estimate_synapse_timescale-influence.py
#”-- Panel B: Assessing the influence of the jitter timescale on the injected synchrony estimation
python generate_comodulated_trains_synchrony-distribution.py
python Fig_detect_synapse_timescale-influence.py










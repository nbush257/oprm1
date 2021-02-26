import sys
import sklearn.preprocessing
import sklearn.decomposition
import click
from itertools import product
from brian2 import *
import os
import scipy.signal
import pickle
import seaborn as sns
import brian_utils.model_IO as bio
import brian_utils.postproc as bup
import time
import itertools
'''
The goal of this simulation is to implement both the
hyperpolarizng and synaptic shut-off effect on OPRM1 neurons


Assume all OPRM1+ neurons are excitatory
Assume all neurons respond in the same graded fashion to DAMGO

We introduce a hyperpolarizing current (I_opioid(t)) and a synaptic conductance (g_synopioid).
damgo is a TimedArray of units amps


'''
def set_ics(neurons):
    neurons.v = -58*mV
    neurons.h = 0.1
    neurons.n = 0.1
    neurons.g_syne = 0 * nS
    neurons.g_syni = 0 *nS

def remap_inhibitory_tonic_to_quiescent(gl_vals,n_inh,t_val=0.5,q_val=1.2):
    '''
    Finds all the inhibitory tonic neurons and turns them into quiescent neurons. Takes the same number of
    excitatory quiescent neurons and turns them into tonic neurons

    This does not change the number of B/T/Q neurons in the overall population
    :param gl_vals:
    :param n_inh:
    :param t_val:
    :param q_val:
    :return:
    '''
    mask = np.where(gl_vals[:n_inh]==t_val*nS)[0]
    n_switch = len(mask)
    gl_vals[:n_inh][mask] = q_val*nS
    idx = np.where(gl_vals[n_inh:]==q_val*nS)[0]
    sub_idx = np.random.choice(idx,n_switch,replace=False)
    gl_vals[n_inh:][sub_idx] = t_val * nS

    return(gl_vals)






def main(basename,idx,n_neurons=300,k_avg=6,we_max=4,wi_max=4,frac_inh=0.2,frac_oprm1=0.5,pt=0.4,pb=0.05,hyp_opioid=10,syn_shut=1,save_tgl=True,run_seed=0):
    prefix = f'{basename}_hyp{hyp_opioid:0.1f}_synshut{syn_shut:0.2f}_run{run_seed:0.0f}_pt{pt:0.1f}_pb{pb:0.2f}_syn{we_max:0.1f}'
    seed(run_seed)
    np.random.seed(run_seed)
    N=n_neurons
    perturbation_dt = 20*second
    synblock = 1 # binary to block the synapses at the end
    pQ = 1-pt-pb # Quiescent neurons are left over
    gl_vals = np.random.choice([0.5, 0.7, 1.2], N, p=[pt, pb, pQ]) * nS
    #Extract params from name
    # Compute gl_vals for tonic_excitatory only case

    eqs_yaml = 'harris_eqs_oprmv1.yaml'
    ns_yaml = 'harris_ns_oprmv1.yaml'
    pct_connect = (k_avg / 2) / (N - 1)

    eqs = bio.import_eqs_from_yaml(eqs_yaml)
    ns = bio.import_namespace_from_yaml(ns_yaml)
    n_inh = int(np.ceil(N*frac_inh))
    n_oprm1 = int((N-n_inh)*frac_oprm1)
    n_excit = N-n_inh
    #gl_vals = remap_inhibitory_tonic_to_quiescent(gl_vals,n_inh,t_val=0.5,q_val = 1.2)

    prefix = f'{prefix}_oprm1'
    savename = prefix+'.dat'
    defaultclock.dt = 0.05*ms

    print('Setting the device')
    set_device('cpp_standalone',clean=True,build_on_run=False)
    prefs.devices.cpp_standalone.openmp_threads = int(os.environ['OMP_NUM_THREADS'])

    print('Setting up neurons...')
    neurons = NeuronGroup(N, eqs['neuron_eqs'],
                          threshold='v>=-20*mV',
                          refractory=2 * ms,
                          method='rk4',
                          namespace=ns)


    # ========================= #
    # Set up synapses.
    # ========================= #
    print('Setting up synapses...')
    Pi = neurons[:n_inh]
    Pe_oprm1 = neurons[n_inh:n_oprm1+n_inh]
    Pe = neurons[n_oprm1+n_inh:]
    con_e = Synapses(Pe, neurons, model=eqs['excitatory_synapse_eqs'], method='exponential_euler', namespace=ns)
    con_oprm1 = Synapses(Pe_oprm1, neurons, model=eqs['opioid_synapse_eqs'], method='exponential_euler', namespace=ns)
    con_i = Synapses(Pi, neurons, model=eqs['inhibitory_synapse_eqs'], method='exponential_euler', namespace=ns)

    np.random.seed(run_seed)
    seed(run_seed)
    con_i.connect(p=pct_connect)
    con_e.connect(p=pct_connect)
    con_oprm1.connect(p=pct_connect)


    # =================== #
    # Set up integrator #
    # =================== #
    sensor_int = NeuronGroup(1, eqs['sensor_eqs'], method='rk4')

    # incoming connections
    con_in = Synapses(neurons, sensor_int, on_pre='v += 1')
    con_in.connect()  # fully connected

    # outgoing connections
    neurons.sensor = linked_var(sensor_int, 'v')
    sensor_int.v = 0.
    set_ics(neurons)

    # ====================== #
    # Set up cell specific parameters #
    # ====================== #
    seed(run_seed)
    np.random.seed(run_seed)
    gl_vals = gl_vals + np.random.normal(0,0.05,N) * nS
    gnap_vals = 0.8 * nS + np.random.normal(0,.05,N) * nS
    neurons.g_l = gl_vals
    neurons.g_k = 11.2 * nsiemens
    neurons.g_na = 28. * nsiemens
    neurons.g_nap = gnap_vals
    neurons.damgo_sensitivity = 0
    Pe_oprm1.damgo_sensitivity = 1



    statemon = StateMonitor(neurons, variables=['v'],
                            record=np.arange(0, N, N / 15).astype('int'), dt=1 * ms)
    ratemon = PopulationRateMonitor(neurons)
    spikemon = SpikeMonitor(neurons)
    # perturbations = StateMonitor(neurons, variables=['I_opioid','synblock','we','wi'],
    #                         record=[0], dt=10* ms)

    # =============== #
    # Set up perturbations
    # =============== #

    op_vals =   np.array([0,0,0,0,0,1,1,0,1,1,0,0])
    we =        np.array([1,1,1,1,1,1,1,1,1,1,1,1])
    wi =        np.array([1,1,0,0,1,1,1,1,0,0,1,1])

    we = TimedArray(we*we_max*nS,dt=perturbation_dt)
    wi = TimedArray(wi*wi_max*nS,dt=perturbation_dt)
    vm_opioid = TimedArray(op_vals*hyp_opioid*pA,dt = perturbation_dt)
    we_opioid = TimedArray((1-op_vals*syn_shut)*we_max*nS,dt = perturbation_dt)
    runtime = len(op_vals)*perturbation_dt

    seed(run_seed)
    np.random.seed(run_seed)

    # =============== #
    # Run baseline
    # =============== #
    print('Setting up run...')
    net = Network(collect())
    net.run(runtime, report='text')

    # =============== #
    # Run intrinsic
    # =============== #
    synblock = 0
    net.run(50*second,report='text')

    #====================== #
    # Build
    #====================== #

    device.build(directory=f'./{idx:04.0f}', compile=True, run=True, debug=False)

    # ============================ #
    # Save and clean up #
    # ============================ #
    if save_tgl:
        states = net.get_states()
        states['ns'] = ns
        states['eqs'] = eqs
        states['n_inh'] = n_inh
        states['n_oprm1'] = n_oprm1
        states['hyp'] = hyp_opioid
        states['syn_shut'] = syn_shut
        states['run_seed'] = run_seed
        states['perturbation_dt'] = perturbation_dt
        states['we_opioid'] = we_opioid.values
        states['vm_opioid'] = vm_opioid.values
        states['we'] = we.values
        states['wi'] = wi.values

        with open(savename,'wb') as fid:
            pickle.dump(states,fid)

    #====================== #
    # Postproc
    #====================== #
    raster, cell_id, bins = bup.bin_trains(spikemon.t, spikemon.i, neurons.N, max_time=statemon.t[-1])
    binsize = 10 * ms
    smoothed_pop_rate = ratemon.smooth_rate('gaussian', binsize)

    # =================== #
    # Plot
    # =================== #
    f = plt.figure(figsize=(5, 7))
    g = f.add_gridspec(11, 1)

    # Plot raster
    ax0 = f.add_subplot(g[:6, 0])
    plt.pcolormesh(bins, cell_id, raster / (binsize / second), cmap='gray_r')
    plt.ylabel('Neuron')

    # Plot population rate
    ax1 = f.add_subplot(g[7, 0], sharex=ax0)
    plt.plot(ratemon.t, smoothed_pop_rate, 'k', linewidth=1,alpha=0.5)
    plt.ylabel('FR\n(sp/s)')

    # Plot DAMGO
    ax1 = f.add_subplot(g[8, 0], sharex=ax0)
    tvec = np.arange(0,runtime/second,.1)*second
    ax1.plot(tvec,we_opioid(tvec)/nS, 'tab:blue', linewidth=1,alpha=0.5)
    ax1.set_ylim(0,we_max+1)
    # plt.ylabel('DAMGO (percent max block)')
    ax1.set_ylabel('$g_{syn}^{opioid}$',fontsize=8)

    ax2 = f.add_subplot(g[9, 0], sharex=ax0)
    tvec = np.arange(0,runtime/second,.1)*second
    ax2.plot(tvec,vm_opioid(tvec)/pA, 'tab:blue', linewidth=1,alpha=0.5)
    ax2.set_ylim(0,hyp_opioid)
    # plt.ylabel('DAMGO (percent max block)')
    ax2.set_ylabel('$I_{opioid}$ (pA)',fontsize=8)
    ax2.set_xlabel('Time (s)')

    # Plot we/wi
    ax3 = f.add_subplot(g[10, 0], sharex=ax0)
    tvec = np.arange(0,runtime/second,.1)*second
    ax3.plot(tvec,we(tvec)/nS, 'tab:blue', linewidth=1,alpha=0.5)
    ax3.plot(tvec,wi(tvec)/nS, 'tab:red', linewidth=1,alpha=0.5)
    ax3.set_ylim(-.1,we_max+1)
    # plt.ylabel('DAMGO (percent max block)')
    ax3.set_ylabel('Syn str (nS)',fontsize=8)
    ax3.set_xlabel('Time (s)')

    # Cleanup and save
    sns.despine()
    # plt.xlim(5,13)
    # plt.tight_layout()
    plt.xlim(10,runtime/second)
    plt.savefig(f'{prefix}_raster.png',dpi=300)
    plt.close('all')


    #plot close ups
    y_dam = ['Control','Block Inh','DAMGO','Block Inh+DAMGO']
    xstarts = perturbation_dt*[0.5,2,5,8] + 5*second
    xstops = xstarts + 30*second
    f,ax = plt.subplots(nrows=4,sharey=True)
    for ii in range(4):
        ax[ii].plot(ratemon.t, smoothed_pop_rate, 'k', linewidth=1,alpha=0.5)

    for ii in range(1,5):
        ax[ii-1].set_xlim(xstarts[ii-1]/second,xstops[ii-1]/second)
        ax[ii-1].set_ylabel(f'{y_dam[ii-1]}\n',fontsize=8)
    plt.tight_layout()
    sns.despine()
    ax[0].set_title('DAMGO application')
    ax[0].set_ylim(-1,75)
    ax[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(f'{prefix}_DAMGOapp.pdf')
    plt.close('all')

    # For intrinsic properties
    ts = spikemon.t
    spk_i = spikemon.i
    keep = ts>runtime
    ts = ts[keep]
    spk_i = spk_i[keep]

    train = bup.create_train(ts,spk_i)
    cell_class = bup.find_bursters_pk_ISI(train,N)[0]
    p_burst = np.mean(cell_class==0)
    p_tonic = np.mean(cell_class==1)
    p_quiescent = np.mean(cell_class==2)

    # f = plt.figure(figsize=(5, 7))
    # g = f.add_gridspec(6, 1)
    # ax0 = f.add_subplot(g[1:, 0])
    # for ii in range(15):
    #     ax0.plot(statemon.t,statemon.v[ii]/mV + 70*ii,'k',lw=0.5)
    # sns.despine()
    # ax0.set_xlim([runtime/second+5,runtime/second+35])
    # ax1 = f.add_subplot(g[0,0])
    # ax1.pie([p_burst,p_tonic,p_quiescent],labels=['B','T','Q'],autopct='%1.1f%%',pctdistance=2.)
    # plt.savefig(f'{prefix}_voltages.png',dpi=300)
    # plt.close('all')

    # ============== #
    # Look at a couple opioid neurons
    # ============== #
    # f,ax = plt.subplots(nrows = 2,ncols=2,figsize=(10, 6))
    # xstart = np.array([opioid_dt,opioid_dt*2])
    # xstop = xstart + 15
    # for ii in range(2):
    #     ax[0,ii].plot(ratemon.t, smoothed_pop_rate/Hz*2 + 150, 'g', linewidth=1, alpha=0.5)
    #     ax[0,ii].plot(statemon.t,statemon.v[0]/mV,'tab:blue',lw=0.5)
    #     ax[0,ii].plot(statemon.t,statemon.v[20]/mV+70,'k',lw=0.5)
    #     ax[0,ii].plot(statemon.t,statemon.v[50]/mV+140,'tab:red',lw=0.5)
    #     ax[1,ii].plot(statemon.t,statemon.I_opioid[0]/pA,'tab:blue',lw=0.5)
    #     ax[1,ii].plot(statemon.t,statemon.I_opioid[20]/pA,'k',lw=0.5)
    #     ax[1,ii].plot(statemon.t,statemon.I_opioid[50]/pA,'tab:red',lw=0.5)
    #     ax[0,ii].set_xlim(xstart[ii],xstop[ii])
    #     ax[1,ii].set_xlim(xstart[ii],xstop[ii])
    #     ax[0,ii].set_ylim(-60,250)
    #
    #
    # ax[0,0].set_ylabel('(mV)')
    # ax[1,0].set_ylabel('Opioid Current (pA)')
    # plt.legend(['Inh','OPRM1+','Excit'])
    # sns.despine()
    #
    # plt.tight_layout()
    # plt.savefig(f'{prefix}_opioidvoltage_effects.png',dpi=300)
    # plt.close('all')




    ## Clean up ##
    device.delete()


@click.command()
@click.argument('idx',type=int)
def batch(idx):
    # Maximum hyp should by 9 - no bursts above that
    # hyp_vals = np.arange(0, 9.5, 0.5)
    hyp_vals = [0,10]
    # Max syn shut is 0.8 - no bursts above that
    # syn_shut_vals = np.arange(0, 0.85, 0.05)
    syn_shut_vals = [0,1]



    # Mutliple runs with different initializations
    # seeds = np.arange(0,4)
    # seeds = [0,1,2,3,4,5,6,7]
    seeds=[0]
    params = list(itertools.product(hyp_vals, syn_shut_vals,seeds))
    main('opnet',idx,hyp_opioid=params[idx][0],syn_shut=params[idx][1],
         run_seed=params[idx][2],
         pt=0.35,
         pb=0.0,
         we_max=4.5,wi_max=4.5)

if __name__ == '__main__':
    batch()

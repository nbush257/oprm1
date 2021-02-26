from brian2 import TimedArray
import time
import pandas as pd
from brian2 import NeuronGroup,Synapses
from brian2.units import *
import numpy as np

def gen_pulse(starts, durs, mags, dt=1 * ms):
    '''
    Generate a pulse timed array
    :param starts: start times (s): an array of values to start the pulse with dimension second (or mutliple)
    :param durs: duration of pulses. Can be single value or array of same length as starts. Must have time dimension
    :param mag: Magnitude of pulses. Can be single value or array of same length as starts. Can be any dimension
    :param dt: Resolution of pulse. Does not need to be the same as the simulation clock
    :return: x - the timed array object

    ex:
    starts = [10,20,30] * second
    durs = 1 * second
    mags = [2,4,6] * namp

    x_t = gen_pulse(starts,durs,mags)

    '''

    # make sure the durations and magnitudes are arrays
    durs = [durs]
    mags = [mags]

    if len(durs)!=len(starts):
        if len(durs)!=1:
            raise ValueError(f'Number of durations{len(durs)} does not match number of starts {len(starts)}')
        else:
            durs = np.ones(len(starts))*durs[0]
    if len(mags)!=len(starts):
        if len(mags)!=1:
            raise ValueError(f'Number of magnitudes {len(mags)} does not match number of starts {len(starts)}')
        else:
            mags = np.ones(len(starts))*mags[0]

    stops = starts+durs
    Nsteps = int(np.max(stops)/dt)
    x = np.zeros(Nsteps+1)
    for start,stop,mag in zip(starts,stops,mags):
        start = int(start/dt)
        stop = int(stop/dt)
        x[start:stop] = mag
    x = TimedArray(x,dt)
    return(x)


def gen_brian2_edgelist(N,k_avg,seed=None):
    '''
    Generate an edgelist for persistence of network architecture
    :param N:
    :param k_avg:
    :return:
    '''
    if seed is None:
        np.random.seed(time.time())
    else:
        np.random.seed(seed)

    neurons = NeuronGroup(N,model='')
    syns = Synapses(neurons,neurons)
    pct_connect = (k_avg/2)/(N-1)
    syns.connect(p=pct_connect)
    ii = list(syns.i)
    jj = list(syns.j)
    edgelist = pd.DataFrame()
    edgelist['i'] = ii
    edgelist['j'] = jj
    fn = f'edgelist_N{N}_k{k_avg}.csv'
    edgelist.to_csv(fn)




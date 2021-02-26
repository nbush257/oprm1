import scipy
import pandas as pd
from scipy.signal import find_peaks,peak_widths
import scipy.signal
from scipy.signal import savgol_filter
from brian2 import *
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler,StandardScaler
try:
    import tensortools as tt
    has_tt=True
except:
    has_tt = False



def spike_state2trains(dat):
    '''

    :param dat: pickle file
    :return:
    '''
    ts = dat['spikemonitor']['t']
    idx = dat['spikemonitor']['i']
    trains = {}
    for ii in np.unique(idx):
        trains[ii] = ts[idx==ii]
    return(trains)


def chi_synch(raster):
    xbar_filt = np.mean(raster,axis=0)
    numer = np.mean(xbar_filt**2) - np.mean(xbar_filt)**2
    denom = np.mean(np.mean(raster**2,axis=1)-np.mean(raster,axis=1)**2)
    return(np.sqrt(numer/denom))


def create_train(ts,idx):
    '''
    Create a spike train dict like the Spikemonitor output
     - useful for after loading in the saved states
    :param ts:
    :param idx:
    :return:
    '''
    train = {}
    keys = np.unique(idx)
    for k in keys:
        train[k] = ts[idx==k]
    return(train)


def smooth_saved_rate(ratemon,width):
    '''
    Takes a dict from the "get_states" method of a Population rate monitor to create a smoothed rate
    :param ratemon: dict from get_states
    :return:
    '''
    dt = np.mean(np.diff(ratemon['t']))
    width_dt = int(np.round(2*width/dt))
    window = np.exp(-np.arange(-width_dt,
                               width_dt + 1)**2 *
                    1./(2*(width/dt)**2))
    return(Quantity(np.convolve(ratemon['rate'],
                                window*1./sum(window),
                                mode='same'),dim=hertz.dim))


def burst_freq(ts):
    '''
    Given a timeseries of spikes, return the Interburst frequency (from onset to onset
    :param ts:
    :return:
    '''
    ISI = np.diff(ts)
    pks = scipy.signal.find_peaks(ISI)
    IBI = np.diff(ts[pks[0]])

    return(np.nanmean(IBI),IBI)


def find_bursters_pk_ISI(train, N_cells, IBI_thresh=0.3):
    '''
    Find bursters using a different metric
    :param S:
    :param IBI_thresh:
    :return:cell_type,cell_type_str
    '''
    cell_type = np.ones(N_cells,dtype='int') * 2

    for cell, ts in train.items():
        if ts.shape[0] < 10:
            cell_type[cell] = 2
            continue
        ISI = np.diff(ts)
        pks = scipy.signal.find_peaks(ISI, threshold=IBI_thresh)[0]
        if len(pks) < 2:
            cell_type[cell] = 1
        else:
            cell_type[cell] = 0
    cell_map = {0:'burst',1:'tonic',2:'quiescent'}
    cell_type_str = [cell_map[x] for x in cell_type]

    return (np.array(cell_type),np.array(cell_type_str))


@check_units(ts=second)
def bin_trains(ts,idx,n_neurons,max_time,binsize=50*ms,start_time=5*second):
    '''
    bin_trains(ts,idx,n_neurons,binsize=50*ms,start_time=5*second):
    :param ts:
    :param idx:
    :param n_neurons:
    :param binsize:
    :param start_time:
    :return:
    '''
    cell_id = np.arange(n_neurons)
    bins = np.arange(start_time, max_time, binsize)
    raster = np.empty([n_neurons, len(bins)])
    # Remove spikes that happened before the start time
    idx = idx[ts>start_time]
    ts = ts[ts>start_time]

    # Loop through cells
    for cell in cell_id:
        cell_ts = ts[idx==cell]
        raster[cell, :-1]= np.histogram(cell_ts, bins)[0]
    return(raster,cell_id,bins)


def find_bursters(train):
    '''
    Take in a spike train dict and spit out a burst,tonic,quiescent cell class
    DEPRECATED
    '''
    pass
    N_cells = np.max(list(train.keys())) + 1

    m = np.empty(N_cells) * np.nan
    std = np.empty(N_cells) * np.nan
    quiescent = []
    for cell, ts in train.items():
        ts = ts[ts > 5 * second]
        if len(ts) < 10:
            quiescent.append(cell)
        m[cell] = np.nanmean(np.diff(ts))
        std[cell] = (np.nanstd(np.diff(ts)))
    cov = std / m

    quiescent = np.array(quiescent)
    cell_type = np.empty(N_cells) * np.nan
    cell_type[np.where(cov > 1.)[0]] = 0  # Bursters
    cell_type[np.where(cov < 1.)[0]] = 1  # Tonic
    cell_type[quiescent] = 2  # Quiescent
    return (cell_type)


def spikemonitor2cell_rates(S):

    for cell_id,times in S.spike_trains():
        pass
    pass


def get_shapes(pk_sigh,pk_eup,Rs,dt_r):
    '''
    Get all sigh burst shapes and  eupnea shapes from a smoothed
    population rate
    Uses a fixed window
    :param pk_sigh: samples of the sigh peaks
    :param pk_eup: samples of the eupnea peaks
    :param Rs: Smoothed spike rate
    :param dt_r: rate time step
    :return sigh_shape,eup_shape,t_sigh,t_eup:
    '''
    pre_win = int(0.5 * second / dt_r)
    post_win = int(2 * second / dt_r)
    sigh_shape = np.empty([pre_win + post_win, len(pk_sigh)])
    for ii, pk in enumerate(pk_sigh):
        if (pk - pre_win) < 0:
            continue
        try:
            sigh_shape[:, ii] = Rs[pk - pre_win:pk + post_win]
        except:
            pass
    t_sigh =  np.arange(-0.5*second,2*second,dt_r)

    # Store the eup shapes
    pre_win = int(0.5 * second / dt_r)
    post_win = int(1 * second / dt_r)
    eup_shape = np.empty([pre_win + post_win, len(pk_eup)]) * np.nan
    for ii, pk in enumerate(pk_eup[1:-1]):
        if pk - pre_win < 0:
            continue
        try:
            eup_shape[:, ii] = Rs[pk - pre_win:pk + post_win]
        except:
            pass

    # eup_shape = np.nanmedian(eup_shape, axis=1)
    # eup_shape = eup_shape[:,np.newaxis]

    t_eup =  np.arange(-0.5*second,1*second,dt_r)

    return(sigh_shape,eup_shape,t_sigh,t_eup)


def get_biphasic(sigh_shapes,dt):
    '''
    Given a matrix of sigh shapes, returns a boolean vector
    of size (n_sighs)
    :param sigh_shapes: sigh shape matrix (n_samps x n_sighs)
    :return is_biphasic
    '''
    n_sighs = sigh_shapes.shape[1]
    is_biphasic = np.zeros(n_sighs,dtype='bool')

    # Create smoothing window (Must be odd)
    win = int(400*ms/dt)
    if win%2==0:
        win+=1

    if n_sighs == 0:
        return(is_biphasic)
    else:
        smoothed = savgol_filter(sigh_shapes, win, 2, axis=0)
        d_smoothed = np.diff(smoothed,axis=0)
        for jj,(sigh,d_sigh) in enumerate(zip(smoothed.T,d_smoothed.T)):
            bigpk = np.argmax(sigh)
            pks = scipy.signal.find_peaks(d_sigh[:bigpk], prominence=0.1)[0]
            if len(pks) == 2:
                is_biphasic[jj] = True
    return is_biphasic


def t_to_phase(t,pk_eup):
    '''
    Given a time vector and a list of eupnic peaks,
    return a vector of same length as t that
    tells us the phase cycle
    :param t:
    :param pk_eup:
    :return:
    '''
    phi = np.zeros_like(t)
    # phi[pk_eup] = 1
    for ii,pk in enumerate(pk_eup[:-1]):
        next_pk = pk_eup[ii+1]
        phi[pk:next_pk] = np.linspace(0,1,next_pk-pk)

    return(phi)


def phase_reset(x,events,phase_transform=False,norm=True):
    '''
    Returns, in samples, the time to the next burst onset (normalized to burst period by default)
    and the time since the burst onset
    :param x:
    :param events:
    :param phase_transform:
    :return: t_to_onset,t_from_onset,pks

    '''
    if phase_transform:
        raise NotImplementedError('Phase transform via hilbert has not been coded')

    scl = RobustScaler()
    x_scl = scl.fit_transform(x[:,np.newaxis]).ravel()
    pks = scipy.signal.find_peaks(x_scl,height=1,prominence=1)[0]
    lips = scipy.signal.peak_widths(x_scl,pks,0.95)[2]
    lips = lips.astype('int')
    t_to_onset = np.empty(len(events))
    t_from_onset = np.empty(len(events))
    for ii,evt in enumerate(events):
        dum = lips-evt
        if not np.any(dum>0):
            t_to_onset[ii] = np.nan
        else:
            t_to_onset[ii] = np.min(dum[dum>0])
        if not np.any(dum<0):
            t_from_onset[ii] = np.nan
        else:
            t_from_onset[ii] = np.min(np.abs(dum[dum<0]))

    # Normalize to burst period
    if norm:
        burst_period = np.mean(np.diff(lips))
        t_to_onset = t_to_onset/burst_period
        t_from_onset = t_from_onset/burst_period

    return(t_to_onset,t_from_onset,lips)


def get_mean_burst_shape(x,events,pre,post):
    datmat = np.empty([pre+post,len(events)])*np.nan
    for ii,evt in enumerate(events):
        if (evt-pre)<0:
            continue
        if (evt+post)>x.shape[0]:
            continue

        datmat[:,ii] = x[(evt-pre):(evt+post)]
    return(datmat,np.nanmedian(datmat,1))

def get_burst_intervals(onsets,offsets):
    '''
    Convinience function so we don't have to think about the arithmetic of
    getting post-burst interval and frequency
    Adds a NaN for the last burst to keep consistent array sizes
    :param onsets:
    :param offsets:
    :return: PBI, IBI (time from offset to next onset, time between onsets)
    '''

    PBI = onsets[1:]-offsets[:-1]
    PBI = np.concatenate([PBI,[np.nan]])
    IBI = np.diff(onsets)
    IBI = np.concatenate([IBI,[np.nan]])
    return (PBI,IBI)


def pop_burst_stats(t,rate,scale=False,height=10,prominence=10,width=0.1,distance=0.5,rel_height=0.9):
    '''
    utility function that will return basic useful burst statsitics
    Expected to change/grow
    :return:
    '''
    if scale:
        scl = StandardScaler(with_mean=False)
        rate_scl =scl.fit_transform(rate[:,np.newaxis]).ravel()
    else:
        rate_scl = rate
    dt = t[1]
    pks,props = find_peaks(rate_scl,height=height,prominence=prominence,
                           width=width*second/dt,
                           distance=distance*second/dt)
    wd,ht,lips,rips = peak_widths(rate,pks,rel_height=rel_height)

    PBI,IBI = get_burst_intervals(t[lips.astype('int')],t[rips.astype('int')])

    pk_t = t[pks]
    wd_t = wd*dt

    df = pd.DataFrame()
    df['Peak Samples'] = pks
    df['Peak Times'] = pk_t
    df['Burst Width'] = wd_t
    df['Burst Amp'] = ht
    df['Onset Times'] = t[np.round(lips).astype('int')]
    df['Offset Times'] = t[np.round(rips).astype('int')]
    df['PBI'] = PBI
    df['IBI'] = IBI
    df['Frequency'] = 1/PBI

    return(df)


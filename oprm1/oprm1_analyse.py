import os
from tqdm import tqdm
import click
import glob
import sys
import seaborn as sns
from brian2 import *
import brian_utils.postproc as bup
import pandas as pd
import pickle
import scipy.signal
from spykes.plot import NeuroVis,PopVis
import glob

def main(fn,p_save=None,plot_tgl=False):

    # Extract run parameters from filename
    fparts = os.path.splitext(os.path.split(fn)[1])[0].split('_')
    prefix = '_'.join(fparts)
    syn_strength = 3.5
    pB = .0
    pT = .35
    max_hyp = float(fparts[2][3:])
    max_syn_shut = float(fparts[3][7:])
    runnum = int(fparts[4][-1])
    if p_save is None:
        p_save = os.path.split(fn)[0]

    # Load saved data
    with open(fn, 'rb') as fid:
        dat = pickle.load(fid)

    # Extract from saved data to vars
    N = int(dat['neurongroup']['N'])
    ratemon = dat['ratemonitor']
    pert_dt = dat['perturbation_dt']/second

    # Compute smoothed population rate
    rate = bup.smooth_saved_rate(ratemon,25*ms)

    pert_cond = pd.DataFrame({'Control':[20,40],
                 'Inh Block':[40,80],
                 'DAMGO': [100,140],
                 'Inh Block + DAMGO':[160,200],
                 'Isolated':[240,290]}).T
    pert_cond = pert_cond.rename({0:'start',1:'stop'},axis=1)


    # Get the burst statistics
    df = bup.pop_burst_stats(ratemon['t'],rate,prominence=7,height=10,width=[0.1,2.5],distance=0.1,rel_height=0.85)
    # Assign run parameters to the dataframe for future groupby needs
    pbi_temp = df['PBI'].values
    df['PBI_IRS'] = np.concatenate([[0],np.abs(pbi_temp[1:]-pbi_temp[:-1])/pbi_temp[:-1]])

    amp_temp = df['Burst Amp'].values
    df['amp_IRS'] = np.concatenate([[0],np.abs(amp_temp[1:]-amp_temp[:-1])/amp_temp[:-1]])

    df['syn'] = syn_strength
    df['pB'] = pB
    df['pT'] = pT
    df['pQ'] = 1-pB-pT
    df['max_hyp'] = max_hyp
    df['max_syn_shut'] = max_syn_shut
    df['Condition'] = 'Wash'
    df['run'] = runnum

    for k,v in pert_cond.iterrows():
        t_start = v['start']
        t_stop = v['stop']
        mask = np.where(np.logical_and(df['Onset Times']<t_stop,df['Onset Times']>t_start))[0]
        df.loc[mask,'Condition'] = k
        sub_df = df.loc[mask]
        pert_cond.loc[k,'n_bursts'] = sub_df.count()['Peak Times']
        pert_cond.loc[k,'mean_amp'] = sub_df.mean()['Burst Amp']
        pert_cond.loc[k,'std_amp'] = sub_df.std()['Burst Amp']

        pert_cond.loc[k,'mean_dur'] = sub_df.mean()['Burst Width']
        pert_cond.loc[k,'std_dur'] = sub_df.std()['Burst Width']

        pert_cond.loc[k,'mean_PBI'] = sub_df.mean()['PBI']
        pert_cond.loc[k,'std_PBI'] = sub_df.std()['PBI']

        # Drop the last one because it will have an outlier on the transition period
        pert_cond.loc[k,'PBI_IRS'] = sub_df.iloc[:-1,:].mean()['PBI_IRS']
        pert_cond.loc[k,'amp_IRS'] = sub_df.iloc[:-1,:].mean()['amp_IRS']

    pert_cond.eval('freq = n_bursts/(stop-start)',inplace=True)
    pert_cond['max_hyp'] = max_hyp
    pert_cond['max_syn_shut'] = max_syn_shut
    pert_cond['run'] = runnum
    pert_cond.eval('dur=stop-start',inplace=True)
    pert_cond['Normed Amp'] = pert_cond['mean_amp']/pert_cond['mean_amp']['Control']
    pert_cond['Normed Burst Dur'] = pert_cond['mean_dur']/pert_cond['mean_dur']['Control']
    pert_cond = pert_cond.fillna(0)

    # Plot sanity checks of rate and burst finding
    if plot_tgl:
        plt.style.use('seaborn-paper')
        f = plt.figure(figsize=(12,2))
        plt.plot(ratemon['t'],rate,'k',lw=0.5)
        for key,data in df.iterrows():
            plt.fill_between([data['Onset Times'],data['Offset Times']],np.min(rate),np.max(rate),color='r',alpha=0.2)
        sns.despine()
        plt.xlim(60,179)
        plt.xlabel('Time(s)')
        plt.ylabel('sp/s')
        plt.tight_layout()
        plt.savefig(os.path.join(p_save,f'{prefix}_population_detect.png'),dpi=300)
        plt.close('all')


    # Calculate normed amplitude By control
    if df.shape[0]>0:
        mean_amp = df.groupby('Condition').mean()['Burst Amp'][0]
        mean_freq = df.groupby('Condition').mean()['Frequency'][0]

        df['Normed Amp'] = df['Burst Amp']/mean_amp
        df['Normed Frequency'] = df['Frequency']/mean_freq
    else:
        # df['Opioid'] = opioids[:,0]

        df['Normed Frequency'] = [0]
        df['Normed Amp'] = [0]

    # ============== #
    # get the cell type #
    # ============== #
    ts = dat['spikemonitor']['t']
    spk_idx = dat['spikemonitor']['i']

    isolated = ts>pert_cond.loc['Isolated','start']*second
    ts = ts[isolated]
    spk_idx = spk_idx[isolated]

    if len(ts)==0:
        p_burst = 0.
        p_tonic= 0.
        p_quiescent =1.
    else:
        train = bup.create_train(ts,spk_idx)
        cell_int,cell_class = bup.find_bursters_pk_ISI(train,N,)
        p_burst = np.sum(cell_class=='burst')/float(len(cell_class))
        p_tonic = np.sum(cell_class=='tonic')/float(len(cell_class))
        p_quiescent = np.sum(cell_class=='quiescent')/float(len(cell_class))

    if df.shape[0] == 0:
        df['% Burst'] = [p_burst]
        df['% Tonic'] = [p_tonic]
        df['% Quiescent'] = [p_quiescent]

    df['% Burst'] = p_burst
    df['% Tonic'] = p_tonic
    df['% Quiescent'] = p_quiescent

    pert_cond['% Burst'] = p_burst
    pert_cond['% Tonic'] = p_tonic
    pert_cond['% Quiescent'] = p_quiescent

    df.to_csv(os.path.join(p_save,f'{prefix}_burst_data.csv'))
    pert_cond.to_csv(os.path.join(p_save,f'{prefix}_perturbation_data.csv'))


    # Do single cell analyses using tidy data
    spikes = pd.DataFrame()
    spikes['ts'] = dat['spikemonitor']['t']
    spikes['idx'] = dat['spikemonitor']['i']
    spikes['is_bursting'] = False
    spikes['Condition'] = 'Wash'
    # spikes = spikes[spikes.ts>60*second]
    # spikes = spikes[spikes.ts<179.9*second]

    ## Do burst/interburst analyses:
    ons = df['Onset Times'].values
    offs = df['Offset Times'].values
    IBI_fr = np.zeros([N,len(ons-1)])
    burst_fr = np.zeros([N,len(ons-1)])
    IDX = df.index.values

    # Label each spike by its perturbation condition
    for k,v in pert_cond.iterrows():
        t_start = v['start']
        t_stop = v['stop']
        mask = np.logical_and(spikes['ts']>t_start,spikes['ts']<t_stop)
        spikes.loc[mask,'Condition'] = k

    # Label each spike by bursting or not:
    for start,stop in zip(ons,offs):
        mask = np.logical_and(spikes['ts']>start,spikes['ts']<stop)
        spikes.loc[mask,'is_bursting'] = True


    # Get cell identities
    n_inh = dat['n_inh']
    n_oprm = dat['n_oprm1']

    spikes['Inh_type'] = 'Exc'
    spikes.loc[spikes.idx<n_inh,'Inh_type'] = 'Inh'

    spikes['OPRM'] = 'OPRM-'
    mask = spikes.query('idx>@n_inh & idx <(@n_inh+@n_oprm)').index
    spikes.loc[mask,'OPRM'] = 'OPRM+'

    # Set intrinsic identity
    iid = pd.DataFrame()
    iid['idx'] = np.arange(N)
    iid['Type'] = cell_class
    spikes = spikes.merge(iid,on='idx')

    # spikes.to_hdf(f'{prefix}_spikes_characterized.h5','spikes')

    # Try new latency calc:
    df_latency = pd.DataFrame()
    cell_params = spikes.groupby('idx').min()
    for jj in df.index[:-1]:
        ibi_start = df.loc[jj, 'Offset Times']
        ibi_stop = df.loc[jj + 1, 'Onset Times']
        ibi_dur = ibi_stop-ibi_start
        conds = df.loc[jj,'Condition']

        sub_spikes = spikes.query('ts>@ibi_start & ts<@ibi_stop')

        first_spike = sub_spikes[['idx','ts']].groupby('idx').min().reindex(index=range(N))
        n_spikes = sub_spikes[['idx','ts']].groupby('idx').count().reindex(index=range(N)).fillna(0)
        latency = first_spike-ibi_start
        perc_fr = n_spikes/ibi_dur

        dum = pd.DataFrame()
        dum['latency2fs'] = latency['ts']
        dum['ibi_dur'] = ibi_dur
        dum['Condition'] = conds
        dum['Inh_type'] = cell_params['Inh_type']
        dum['OPRM'] = cell_params['OPRM']
        dum['Type'] = cell_params['Type']
        dum['perc_fr'] = perc_fr
        dum = dum.eval('norm_dur = latency2fs/ibi_dur')
        dum['post_burst_idx'] = jj
        dum = dum.reset_index()
        df_latency = pd.concat([df_latency,dum])


    df_latency.to_csv(os.path.join(p_save,f'{prefix}_latency_data.csv'))



    ## Do burst/interburst analyses:
    ons = df['Onset Times'].values
    offs = df['Offset Times'].values
    IBI_fr = np.zeros([N,len(ons-1)])
    burst_fr = np.zeros([N,len(ons-1)])
    IDX = df.index.values

    # get interburst spiking
    for ii in range(len(ons)-1):
        start = offs[ii]
        stop = ons[ii+1]
        mask = np.logical_and(spikes['ts']>start,spikes['ts']<stop)
        sub_spikes = spikes[mask]
        fr = sub_spikes.groupby('idx').count()['ts']/(stop-start)
        idx = fr.index
        fr =fr.values
        IBI_fr[idx,ii] = fr

    # Get burst spiking
    for ii,(on,off) in enumerate(zip(ons,offs)):
        mask = np.logical_and(spikes['ts']>on,spikes['ts']<off)
        sub_spikes = spikes[mask]
        fr = sub_spikes.groupby('idx').count()['ts']/(off-on)
        idx = fr.index
        fr =fr.values
        burst_fr[idx,ii] = fr

    # Do dataframe munging
    df_IBI = pd.DataFrame(IBI_fr.T)
    df_IBI.index = IDX
    df_IBI['Condition'] = df['Condition']
    df_IBI = df_IBI.groupby('Condition').mean().T

    df_burst = pd.DataFrame(burst_fr.T)
    df_burst.index = IDX
    df_burst['Condition'] = df['Condition']
    df_burst = df_burst.groupby('Condition').mean().T
    df_burst = df_burst.melt(value_name = 'Burst Firing Rate')

    df_IBI.set_index(cell_class,append=True,inplace=True)
    n_inh = dat['n_inh']
    n_oprm = dat['n_oprm1']
    is_oprm = np.repeat('OPRM-',N)
    is_oprm[n_inh:(n_inh+n_oprm)] = 'OPRM+'

    is_inh = np.repeat('Exc',N)
    is_inh[:n_inh] = 'Inh'

    df_IBI.set_index(is_oprm,append=True,inplace = True)
    df_IBI.set_index(is_inh,append=True,inplace=True)
    df_IBI.index.names = ['Cell ID','Type','OPRM1+','Inh']
    df_IBI = df_IBI.stack()
    df_IBI.name='IBI Firing Rate'
    df_IBI = df_IBI.reset_index()
    df_IBI['Burst Firing Rate'] = df_burst['Burst Firing Rate']
    df_IBI['FR ratio'] = df_IBI['Burst Firing Rate']/df_IBI['IBI Firing Rate']

    # add the run data
    df_IBI['syn'] = syn_strength
    df_IBI['pB'] = pB
    df_IBI['pT'] = pT
    df_IBI['pQ'] = 1-pB-pT
    df_IBI['Max Syn Shut'] = max_syn_shut
    df_IBI['Max Hyp'] = max_hyp
    df_IBI = df_IBI[df_IBI['Condition']!='Wash']
    df_IBI['run'] = runnum

    # Save firing rate data
    df_IBI.to_csv(os.path.join(p_save,f'{prefix}_spikerate_data.csv'))

    #
    # df_IBI = df_IBI.replace(np.inf,np.nan)
    # df_IBI = df_IBI.dropna()
    #
    # plt.style.use('seaborn-paper')
    # plt.close('all')
    # f, ax = plt.subplots(nrows=3, ncols=3, sharex=True, figsize=(7, 7))
    # for ii,key in enumerate(['IBI Firing Rate', 'Burst Firing Rate', 'FR ratio']):
    #     plt.sca(ax[0,ii])
    #     sns.pointplot(x='Condition',y=key,data = df_IBI,hue='Type',palette='Dark2',linewidth=1)
    #     if ii==2:
    #         plt.legend(bbox_to_anchor=(0.9,0.5),fontsize=8)
    #     else:
    #         plt.gca().get_legend().remove()
    #
    #     plt.sca(ax[1,ii])
    #     sns.pointplot(x='Condition',y=key,data = df_IBI,hue='OPRM1+',palette='PiYG',linewidth=1)
    #     if ii==2:
    #         plt.legend(bbox_to_anchor=(0.9,0.5),fontsize=8)
    #     else:
    #         plt.gca().get_legend().remove()
    #
    #     plt.sca(ax[2,ii])
    #     sns.pointplot(x='Condition',y=key,data = df_IBI,hue='Inh',palette='RdBu',linewidth=1)
    #     if ii==2:
    #         plt.legend(bbox_to_anchor=(0.9,0.5),fontsize=8)
    #     else:
    #         plt.gca().get_legend().remove()
    #     plt.gca().set_xticklabels(plt.gca().get_xticklabels(),rotation=50,ha='right')
    #
    #     ax[0,ii].set_title(key)
    #     sns.despine()
    #     plt.tight_layout()
    #
    # plt.savefig(os.path.join(p_save,f'{prefix}_firing_rates.png'),dpi=300)
    # plt.savefig(os.path.join(p_save,f'{prefix}_firing_rates.pdf'))
    # plt.close('all')
    #
    # Do a little summary
    # means = df.groupby('Condition').mean().drop(['Peak Samples', 'Peak Times', 'Onset Times', 'Offset Times'],
    #                                             axis=1)
    # means['n_burst'] = df.groupby('Condition').count()['Peak Times']
    #
    # cvs = df.groupby('Condition').std()/df.groupby('Condition').mean()
    # cvs = cvs[['Burst Width','Burst Amp','PBI','IBI','Frequency']]
    # means['cv_freq'] = cvs['Frequency']
    # means['cv_amp'] = cvs['Burst Amp']
    # means['cv_pbi'] = cvs['PBI']
    # means['cv_ibi'] = cvs['IBI']
    # means['cv_width'] = cvs['Burst Width']
    #
    # means.to_csv(f'{prefix}_mean_vals.csv')

    # ## ========================= ##
    # # Make psths for each cell #  -- Move to the plotting code
    # ## ========================= ##
    # neuron_list = []
    # for n_id in range(300):
    #     ts = spikes[spikes['idx']==n_id]['ts'].values
    #     if len(ts)<2:
    #         continue
    #     neuron_list.append(NeuroVis(ts,name=n_id))
    #
    # pop = PopVis(neuron_list)
    # psth = pop.get_all_psth(event='Onset Times',df = df.query('Condition!="Wash"'),conditions='Condition',plot=False,window=[-2500,500],binsize=50)
    # pop.plot_population_psth(psth)
    # nn = neuron_list[200]
    # tonic_fr = len(nn.spiketimes>240)/50
    # nn.get_psth(event='Onset Times',df=df.query('Condition!="Wash"'),conditions='Condition',window=[-3500,500],binsize=250)
    # plt.axhline(tonic_fr)




    return(df,df_IBI,pert_cond,df_latency)

@click.command()
@click.argument('f_spec')
@click.argument('version')
def batch(f_spec,version):
    flist = glob.glob(f_spec)

    DF = pd.DataFrame() # Analyses on each burst
    DF_IBI = pd.DataFrame() # Analyses on burst/IBI firing rates
    DF_PERT = pd.DataFrame() # Analyses on perturbation conditions.
    DF_LATENCY = pd.DataFrame() # Analyses on latency to first spikes

    uid = 0
    for f in tqdm(flist):
        # print(f'Working on {f}')

        df,df_IBI,perts,latencies = main(f)
        df['uid'] = uid
        df_IBI['uid'] = uid
        perts['uid'] = uid
        latencies['uid'] = uid
        DF = pd.concat([DF,df])
        DF_IBI = pd.concat([DF_IBI,df_IBI])
        DF_PERT = pd.concat([DF_PERT, perts])
        DF_LATENCY = pd.concat([DF_LATENCY, latencies])
        uid+=1
    DF.to_csv(f'summary_burst_v{version}.csv')
    DF_IBI.to_csv(f'summary_firingrates_v{version}.csv')
    DF_PERT.to_csv(f'perturbation_summary_v{version}.csv')
    DF_LATENCY.to_csv(f'latency_summary_v{version}.csv')




if __name__=='__main__':
    batch()

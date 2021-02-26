import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import networkx as nx
import seaborn as sns
import numpy as np

def shade_stim(phi_t,phi,color='c'):
    ax = plt.gca()
    phi_bin = phi>.1
    phi_bin = phi_bin.astype('int')
    phi_on = np.where(np.diff(phi_bin)==1)[0]
    phi_off = np.where(np.diff(phi_bin)==-1)[0]-1
    for pon,poff in zip(phi_on,phi_off):
        ax.axvspan(phi_t[pon],phi_t[poff],color=color,alpha=0.3)




def make_pop_movie(neurons,raster,src,dest):
    '''
    Make a series of images that looks like a calcium imaging movie
    :param neurons:
    :param raster:
    :return:
    '''

    G = nx.Graph()
    G.add_nodes_from(np.arange(raster.shape[0]))
    G.add_edges_from(np.vstack([src,dest]).T)
    xy = nx.spring_layout(G)
    xy = pd.DataFrame(xy).values.T

    cgx = np.zeros(raster.shape[1])
    cgy = np.zeros(raster.shape[1])
    for ii,activity in enumerate(raster.T):
        cgy[ii] = np.sum(xy[:, 1] * activity) / np.sum(activity)
        cgx[ii] = np.sum(xy[:, 0] * activity) / np.sum(activity)
    cgx[np.isnan(cgx)] = 0
    cgy[np.isnan(cgy)] = 0
    cgys = scipy.signal.savgol_filter(cgy,5,1)
    cgxs = scipy.signal.savgol_filter(cgx,5,1)

    plt.style.use('dark_background')
    f = plt.figure()
    for ii,activity in enumerate(raster.T[:100]):
        plt.cla()
        plt.axis('off')
        plt.scatter(xy[:,0],xy[:,1],c=activity,vmin=0,vmax=np.max(raster),s=15)
        plt.plot(cgxs[ii],cgys[ii],'o',markersize=10,color='r')
        plt.pause(0.05)

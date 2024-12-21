from scipy import signal as ss
from scipy.stats import zscore
import neo
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from itertools import groupby
from sklearn.metrics import accuracy_score as acc, balanced_accuracy_score as bacc, confusion_matrix as cm, cohen_kappa_score as kappa, davies_bouldin_score as dbs, calinski_harabasz_score as chs, silhouette_score as silhs, f1_score
from sklearn.mixture import GaussianMixture as GM
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope as EE
from scipy.stats import kurtosis, skew, zscore
import sys
import warnings 
from math import ceil
import os
from ipywidgets import IntText, interact, IntSlider, Layout
from sklearn.neighbors import KernelDensity as KD
from sklearn.linear_model import LinearRegression as LR
import matplotlib.font_manager as font_manager
from matplotlib.colors import LinearSegmentedColormap as LSC
from scipy.spatial import ConvexHull, Delaunay
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering as SC
import pickle
from time import perf_counter
from datetime import datetime
import logging
from scipy.signal import windows
from tqdm import tqdm
# from multitaper_spectrogram_python import multitaper_spectrogram
from sklearn.svm import OneClassSVM as ocSVM
from scipy.spatial import Delaunay, ConvexHull
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import hvplot.pandas
# Import plotting libs
import holoviews as hv
from holoviews.plotting.links import DataLink
from sklearn.naive_bayes import GaussianNB
from collections import OrderedDict as OD
import base64
from io import BytesIO
hv.notebook_extension('bokeh')


font = font_manager.FontProperties(family='Cambria', size=18)
plt.rcParams['font.family'] = 'Cambria'
warnings.simplefilter('ignore')
mne.set_log_level('ERROR')
import h5py

def dc_remove(data, window_sec=2, fs=0.1):
    i, res, baseline = 0, [], []
    window = round(window_sec * fs)
#     print(window)
    while i < len(data):
        baseline.append(np.min(data[max(0, i-window): min(len(data)-1, i+window)]))
        res.append(data[i] - baseline[-1])
        i += 1
    return np.array(res).flatten()

def no_singles(ser, val, n=1):
    n = int(n)
    cur = 0
    res = []
    for c, length in [(c, len(list(lst))) for c, lst in groupby(ser)]:
        if (c == val) and (length <= n):
            if cur > 0:
                res.extend([ser[cur - 1]]*length)
            else: 
                res.extend([c] * length)
        else:
            res.extend([c] * length)
        cur += length
    return res

def scale_to_range(x, mi, ma):
    OldMin, OldMax = np.min(x), np.max(x)
    x = (((x - OldMin) * (ma - mi)) / (OldMax - OldMin)) + mi
    return x

def stage_duration_report(fname, res_, return_=False, duration_ranges=((0, 30), (30, 120), (120, 300), (300, 600), (600, 5000)), stage_map={0: 'W', 1: 'NR', 2: 'R', 3:'A', 8:'_', 4: 'x_spindle'}):
    dfs = []
    for num, res in enumerate(res_):
        stats = {stage: [0]*len(duration_ranges) for stage in np.unique(list(res.keys()))}
        for stage in np.unique(list(res.keys())):
            for dur in res[stage]:
                for i in range(len(duration_ranges)):
                    if (dur >= duration_ranges[i][0]) and (dur < duration_ranges[i][1]): break
                stats[stage][i] += 1
        res_df = pd.DataFrame(columns=('Stages #{}'.format(num), 'Duration range (sec) #{}'.format(num), 'Count #{}'.format(num)), index=range(len(stats.keys())*len(duration_ranges)))
        cur_row = 0
        for stage in stats.keys():
            for i, score in enumerate(stats[stage]):
                res_df.iloc[cur_row] = (stage_map[int(stage[0])]+' '+stage[1:], '{}<...<={}'.format(*duration_ranges[i]), score)
                cur_row += 1
        dfs.append(res_df)
    if return_:
        return pd.concat(dfs, axis=1)
    else:
        pd.concat(dfs, axis=1).to_csv(fname, index=False)

def fig_style(xlabel, ylabel, fig):
    plt.xlabel(xlabel, fontsize=18, family='Cambria')
    plt.ylabel(ylabel, fontsize=18, family='Cambria')
    plt.tick_params(axis='both', labelsize=13)
    plt.legend(prop=font)
    plt.xticks(fontname='Cambria')
    plt.yticks(fontname='Cambria')
        
cut_percent = lambda x, p: x[(x > np.percentile(x, p)) & (x < np.percentile(x, 100-p))]
def scatter_hist(rt, thr, thr2, mask = None, fit_lr=False, transparent=False): #     def scatter_hist(rt, thr, thr2, aut, exp):
    plt.rcParams['figure.figsize'] = (12, 6)
    if transparent: cmap1 = LSC.from_list('', ['white', 'white', 'yellow'])
    else: cmap1 = LSC.from_list('', ['purple', 'cyan', 'yellow'])
    fig = plt.figure(figsize=(12, 7))
    # plt.xlabel('Theta band PSD', fontsize=18, family='Cambria')
    # plt.ylabel('Delta band PSD', fontsize=18, family='Cambria')
    
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(3, 1), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax.tick_params(axis='both', labelsize=13)
    
    ax_histx = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_histx.tick_params(axis='both', labelsize=13)
    
    ax_histy = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_histy.tick_params(axis='both', labelsize=13)
    
    # ax_histx.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    x, y = rt.T
    # print(thr)
    # x, y = cut_percent(x, .5), cut_percent(y, .5)

    if mask is None:
        ax.scatter(x, y, c=y>thr, alpha=.7, cmap=cmap1)
    else: 
        scatter = ax.scatter(x, y, c=mask, cmap=cmap1, alpha=.7) #[(rt[:, 0] > np.percentile(rt[:, 0], .5)) & (rt[:, 0] < np.percentile(rt[:, 0], 100-.5))]
        legend1 = ax.legend(scatter.legend_elements()[0], ('NREM', 'Wake', 'REM'), prop=font)
        ax.add_artist(legend1)
        
    if fit_lr:
        x_, y_ = rt.T
        # print(thr)
        x_, y_ = cut_percent(x_, 5), cut_percent(y_, 5)
        lr = LR(fit_intercept=False).fit(x_[y_ > thr].reshape(-1, 1), y_[y_ > thr].reshape(-1, 1))
        xl = np.linspace(np.min(x_), np.max(x_), 10)
        k = lr.coef_.flatten()[0]#, b, lr.intercept_.flatten()[0]
        yl = k*xl #+ b#k * (xl - thr2) + thr
        ax.plot(xl, yl, c='grey')
        ax.set_title(f'{abs(90-np.arctan(k)*180/np.pi):.2f}, {np.corrcoef(x_[y_ > thr], y_[y_ > thr])[0, 1]:.3f}')   
    # newhor = np.tan(abs(90-np.arctan(k)*180/np.pi))*(thr2-xl)+thr
    # ax.plot(xl, newhor, ls='--', c='r')
    # lrhr = LR().fit(xl.reshape((-1, 1)), newhor.reshape((-1, 1)))
    # kref, bref = lrhr.coef_.flatten()[0], lrhr.intercept_.flatten()[0]
    # angle = np.arctan(k)*180/np.pi
    # newvert = np.tan(angle+90)*(xl-thr2)+thr
    # ax.plot(xl, newvert, ls='--', c='r')
    
    
    
    # theta = (90-np.arctan(k)*180/np.pi)*np.pi/180
    # rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # data_rot = (rot_matrix @ np.stack((x, y), axis=1).T).T
    # print(np.min(data_rot, axis=0), np.max(data_rot, axis=0))
#     color_mask = np.zeros(len(rt))
#     color_mask[(exp != 0) & (aut == 1)] = 1 #tp
#     color_mask[(exp == 0) & (aut == 1)] = 2 #fp
#     color_mask[(exp != 0) & (aut != 1)] = 3 #fn
#     colors = ('k', 'g', 'r', 'b')
#     labels = ('tn', 'tp', 'fp', 'fn')
#     counter = 0
#     for c in np.unique(color_mask).astype(int):
#         counter += np.where(color_mask==c)[0].size
#         ax.scatter(rt[color_mask==c, 0], rt[color_mask==c, 1], c=colors[c], label=labels[c], alpha=.6)
#     ax.legend()
    ax.set_xlim(np.percentile(rt[:, 0], .5), np.percentile(rt[:, 0], 99.5))
    ylims = np.percentile(rt[:, 1], .5), np.percentile(rt[:, 1], 95)
    ax.set_ylim(*ylims)
    
    n, b, p = ax_histy.hist(y, 60, orientation='horizontal', color='white')
    ax_histy.hist(y[y > thr], b, orientation='horizontal', color='g', alpha=.7)
    ax_histy.hist(y[y <= thr], b, orientation='horizontal', color='r', alpha=.7)
    x = x[y < thr]
    n, b, p = ax_histx.hist(x, 40, color='white')
    ax_histx.hist(x[x > thr2], b, color='g', alpha=.7)
    ax_histx.hist(x[x <= thr2], b, color='r', alpha=.7)
    ax.axhline(thr, c='k')
    ax.axvline(thr2, c='k', ymax=(thr-ylims[0])/(ylims[1]-ylims[0]))
    plt.show()
    
    # rotated
#     plt.rcParams['figure.figsize'] = (12, 6)
#     fig = plt.figure(figsize=(12, 7))
#     gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(3, 1), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
#     ax = fig.add_subplot(gs[0, 0])
#     ax_histx = fig.add_subplot(gs[1, 0], sharex=ax)
#     ax_histy = fig.add_subplot(gs[0, 1], sharey=ax)
#     # ax_histx.tick_params(axis="x", labelbottom=False)
#     ax.tick_params(axis="x", labelbottom=False)
#     ax_histy.tick_params(axis="y", labelleft=False)
#     x, y = data_rot.T
#     if mask is None:
#         ax.scatter(x, y, c=y>thr, alpha=.7)
#     else: 
#         # mask = mask[:len(data_rot)]
#         ax.scatter(x, y, c=mask[(rt[:, 0] > np.percentile(rt[:, 0], .5)) & (rt[:, 0] < np.percentile(rt[:, 0], 100-.5))], alpha=.7)
#         ax.set_xlim(np.percentile(data_rot[:, 0], 0), np.percentile(data_rot[:, 0], 100))
#     ylims = np.percentile(data_rot[:, 1], 0), np.percentile(data_rot[:, 1], 100)
#     ax.set_ylim(*ylims)
    
#     n, b, p = ax_histy.hist(y, 60, orientation='horizontal', color='white')
#     ax_histy.hist(y[y > thr], b, orientation='horizontal', color='yellow', alpha=.7)
#     ax_histy.hist(y[y <= thr], b, orientation='horizontal', color='purple', alpha=.7)
#     x = x[y < thr]
#     n, b, p = ax_histx.hist(x, 40, color='white')
#     ax_histx.hist(x[x > thr2], b, color='g', alpha=.7)
#     ax_histx.hist(x[x <= thr2], b, color='r', alpha=.7)
#     ax.axhline(thr, c='k')
#     ax.axvline(thr2, c='k', ymax=(thr-ylims[0])/(ylims[1]-ylims[0]))
#     plt.show()

def rotate_clusters(rt, thr):
    x, y = rt.T
    x, y = cut_percent(x, 5), cut_percent(y, 5) # ненадежно
    lr = LR(fit_intercept=False).fit(x[y > thr].reshape(-1, 1), y[y > thr].reshape(-1, 1))    
    k = lr.coef_.flatten()[0]
    theta = (90-np.arctan(k)*180/np.pi)*np.pi/180
    print(f'Rotated {theta*180/np.pi:.2f} degrees counterclockwise')
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    data_rot = (rot_matrix @ rt.T).T
    return data_rot
    
def ee_thr(rt, f = 2, lowest_cont=20):
    for cont_ in np.arange(lowest_cont, 30.1, .5):
        cont = cont_/100
        clf = EE(contamination=cont)
        y = clf.fit_predict(rt.reshape((-1, f))) 
        kurt = kurtosis(rt[y == 1], axis=0)
        if np.all(kurt < 0):
            break
    if f ==2: return y, np.min(rt[y==1][:, f - 1])
    else: return np.max(rt[y==1]) + (np.percentile(rt,99)-np.percentile(rt, 1))/12

def combine_masks(rem_mask, mask1, pps2, art_mask):
    rem_mask = 2*np.array(rem_mask)    
    res_mask_ = mask1 + rem_mask[art_mask]
#     res_mask = np.zeros(len(art_mask))
#     res_mask[art_mask] = res_mask_
    res_mask_[res_mask_ > 2] = 2
    dummy = np.zeros(min(pps2*1, len(art_mask)))
    
    ####
    # print(pps2, dummy.shape, art_mask.shape, res_mask_.shape)
    
    dummy[art_mask[:len(dummy)]] = res_mask_
    j = 0
    not_mask = np.argwhere(np.logical_not(art_mask)).flatten()
    while j < len(not_mask):
        idx = not_mask[j]
        k = idx
        while (k-idx) < 1: # заменяем каждую артефактную эпоху и каждую эпоху (в т.ч. и не обязательно артефактную) на состояние, которое было за 2 эпохи до артефакта
            dummy[k] = dummy[idx-2]
            k += 1
            if k >= len(not_mask):
                break
#         j = k # это некорректно, так как k нумеруется по всем 10-секундным эпохам, а j - по артефактным индексам
        j += 1
    return dummy

def cluster_art(hypno, art_mask):
    dummy = np.array(hypno)
    j = 0
    not_mask = np.argwhere(np.logical_not(art_mask)).flatten()
    while j < len(not_mask):
        idx = not_mask[j]
        k = idx
        while (k-idx) < 1: # заменяем каждую артефактную эпоху и каждую эпоху (в т.ч. и не обязательно артефактную) на состояние, которое было за 2 эпохи до артефакта
            dummy[k] = dummy[idx-2]
            k += 1
            if k >= len(not_mask):
                break
#         j = k # это некорректно, так как k нумеруется по всем 10-секундным эпохам, а j - по артефактным индексам
        j += 1
    return dummy

def res_scatter(delta_rem, rms2, exp, aut, thr, thr2):
    plt.rcParams['figure.figsize'] = (12, 6)
    color_mask = np.zeros(len(rms2))
    color_mask[(exp != 0) & (aut == 2)] = 1 #tp
    color_mask[(exp == 0) & (aut == 2)] = 2 #fp
    color_mask[(exp != 0) & (aut != 2)] = 3 #fn
    colors = ('k', 'g', 'r', 'b')
    labels = ('tn', 'tp', 'fp', 'fn')
    for c in np.unique(color_mask).astype(int):
        plt.scatter(rms2[color_mask==c], delta_rem[color_mask==c], c=colors[c], label=labels[c], alpha=.6)
    plt.legend()
    plt.axhline(thr)
    plt.xlim(np.percentile(rms2, .5), np.percentile(rms2, 99.5))
    ylims = np.percentile(delta_rem, .5), np.percentile(delta_rem, 99.5)
    plt.ylim(*ylims)
    plt.axvline(thr2, ymax=(thr-ylims[0])/(ylims[1]-ylims[0]))
    plt.show()
    
def red_masks(scoring, mask, n_back):
        # сделаем красненькие
    if scoring.delta_refs_copy is not None: scoring.delta_refs = scoring.delta_refs_copy
    scoring.pick_delta(scoring.ratios[0], scoring.delta_refs, plot=False, normalize=True, cycle_hours=1)
    masks = []
    scoring.priority_theta_nan = False
    for delta in scoring.delta_refs:
        scoring.delta_ref = delta
        scoring.theta_nan = []
        scoring.rem_scoring(n_hours_cycle=scoring.actual_hours, n_back=n_back)
        masks.append([el.astype(int) for el in scoring.theta_nan])
    cycles_rem = []
    for c in range(len(masks[0])):
        chunk = np.sum(np.vstack([el[c].reshape((1, -1)) for el in masks]), axis=0)
        chunk[chunk != 0] = 1
        cycles_rem.append(chunk)
    # вручную отсеим REM красненькими
    res = np.zeros(len(mask))
    res[scoring.art_mask] = ((mask[scoring.art_mask] == 1) & (cycles_rem[0] == 1)).astype(int)
    mask = res
    red = np.zeros(len(mask))
    red[scoring.art_mask] = cycles_rem[0]
    red[red != 1] = None
    return mask, red

def thr_to_left(seq, thr_stat=0):
    step = (np.percentile(seq, 99)-np.percentile(seq, 1))/100
    thr = np.percentile(seq, 99) - step
    # stat = kurtosis(seq[seq < thr])
    stat = skew(seq[seq < thr])
    while stat > thr_stat:
        thr -= step
        # stat = kurtosis(seq[seq < thr])
        stat = skew(seq[seq < thr])
    return thr

def thr_lowest(seq):
    n, b, p = plt.hist(seq, 40)
    plt.close()
    b = b[:-1]
    res = []
    mod = np.argmax(n)+1
    for i in range(1, len(n)-1):
        res.append((int((n[i]<n[i-1]) and (n[i]<n[i+1]) and (i > mod)), np.mean((n[i-1], n[i+1])))) #, 1/(i - mod)
    res = np.array(res).reshape((-1, 2)) # 3
    res = res[:, 0] * res[:, 1] #* res[:, 2]
    idx = np.argmax(res)+1
    return b[idx]

def rem_vote(to_vote, rule=2):
    arr = np.vstack([mask.reshape((1, -1)) for mask in to_vote])
    arrs = np.sum(arr, axis=0)
    return (arrs >= rule).astype(int)        

DATA_EXTENSION = 'smr'
def read_hypnos(fnames, dir_):
    expert_rem_hypnos = {}
    for fname in fnames:
        expert_rem_hypnos[fname[:-3]+DATA_EXTENSION] = {}
        df = pd.read_csv(os.path.join(dir_, fname), sep='\t')
        print(os.path.join(dir_, fname), np.unique(df.values))
        data = df.values
        # a_d = np.array([el.split('_') for el in df.columns]).reshape((-1, 2))
        for el in df.columns:
            a, d = list(map(int, el.split('_')))
            _ = expert_rem_hypnos[fname[:-3]+DATA_EXTENSION].setdefault(a, {})
            expert_rem_hypnos[fname[:-3]+DATA_EXTENSION][a][d] = df[el].values.astype(int)
            expert_rem_hypnos[fname[:-3]+DATA_EXTENSION][a][d][expert_rem_hypnos[fname[:-3]+DATA_EXTENSION][a][d] > 2] = 0
        # a, d = len(np.unique(a_d[:, 0])), len(np.unique(a_d[:, 1]))
        # expert_rem_hypnos[fname[:-3]+'smr'] = df.values
        # expert_rem_hypnos[fname[:-3]+'smr'] = df.values
    return expert_rem_hypnos

CLUSTER_NAMES = []
def prescoring(fnames, dir_, **kwargs):
    global CLUSTER_NAMES
    red_to_interactive = None
    vote_dict, scrngs = {}, {}
    for fn, fname in enumerate(fnames):
        print('File \"'+fname+'\"')
        vote_dict[fname] = {}
        scrngs[fname] = {}
        window_sec1, window_sec2, n_hours_cycle, delta_cluster_bands, theta, delta_mode, theta_mode, filt_params, pre_filt, depr, spindles, w_nr_cluster_strictness, red_mask, n_back, verbose, expert_hypnos, parts, w_humps, ep = kwargs['window_sec1'], kwargs['window_sec2'], kwargs['n_hours_cycle'], kwargs['delta_cluster_bands'], kwargs['theta'], kwargs['delta_mode'], kwargs['theta_mode'], kwargs['filt_params'], kwargs['pre_filt'], kwargs['depr'], kwargs['spindles'], kwargs['cluster_strictness'], kwargs['red_mask'], kwargs['n_back'], kwargs['verbose'], kwargs['expert_hypnos'], kwargs['parts'], kwargs['w_humps'], kwargs['ep']
        
        scoring = Scorer(filename=os.path.join(dir_, fname), window_sec1=window_sec1, window_sec2=window_sec2, n_hours_cycle=n_hours_cycle, delta=delta_cluster_bands, theta=theta, delta_mode=delta_mode, theta_mode=theta_mode, filt_params=filt_params, pre_filt=pre_filt, depr=depr, spindles=spindles, cluster_strictness=w_nr_cluster_strictness)
        for eeg_idx in range(len(scoring.eeg_map)): #(1, 2) или вместо "" просто список/кортеж нужных индексов животных (считая с 0), например (0, 2, 3, 5)
            vote_dict[fname][eeg_idx] = []
            scrngs[fname][eeg_idx] = []
            print('Animal '+str(eeg_idx))
            ids = scoring.eeg_map[eeg_idx]
            n_days = round(float(scoring.data.analogsignals[ids[0]].duration/3600/24))
            if n_days == 0:
                n_days = ceil(float(scoring.data.analogsignals[ids[0]].duration/3600/24))

            for day in range(n_days):
                print('Day '+str(day))
                cache_name = os.path.splitext(os.path.join(dir_, fname))[0]+f'{eeg_idx}_{day}.pickle'
                cache_flag = os.path.exists(cache_name)
                # if cache_flag: print('Cached prescoring found')
                scoring = Scorer(filename=os.path.join(dir_, fname), window_sec1=window_sec1, window_sec2=window_sec2, n_hours_cycle=n_hours_cycle, delta=delta_cluster_bands, theta=theta, delta_mode=delta_mode, theta_mode=theta_mode, filt_params=filt_params, pre_filt=pre_filt, depr=depr, spindles=spindles, verbose=verbose)
                scrngs[fname][eeg_idx].append(scoring)
                if not cache_flag:
                    scoring.pick_eeg(eeg_idx=eeg_idx, day_n=day, skip_minutes=kwargs['skip_minutes'][eeg_idx])
                    scoring.delta_ref_bands = kwargs['delta_ref_bands_red']
                    scoring.prepare_rms(delta_mode='diff', smooth_fft=kwargs['smooth_fft'], theta_mode='diff', theta_res='ratio', add_mode='diff', w=kwargs['w']) 
                    full_spec, freqs = scoring.fft_feats(([]), use_welch=False, log=False, manual=True, use_MT=False)                   
                    # 
                    if kwargs['cluster_artifact']: # тут точно не должно быть сдвинуто левее?
                        vote_dict[fname][eeg_idx].append([[], 0, None, None, red_to_interactive, None, scoring.art_rms, (full_spec, freqs), []]) # 2-scoring.mask1, 3-scoring.art_mask, 5-scoring.nr_mask
                    else:
                        vote_dict[fname][eeg_idx].append([[], 0, None, None, red_to_interactive, []]) #scoring.mask1, 3-scoring.art_mask, 5-scoring.nr_mask
                        
                    if kwargs['save_pickle']: 
                        with open(cache_name, 'wb') as f: 
                            pickle.dump([to_vote, 0, None, None, red_to_interactive, scoring.nr_mask, scoring.art_rms, (full_spec, freqs), []], f)
                else:
                    with open(cache_name, 'rb') as f: cache = pickle.load(f)
                    vote_dict[fname][eeg_idx].append(cache) #[:-3]+[tuple(cache[-3:-1])]+[cache[-1]]# тогда в return может быть недостаточно данных в scoring
    return vote_dict, scoring, scrngs

def prescoring_artifacts(vote_dict, scorings, **kwargs):
    window_sec1, window_sec2, n_hours_cycle, delta_cluster_bands, theta, delta_mode, theta_mode, filt_params, pre_filt, depr, spindles, w_nr_cluster_strictness, red_mask, n_back, verbose, expert_hypnos, parts, w_humps, ep, w_humps_rises, rise_len_thr, refined_ridges, fad = kwargs['window_sec1'], kwargs['window_sec2'], kwargs['n_hours_cycle'], kwargs['delta_cluster_bands'], kwargs['theta'], kwargs['delta_mode'], kwargs['theta_mode'], kwargs['filt_params'], kwargs['pre_filt'], kwargs['depr'], kwargs['spindles'], kwargs['cluster_strictness'], kwargs['red_mask'], kwargs['n_back'], kwargs['verbose'], kwargs['expert_hypnos'], kwargs['parts'], kwargs['w_humps'], kwargs['ep'], kwargs['w_humps_rises'], kwargs['rise_len_thr'], kwargs['refined_ridges'], kwargs['fad']
    
    fnames = list(vote_dict.keys()) if fad is None else [fad[0]]     
    for fn, fname in enumerate(fnames):
        print('File \"'+fname+'\"')
        eeg_ids = list(vote_dict[fname].keys()) if fad is None else [fad[1]]
        for eeg_idx in eeg_ids:
            print('Animal '+str(eeg_idx))                                       #перестанет работать, если много файлов !
            n_days = range(len(vote_dict[fname][eeg_idx])) if fad is None else [fad[2]]
            for day in n_days:
                scoring = scorings[fname][eeg_idx][day]

                copy_data = np.zeros(len(scoring.art_rms))

                
                scoring.force_art_thr = kwargs['force_art_thr'][eeg_idx]
                # определяем пороговые артефакты
                scoring.verbose = verbose
                scoring.thr_artifacts()
                if (refined_ridges is None): scoring.first_art_mask = np.array(scoring.art_mask)
                # вырезаем горбы
                
                if not (expert_hypnos is None): scoring.txthypno = expert_hypnos[fname][eeg_idx][day][:min(len(expert_hypnos[fname][eeg_idx][day]), len(scoring.raw_epochs))]
                scoring.set_thr2_auto(copy_data, plot=False, zbefore=kwargs['zbefore'], oafter=kwargs['oafter'], ridge_thrs=kwargs['ridge_thrs'][eeg_idx], w_hump=w_humps, w_humps_rises=w_humps_rises, rise_len_thr=rise_len_thr, refined_ridges=refined_ridges)                    

                grid = kwargs['theta_cluster_params'] 
                to_vote = []
                failed = 0
                clusters = []

                # Коррекция артефактов Епанечниковым (должен идти после горбов)
                # scoring.epan_mask = np.full(len(scoring.art_rms), True)
                if kwargs['cluster_artifact']:
                    full_spec, freqs = vote_dict[fname][eeg_idx][day][-2]
                    cluster_artifact = kwargs['cluster_artifact']
                    nr_mask, art_rms = scoring.nr_mask, scoring.art_rms
                    band = (2, 10)
                    theta_210 = np.sum(full_spec[:, (freqs >= band[0])&(freqs <= band[1])], axis=1)
                    if cluster_artifact == 'outside_ridges': cluster = np.hstack((art_rms[np.logical_not(nr_mask), None], theta_210[np.logical_not(nr_mask), None]))
                    elif cluster_artifact == 'everywhere': cluster = np.hstack((art_rms[:, None], theta_210[:, None]))

                    sample = zscore(cluster, axis=0)
                    kd = KD(kernel='epanechnikov').fit(sample)
                    dens_pred = kd.score_samples(sample)
                    if ep[eeg_idx] is None: thr_ca = -5
                    else: thr_ca = ep[eeg_idx]

                    if verbose > 0:
                        plt.rcParams['figure.figsize'] = (30, 8)
                        ax = plt.subplot(121)
                        ax.hist(dens_pred, 50)
                        ax.set_xlabel('Log-likelihood probability density', fontsize=18)
                        ax.set_ylabel('Count', fontsize=18)
                        ax.axvline(thr_ca)
                        ax.tick_params(axis='both', labelsize=13)

                        ax = plt.subplot(122)
                        ax.set_xlabel('Raw signal RMS', fontsize=18)
                        ax.set_ylabel('Theta 2-10 Hz PSD', fontsize=18)
                        scatter = ax.scatter(*sample.T, c=dens_pred < thr_ca)
                        legend1 = ax.legend(scatter.legend_elements()[0], ('Normal', 'Outlier'), prop=font)
                        ax.tick_params(axis='both', labelsize=13)
                        ax.add_artist(legend1)
                        plt.show()
                        plt.rcParams['figure.figsize'] = (30, 5)
                        plt.plot(art_rms)

                    slice_mask = ~nr_mask if cluster_artifact == 'outside_ridges' else np.full(len(nr_mask), True)
                    masked = np.array(art_rms[slice_mask])
                    masked[dens_pred > thr_ca] = None
                    mask_ep = dens_pred > thr_ca
                    print(f'{np.argwhere(~mask_ep).size} artefacts detected by Epanechnikov')
                    if cluster_artifact == 'everywhere': 
                        scoring.art_mask = np.logical_and(scoring.art_mask, mask_ep)  
                        # scoring.epan_mask = mask_ep
                    else: 
                        scoring.art_mask[np.logical_not(nr_mask)] = np.logical_and(scoring.art_mask[np.logical_not(nr_mask)], mask_ep)
                        # scoring.epan_mask[nr_mask] = mask_ep

                    scoring.nr_mask = np.logical_and(scoring.nr_mask, scoring.art_mask)
                    nr_mask = scoring.nr_mask
                    to_plot = np.array(art_rms)
                    to_plot[np.logical_not(slice_mask)] = None
                    to_plot[slice_mask] = masked

                    to_plot_thr = np.array(art_rms)
                    to_plot_thr[scoring.art_mask] = None

                    if verbose > 0:
                        plt.plot(to_plot_thr, c='r')
                        plt.plot(to_plot)
                        plt.ylabel('Raw signal RMS', fontsize=18)
                        plt.xlabel('Epoch #', fontsize=18)
                        plt.tick_params(axis='both', labelsize=13)
                        plt.show()
                vote_dict[fname][eeg_idx][day][3] = scoring.art_mask
                vote_dict[fname][eeg_idx][day][5] = scoring.nr_mask
                scorings[fname][eeg_idx][day] = scoring
                print(f'{scoring.thr_arts_count} artefacts detected by thresholds')
    return vote_dict, scorings

def prescoring_nrem(vote_dict, scorings, visualize_clusters=False, expert_hypnos=[], verbose=0, fad=None):
    fnames = list(vote_dict.keys()) if fad is None else [fad[0]]     
    for fn, fname in enumerate(fnames):
        print('File \"'+fname+'\"')
        eeg_ids = list(vote_dict[fname].keys()) if fad is None else [fad[1]]
        for eeg_idx in eeg_ids:
            print('Animal '+str(eeg_idx))                                       #перестанет работать, если много файлов !
            n_days = range(len(vote_dict[fname][eeg_idx])) if fad is None else [fad[2]]
            for day in n_days:
                scoring = scorings[fname][eeg_idx][day]
                scoring.visualize_clusters = visualize_clusters
                if (verbose > 0) and (scoring.visualize_clusters) and (len(expert_hypnos) > 0):
                    scoring.txthypno = expert_hypnos[fname][eeg_idx][day][:min(len(expert_hypnos[fname][eeg_idx][day]), len(scoring.raw_epochs))]
                # NREM-скоринг, уже должен быть art_mask, делает mask1
                scoring.scoring()
                vote_dict[fname][eeg_idx][day][2] = scoring.mask1
    return vote_dict

horizontal_r_w = []
def prescoring_theta(vote_dict, scoring, **kwargs):
    # vote_dict[fname][eeg_idx].append((to_vote, failed, scoring.mask1, scoring.art_mask, red_to_interactive, scoring.nr_mask, scoring.art_rms, (full_spec, freqs), clusters))
    #LOG.info(kwargs['deltas_for_rem'])
    expert_hypnos = kwargs['expert_hypnos']
    rough_sep = kwargs['rough_separation']
    show_plot = kwargs['show_plot']
    for fn, fname in enumerate(vote_dict):
        # print('File \"'+fname+'\"')
        for eeg_idx in vote_dict[fname]:
            # print('Animal '+str(eeg_idx))                                       #перестанет работать, если много файлов !
            n_days = len(vote_dict[fname][eeg_idx])
            for day in range(n_days):
                # print('Day '+str(day))
                vote_dict[fname][eeg_idx][day][0] = []
                grid = kwargs['theta_cluster_params'] 
                nr_mask = vote_dict[fname][eeg_idx][day][5] #, art_mask   :7
                # if grid['mode'] == 'multiD': 
                full_spec, freqs = vote_dict[fname][eeg_idx][day][-2]
                full_spec = full_spec[nr_mask]
                delta_ = np.sum(full_spec[:, (freqs >= 2) & (freqs <= 6)], axis=1)
                for cl_n, rt in enumerate(vote_dict[fname][eeg_idx][day][-1]):
                    # if cl_n > 0: break
                    # print(f'Cluster # {cl_n+1} - ') #, CLUSTER_NAMES[cl_n]
                    theta_range = grid['ranges'][cl_n]
                    # отделяем верхний NR-кластер
                    # labels_up = GM(n_components=2, covariance_type='full').fit_predict(rt)
                    if show_plot:
                        fig = plt.figure(figsize=(4, 4), dpi=96)
                        plt.scatter(*rt.T, c=expert_hypnos[fname][eeg_idx][day][:len(nr_mask)][nr_mask] if not (expert_hypnos is None) else np.zeros(len(rt)), s=10)
                        plt.ylabel('Delta (2-6 Hz) amplitude, uV') #
                        plt.xlabel(f'Theta {theta_range} amplitude, uV') #(7-9 Hz) 
                        plt.ylim(np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                        plt.xlim(np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99))
                        plt.show()
                    
                    if not rough_sep[eeg_idx]:
                        to_vote_nr = []
                        fig = plt.figure(figsize=(4*len(kwargs['deltas_for_rem']), 4))
                        for db, dband in enumerate(kwargs['deltas_for_rem']):
                            multiDd = full_spec[:, ((freqs >= dband[0]) & (freqs <= dband[1]))]
                            labels_up = GM(n_components=2, covariance_type='full').fit_predict(multiDd) #| ((freqs >= theta_range[0]) & (freqs <= theta_range[1])) 
                            upper_cl, lower_cl = np.mean(rt[:, 1][labels_up == 1]), np.mean(rt[:, 1][labels_up == 0])
                            if upper_cl < lower_cl:
                                labels_up = np.abs(labels_up-1)
                            to_vote_nr.append(labels_up[:, None])
                            if show_plot:
                                ax = plt.subplot(100 + len(kwargs['deltas_for_rem'])*10 + db+1)
                                ax.set_ylabel('Delta '+str(dband))
                                ax.set_xlabel('Theta '+str(grid['ranges'][cl_n]))
                            multiDd = np.sum(multiDd, axis=1)
                            if show_plot:
                                ax.scatter(rt[:, 0], multiDd, c=labels_up, cmap='Accent', s=10)
                                ax.set_ylim(np.percentile(multiDd, 1), np.percentile(multiDd, 99))
                                ax.set_xlim(np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99))
                        if show_plot: plt.show()
                        labels_up = (np.hstack(to_vote_nr).sum(axis=1) >= (len(to_vote_nr) - kwargs['stringency'][eeg_idx])).astype(int) #np.round(np.hstack(to_vote_nr).mean(axis=1)).astype(int)                   
                    if show_plot:
                        fig = plt.figure(figsize=(44, 10))
                        gs = fig.add_gridspec(2, 6,  width_ratios=(2, 1, 2, 2, 2, 2), height_ratios=(2, 1), left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.05, hspace=0.05)
                        ax = fig.add_subplot(gs[0, 0])
                        ax_histy = fig.add_subplot(gs[0, 1], sharey=ax)
                        ax_histy.tick_params(axis="y", labelleft=False)
                        n_, b_, p_ = ax_histy.hist(rt[:, 1], 40, orientation='horizontal', alpha=0)
                    
                    if type(rough_sep[eeg_idx]) in (int, float): # cl_n
                        shift_hor = rough_sep[eeg_idx]#cl_n #grid['hor_shift_threshold_percent']
                        y_range = (np.percentile(rt[:, 1], 99) - np.percentile(rt[:, 1], 1))/100
                        appr_thr = shift_hor#np.mean((np.max(rt[:, 1][labels_up == 0]), np.min(rt[:, 1][labels_up == 1])))
                        # print(f'initial thr = {appr_thr:.2f}, 1% = {y_range:.2f}')
                        # appr_thr += shift_hor*y_range
                        ax.axhline(appr_thr, c='k')
                        # ax.axhline(appr_thr-shift_hor*y_range, c='grey')
                        labels_up = (rt[:, 1] > appr_thr).astype(int)
                    if show_plot:
                        ax.scatter(*rt.T, c=labels_up, cmap='Accent')
                        ax.set_ylim(np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                        ax.set_xlim(np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99))
                        ax_histy.hist(rt[:, 1][labels_up==1], b_, orientation='horizontal', alpha=.7, color='yellow')
                        ax_histy.hist(rt[:, 1][labels_up==0], b_, orientation='horizontal', alpha=.7, color='purple')
                    
                    rt = rt[labels_up == 0]# отрезаем верхний NR-кластер
                    
                    if show_plot: ax = fig.add_subplot(gs[0, 2])
                    if not (expert_hypnos is None):
                        min_len_ = min(len(nr_mask), len(expert_hypnos[fname][eeg_idx][day]))
                        part_expert = expert_hypnos[fname][eeg_idx][day][:min_len_][nr_mask[:min_len_]][labels_up[:min_len_] == 0]
                        if show_plot:
                            ax.scatter(*rt[:min_len_].T, c=part_expert)
                            ax.set_ylim(np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                            ax.set_xlim(np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99))
                            ax_histx = fig.add_subplot(gs[1, 2], sharex=ax)
                            n_, b_, p_ = ax_histx.hist(rt[:, 0], 40, alpha=0)
                            ax_histx.hist(rt[:, 0][:min_len_][part_expert==2], b_, alpha=.7, color='yellow')
                            ax_histx.hist(rt[:, 0][:min_len_][part_expert!=2], b_, alpha=.7, color='purple')
                    
                    # if grid['mode'] == '2D':
                    model = GM(n_components=2, covariance_type='full').fit(rt)
                    labels2d = model.predict(rt)
                    left, right = rt[:, 0][labels2d == 0], rt[:, 0][labels2d == 1]
                    if np.mean(left) > np.mean(right):
                        labels2d = np.abs(labels2d - 1)
                    if show_plot:
                        ax = fig.add_subplot(gs[0, 3])
                        ax.scatter(*rt.T, c=labels2d)
                        ax.set_ylim(np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                        ax.set_xlim(np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99))
                        ax.set_title('Option #0: "natural" separation (2D)')
                        ax_histx = fig.add_subplot(gs[1, 3], sharex=ax)
                        n_, b_, p_ = ax_histx.hist(rt[:, 0], 40, alpha=0)
                        ax_histx.hist(rt[:, 0][labels2d==1], b_, alpha=.7, color='yellow')
                        ax_histx.hist(rt[:, 0][labels2d==0], b_, alpha=.7, color='purple')
                                        
                    # multiD
                    thetaband = grid['ranges'][cl_n]
                    # print(labels_up.shape, full_spec.shape, freqs.shape, thetaband)
                    rt_ = full_spec[labels_up == 0]
                    rt_ = rt_[:, (freqs >= thetaband[0]) & (freqs <= thetaband[1])]
                    model1D = GM(n_components=2, covariance_type='full').fit(rt_)
                    labels1d = model1D.predict(rt_)
                    left, right = rt[:, 0][labels1d == 0], rt[:, 0][labels1d == 1]
                    if np.mean(left) > np.mean(right):
                        labels1d = np.abs(labels1d - 1)
                    if show_plot:    
                        ax = fig.add_subplot(gs[0, 4])
                        ax.scatter(*rt.T, c=labels1d)
                        ax.set_ylim(np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                        ax.set_xlim(np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99))
                        ax.set_title('Option #1: "natural" separation (multiD)')
                        ax_histx = fig.add_subplot(gs[1, 4], sharex=ax)
                        n_, b_, p_ = ax_histx.hist(rt[:, 0], 40, alpha=0)
                        ax_histx.hist(rt[:, 0][labels1d==1], b_, alpha=.7, color='yellow')
                        ax_histx.hist(rt[:, 0][labels1d==0], b_, alpha=.7, color='purple')
                    
                    # ax = fig.add_subplot(gs[0, 4])
                    # ax.set_title('Option #1: vertical separation (1D)')
                    model1d_old = GM(n_components=2, covariance_type='full').fit(rt[:, 0][:, None])
                    labels1d_old = model1d_old.predict(rt[:, 0][:, None])
                    left, right = rt[:, 0][labels1d_old == 0], rt[:, 0][labels1d_old == 1]
                    if np.mean(left) > np.mean(right):
                        labels1d_old = np.abs(labels1d_old - 1)
                        left, right = right, left
                    right0, left1 = np.max(left), np.min(right)
                    if left1 > right0: vert_thr = np.mean((right0, left1))
                    else: vert_thr = right0
                    # ax.scatter(*rt.T, c=labels1d)
                    # ax_histx = fig.add_subplot(gs[1, 4], sharex=ax)
                    # n_, b_, p_ = ax_histx.hist(rt[:, 0], 40, alpha=0)
                    # ax_histx.hist(rt[:, 0][labels1d==1], b_, alpha=.7, color='yellow')
                    # ax_histx.hist(rt[:, 0][labels1d==0], b_, alpha=.7, color='purple')
                    # ax.set_ylim(np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                    # ax.set_xlim(np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99))
                    
                    if show_plot:
                        ax = fig.add_subplot(gs[0, 5])
                        ax.set_title('Option 2: rotated vertical separation (1D-2D)')
                    theta_percent = (np.percentile(rt[:, 0], 99)-np.percentile(rt[:, 0], 1))/100
                    rot_point = (vert_thr+theta_percent*grid['shift_thr_percent'][eeg_idx][cl_n], np.mean(rt[:, 1]))
                    shps = grid['shift_thr_percent'][eeg_idx][cl_n]
                    # print(f'Initial thr = {vert_thr:.2f}; Shifting {theta_percent:.2f}*{shps} = {theta_percent*shps:.2f}')
                    fig_xlims, fig_ylims = (np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99)), (np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                    yr, xr = fig_ylims[1]-fig_ylims[0], fig_xlims[1]-fig_xlims[0], 
                    yl_rot = (yr/xr)*np.tan(np.pi*(90-grid['thr_angle'][eeg_idx][cl_n])/180)*(rt[:, 0]-rot_point[0])+rot_point[1] #
                    rot_mask = (rt[:, 1] < yl_rot).astype(int)
                    if show_plot:
                        ax.scatter(*rt.T, c=rot_mask)
                        ax.set_ylim(np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                        ax.set_xlim(np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99))
                        ax.plot(rt[:, 0], yl_rot, c='grey')
                        ax.scatter(*rot_point, c='cyan', s=60)
                        # angle_title = grid['thr_angle']
                        # plt.title(f'Rotation point: ({rot_point[0]:.2f}, {rot_point[1]:.2f}); angle = {angle_title}')
                        ax_histx = fig.add_subplot(gs[1, 5], sharex=ax)
                        n_, b_, p_ = ax_histx.hist(rt[:, 0], 40, alpha=0)
                        ax_histx.hist(rt[:, 0][rot_mask==1], b_, alpha=.7, color='yellow')
                        ax_histx.hist(rt[:, 0][rot_mask==0], b_, alpha=.7, color='purple')
                        plt.tight_layout()
                        plt.show()
                    
                    masks_to_vote = []
                    for cur_mask in (labels2d, labels1d, rot_mask):
                        mask = np.zeros(len(nr_mask))
                        mid1 = mask[nr_mask] 
                        mid2 = cur_mask
                        mid1[labels_up == 0] = mid2 
                        mask[nr_mask] = mid1
                        masks_to_vote.append(mask)
                    # if red_mask: 
                    #     mask, red_to_interactive = red_masks(scoring, mask, n_back) # можно оптимизировать: рассчитать красненькие один раз, потом только применять к каждой кластеризации
                    
                    vote_dict[fname][eeg_idx][day][0].append(masks_to_vote) # mask

    return vote_dict, scoring
TO_AVE = {}
def prescoring_theta_comb(vote_dict, scoring, **kwargs):
    # vote_dict[fname][eeg_idx].append((to_vote, failed, scoring.mask1, scoring.art_mask, red_to_interactive, scoring.nr_mask, scoring.art_rms, (full_spec, freqs), clusters))
    #LOG.info(kwargs['deltas_for_rem'])
    global TO_AVE, CLUSTER_NAMES
    expert_hypnos = kwargs['expert_hypnos']
    rough_sep = kwargs['rough_separation']
    show_plot = kwargs['show_plot']
    delta_theta_combs = kwargs['delta_theta_combs']
    fad = kwargs['fad']
    fnames = list(vote_dict.keys()) if fad is None else [fad[0]]     
    for fn, fname in enumerate(fnames):
        TO_AVE[fname] = {}
        print('File \"'+fname+'\"')
        eeg_ids = list(vote_dict[fname].keys()) if fad is None else [fad[1]]
        for eeg_idx in eeg_ids:
            TO_AVE[fname][eeg_idx] = []
            print('Animal '+str(eeg_idx))                                       #перестанет работать, если много файлов !
            n_days = range(len(vote_dict[fname][eeg_idx])) if fad is None else [fad[2]]
            for day in n_days:
                random_ids = None
                TO_AVE[fname][eeg_idx].append([])
                print('Day '+str(day))
                vote_dict[fname][eeg_idx][day][0] = []
                grid = kwargs['theta_cluster_params'] 
                nr_mask = vote_dict[fname][eeg_idx][day][5] #, art_mask   :7
                # if grid['mode'] == 'multiD': 
                full_spec, freqs = vote_dict[fname][eeg_idx][day][-2]
                full_spec = full_spec[nr_mask]
                delta_ = np.sum(full_spec[:, (freqs >= 2) & (freqs <= 6)], axis=1)
                CLUSTER_NAMES = []
                for cl_n, (theta_range, dband) in enumerate(delta_theta_combs): #vote_dict[fname][eeg_idx][day][-1]): # цикл по тета-диапазонам (вертикальное разделение)
                    # if cl_n > 0: break
                    #theta_range = grid['ranges'][cl_n]
#                     print(f'Theta # {cl_n+1} - {theta_range}') #, CLUSTER_NAMES[cl_n]
                    # отделяем верхний NR-кластер
                    # labels_up = GM(n_components=2, covariance_type='full').fit_predict(rt)
                    
                    
                    #if not rough_sep[eeg_idx]:
                        # to_vote_nr = []
                        # fig = plt.figure(figsize=(4*len(kwargs['deltas_for_rem']), 4))
                        #for db, dband in enumerate(kwargs['deltas_for_rem']): # цикл по дельтам для горизонтального разделения
                    # print(f'\nDelta # {db+1} - {dband}') #, CLUSTER_NAMES[cl_n]
#                             LOG.info(f'Theta - {theta_range}, delta - {dband}')
                    print((f'Theta - {theta_range}, delta - {dband}'))
                    CLUSTER_NAMES.append((str(theta_range), str(dband)))
                    multiDd = full_spec[:, ((freqs >= dband[0]) & (freqs <= dband[1]))]
                    labels_up = GM(n_components=2, covariance_type='full').fit_predict(multiDd) #| ((freqs >= theta_range[0]) & (freqs <= theta_range[1])) 
                    #rt = vote_dict[fname][eeg_idx][day][-1][cl_n]
                    

                    # to_vote_nr.append(labels_up[:, None])
                    # ax = plt.subplot(100 + len(kwargs['deltas_for_rem'])*10 + db+1)
                    # ax.set_ylabel('Delta '+str(dband))
                    # ax.set_xlabel('Theta '+str(grid['ranges'][cl_n]))
                    multiDd = np.sum(multiDd, axis=1)                                     # сжатая многомерная дельта
                #     ax.scatter(rt[:, 0], multiDd, c=labels_up, cmap='Accent', s=10)
                #     ax.set_ylim(np.percentile(multiDd, 1), np.percentile(multiDd, 99))
                #     ax.set_xlim(np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99))
                # plt.show()
                # labels_up = (np.hstack(to_vote_nr).sum(axis=1) >= (len(to_vote_nr) - kwargs['stringency'][eeg_idx])).astype(int) #np.round(np.hstack(to_vote_nr).mean(axis=1)).astype(int)                    

                    rt_ = full_spec[:, (freqs >= theta_range[0]) & (freqs <= theta_range[1])] # многомерная тета
                    rt_shr = rt_.sum(axis=1) # сжатая многмоерная тета
                    
                    rt = np.hstack((rt_shr[:, None], multiDd[:, None]))
                    
                    if show_plot:
                        fig = plt.figure(figsize=(4, 4), dpi=96)
                        plt.scatter(*rt.T, c=expert_hypnos[fname][eeg_idx][day][:len(nr_mask)][nr_mask] if not (expert_hypnos is None) else np.zeros(len(rt)), s=10)
                        plt.ylabel(f'Delta ({dband} Hz) amplitude, uV') #
                        plt.xlabel(f'Theta {theta_range} amplitude, uV') #(7-9 Hz) 
                        plt.ylim(np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                        plt.xlim(np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99))
                        plt.show()
                    
                    upper_cl, lower_cl = np.mean(rt[:, 1][labels_up == 1]), np.mean(rt[:, 1][labels_up == 0])
                    if upper_cl < lower_cl:
                        labels_up = np.abs(labels_up-1)
                    
                    if random_ids is None:
                        rand_borders = np.argwhere((rt_shr > np.percentile(rt_shr, 50)) & (multiDd < np.percentile(multiDd, 30))).flatten()
                        if rand_borders.size == 0: rand_borders = np.arange(len(rt_shr))
                        random_ids =  np.random.choice(rand_borders, 10)#np.arange(10)
                    if show_plot:
                        fig = plt.figure(figsize=(44, 10)) # большой длинный график
                        gs = fig.add_gridspec(2, 6,  width_ratios=(2, 1, 2, 2, 2, 2), height_ratios=(2, 1), left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.05, hspace=0.05)
                        ax = fig.add_subplot(gs[0, 0])
                        ax_histy = fig.add_subplot(gs[0, 1], sharey=ax)
                        ax_histy.tick_params(axis="y", labelleft=False)

                        n_, b_, p_ = ax_histy.hist(multiDd, 40, orientation='horizontal', alpha=0)

                    if type(rough_sep[eeg_idx]) in (int, float): # жесткий порог для горизонтального разделения
                        shift_hor = rough_sep[eeg_idx]#cl_n #grid['hor_shift_threshold_percent']
                        y_range = (np.percentile(rt[:, 1], 99) - np.percentile(rt[:, 1], 1))/100
                        appr_thr = shift_hor#np.mean((np.max(rt[:, 1][labels_up == 0]), np.min(rt[:, 1][labels_up == 1])))
                        # print(f'initial thr = {appr_thr:.2f}, 1% = {y_range:.2f}')
                        # appr_thr += shift_hor*y_range
                        if show_plot: ax.axhline(appr_thr, c='k')
                        # ax.axhline(appr_thr-shift_hor*y_range, c='grey')
                        labels_up = (rt[:, 1] > appr_thr).astype(int)
                    TO_AVE[fname][eeg_idx][day].append((rt_shr, multiDd))
                    if show_plot:    
                        ax.scatter(rt_shr, multiDd, c=labels_up, cmap='Accent') # полный кластер с окрашенным горизонтальным разделением
                        ax.scatter(rt_shr[random_ids], multiDd[random_ids], marker='x', c='k')
                        ax.set_ylim(np.percentile(multiDd, 1), np.percentile(multiDd, 99))
                        ax.set_xlim(np.percentile(rt_shr, 1), np.percentile(rt_shr, 99))
                        ax_histy.hist(multiDd[labels_up==1], b_, orientation='horizontal', alpha=.7, color='yellow')
                        ax_histy.hist(multiDd[labels_up==0], b_, orientation='horizontal', alpha=.7, color='purple') 
                    
                    
                    rt = rt[labels_up == 0]# отрезаем верхний NR-кластер
                    rt_shr = rt_shr[labels_up == 0]
                    multiDd = multiDd[labels_up == 0]

                    if show_plot: ax = fig.add_subplot(gs[0, 2])
                    if not (expert_hypnos is None):
                        min_len_ = min(len(nr_mask), len(expert_hypnos[fname][eeg_idx][day]))
                        part_expert = expert_hypnos[fname][eeg_idx][day][:min_len_][nr_mask[:min_len_]][labels_up[:min_len_] == 0]
                        if show_plot:
                            ax.scatter(rt_shr, multiDd, c=part_expert)
                            ax.set_ylim(np.percentile(multiDd, 1), np.percentile(multiDd, 99))
                            ax.set_xlim(np.percentile(rt_shr, 1), np.percentile(rt_shr, 99))
                            ax_histx = fig.add_subplot(gs[1, 2], sharex=ax)
                            n_, b_, p_ = ax_histx.hist(rt_shr, 40, alpha=0)
                            ax_histx.hist(rt_shr[:min_len_][part_expert==2], b_, alpha=.7, color='yellow')
                            ax_histx.hist(rt_shr[:min_len_][part_expert!=2], b_, alpha=.7, color='purple')

                    # if grid['mode'] == '2D':
                    model = GM(n_components=2, covariance_type='full').fit(rt) # старое вертикаьное разделение с двумерным Гауссом
                    labels2d = model.predict(rt)
                    left, right = rt[:, 0][labels2d == 0], rt[:, 0][labels2d == 1]
                    if np.mean(left) > np.mean(right):
                        labels2d = np.abs(labels2d - 1)
                    if show_plot:
                        ax = fig.add_subplot(gs[0, 3])
                        # ax.scatter(*rt.T, c=labels2d)
                        # ax.set_ylim(np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                        # ax.set_xlim(np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99))
                        # ax.set_title('Option #0: "natural" separation (2D)')
                        ax_histx = fig.add_subplot(gs[1, 3], sharex=ax)
                        # n_, b_, p_ = ax_histx.hist(rt[:, 0], 40, alpha=0)
                        # ax_histx.hist(rt[:, 0][labels2d==1], b_, alpha=.7, color='yellow')
                        # ax_histx.hist(rt[:, 0][labels2d==0], b_, alpha=.7, color='purple')

                    # multiD
                    rt_ = rt_[labels_up == 0]

                    # print(labels_up.shape, full_spec.shape, freqs.shape, thetaband)

                    model1D = GM(n_components=2, covariance_type='full').fit(rt_)              # новый многомерный Гаусс по тета
                    labels1d = model1D.predict(rt_)
                    left, right = rt_[labels1d == 0].sum(axis=1), rt_[labels1d == 1].sum(axis=1)

                    if np.mean(left) > np.mean(right):
                        labels1d = np.abs(labels1d - 1)
                    if show_plot:
                        ax = fig.add_subplot(gs[0, 4])
                        ax.scatter(rt_shr, multiDd, c=labels1d)
                        ax.set_ylim(np.percentile(multiDd, 1), np.percentile(multiDd, 99))
                        ax.set_xlim(np.percentile(rt_shr, 1), np.percentile(rt_shr, 99))
                        ax.set_title('Option #1: "natural" separation (multiD)')
                        ax_histx = fig.add_subplot(gs[1, 4], sharex=ax)
                        n_, b_, p_ = ax_histx.hist(rt_shr, 40, alpha=0)
                        ax_histx.hist(rt_shr[labels1d==1], b_, alpha=.7, color='yellow')
                        ax_histx.hist(rt_shr[labels1d==0], b_, alpha=.7, color='purple')


                                                                                    # старая хрень с крутящимся сдвигаемым порогом
                    # ax = fig.add_subplot(gs[0, 4])
                    # ax.set_title('Option #1: vertical separation (1D)')
                    model1d_old = GM(n_components=2, covariance_type='full').fit(rt[:, 0][:, None])
                    labels1d_old = model1d_old.predict(rt[:, 0][:, None])

                    if np.unique(labels1d_old).size == 1:
                        labels1d_old = np.zeros(len(labels1d_old))
                        labels1d_old[rt[:, 0] > np.median(rt[:, 0])] = 1

                    left, right = rt[:, 0][labels1d_old == 0], rt[:, 0][labels1d_old == 1]

                    if np.mean(left) > np.mean(right):
                        labels1d_old = np.abs(labels1d_old - 1)
                        left, right = right, left
                    right0, left1 = np.max(left), np.min(right)
                    if left1 > right0: vert_thr = np.mean((right0, left1))
                    else: vert_thr = right0

                    if show_plot: 
                        ax = fig.add_subplot(gs[0, 5])
                        # ax.set_title('Option 2: rotated vertical separation (1D-2D)')
                    theta_percent = (np.percentile(rt[:, 0], 99)-np.percentile(rt[:, 0], 1))/100
                    rot_point = (0, 0) #(vert_thr+theta_percent*grid['shift_thr_percent'][eeg_idx][cl_n], np.mean(rt[:, 1]))
                    shps = 0 #grid['shift_thr_percent'][eeg_idx][cl_n]
                    # print(f'Initial thr = {vert_thr:.2f}; Shifting {theta_percent:.2f}*{shps} = {theta_percent*shps:.2f}')
                    fig_xlims, fig_ylims = (np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99)), (np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                    yr, xr = fig_ylims[1]-fig_ylims[0], fig_xlims[1]-fig_xlims[0], 
                    yl_rot = 0#(yr/xr)*np.tan(np.pi*(90-grid['thr_angle'][eeg_idx][cl_n])/180)*(rt[:, 0]-rot_point[0])+rot_point[1] #
                    rot_mask = (rt[:, 1] < yl_rot).astype(int)
                    # ax.scatter(*rt.T, c=rot_mask)
                    # ax.set_ylim(np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                    # ax.set_xlim(np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99))
                    # ax.plot(rt[:, 0], yl_rot, c='grey')
                    # ax.scatter(*rot_point, c='cyan', s=60)
                    # angle_title = grid['thr_angle']
                    # plt.title(f'Rotation point: ({rot_point[0]:.2f}, {rot_point[1]:.2f}); angle = {angle_title}')
                    if show_plot: 
                        ax_histx = fig.add_subplot(gs[1, 5], sharex=ax)
                        # n_, b_, p_ = ax_histx.hist(rt[:, 0], 40, alpha=0)
                        # ax_histx.hist(rt[:, 0][rot_mask==1], b_, alpha=.7, color='yellow')
                        # ax_histx.hist(rt[:, 0][rot_mask==0], b_, alpha=.7, color='purple')
                        plt.tight_layout()
                        plt.show()

                    ########## check 4th stage
#                             if not (expert_hypnos is None):
#                                 th_bckp, lhbck = np.array(scoring.txthypno), np.array(labels1d)

#                                 scoring.txthypno = np.array(expert_hypnos[fname][eeg_idx][day])[:len(nr_mask)]
#                                 scoring.txthypno = scoring.txthypno[nr_mask]
#                                 # scoring.txthypno = scoring.txthypno[labels_up == 0]
#                                 scoring.txthypno[scoring.txthypno != 2] = 0
#                                 scoring.txthypno[scoring.txthypno == 2] = 1

#                                 scoring.last_hypno = np.zeros(len(scoring.txthypno))
#                                 scoring.last_hypno[labels_up == 0] = labels1d
#                                 # scoring.last_hypno, scoring.txthypno = labels1d, scoring.txthypno

#                                 scoring.compare_binary(None)
#                                 scoring.txthypno, labels1d = th_bckp, lhbck
                            # return
                            ##########

                    masks_to_vote = []
                    for cur_mask in (labels2d, labels1d, rot_mask):
                        mask = np.zeros(len(nr_mask))
                        mid1 = mask[nr_mask] 
                        mid2 = cur_mask
                        mid1[labels_up == 0] = mid2 
                        mask[nr_mask] = mid1
                        masks_to_vote.append(mask)
                            # if red_mask: 
                            #     mask, red_to_interactive = red_masks(scoring, mask, n_back) # можно оптимизировать: рассчитать красненькие один раз, потом только применять к каждой кластеризации

                    vote_dict[fname][eeg_idx][day][0].append(masks_to_vote) # mask
                    
                    # hor sep fig
#                     if theta_range ==(6, 9) and dband == (2, 7):
#                         rt_shr = full_spec[:, (freqs >= theta_range[0]) & (freqs <= theta_range[1])].sum(axis=1) 
#                         multiDd = full_spec[:, ((freqs >= dband[0]) & (freqs <= dband[1]))].sum(axis=1)
#                         plt.rcParams['font.family'] = 'Cambria'
#                         hypno_ = expert_hypnos[fname][eeg_idx][day][:len(nr_mask)][nr_mask]
#                         stage_map = {0: 'Wake', 1: 'NREM', 2: 'REM'}
#                         colors = {'Wake': 'darkcyan', 'NREM': 'purple', 'REM': 'gold'}
#                         fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.7), sharey=True)
#                         fig.text(0.5, -.01, 'Theta (6-9 Hz) amplitude, μV', ha='center', fontsize=18, family='Cambria')
#                         ax = axes.flatten()[0]
#                         for c in range(3):
#                             ax.scatter(rt_shr[hypno_ == c], multiDd[hypno_ == c], alpha=.5, s=10, label=stage_map[c], c=colors[stage_map[c]])                
#                         ax.legend(fontsize=14, handletextpad=0, borderpad=.5)
#                         # ax.set_ylim(140, 600)
#                         # ax.set_xlim(90, 360)
#                         ax.set_ylabel('Delta (2-6 Hz) amplitude, μV', fontsize=18, family='Cambria')
#                         ax.tick_params(axis='both', labelsize=14)
#                         ax.text(300, 470, 'A', fontsize=24)
                        
#                         ax = axes.flatten()[1]
#                         colors2 = {1: 'purple', 0: 'grey'}
#                         stage_map2 = {1: 'NREM', 0: 'Wake+REM'}
#                         for c in range(2):
#                             ax.scatter(rt_shr[labels_up==c], multiDd[labels_up==c], c=colors2[c], alpha=.5, s=10, label=stage_map2[c]) # полный кластер с окрашенным горизонтальным разделением
#                         ax.legend(fontsize=14, handletextpad=0, borderpad=.5)
#                         # ax.set_ylim(140, 600)
#                         # ax.set_xlim(90, 360)
#                         ax.tick_params(axis='both', labelsize=14)
#                         ax.text(300, 470, 'B', fontsize=24)
                        
#                         plt.tight_layout()
#                         plt.savefig(r'G:\LA\2024\qual_figs\horizontal_separation.png', dpi=300, bbox_inches='tight')
#                         plt.show()
                        # return
                        
                        
#                     vert sep fig
#                         fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.7), sharey=True)
#                         fig.text(0.5, -.01, 'Theta (6-9 Hz) amplitude, μV', ha='center', fontsize=18, family='Cambria')
#                         rt_shr, multiDd, hypno_ = rt_shr[labels_up == 0], multiDd[labels_up == 0], hypno_[labels_up == 0]
#                         ax = axes.flatten()[0]
#                         for c in range(3):
#                             ax.scatter(rt_shr[hypno_ == c], multiDd[hypno_ == c], alpha=.5, s=10, label=stage_map[c], c=colors[stage_map[c]])                
#                         ax.legend(fontsize=14, handletextpad=0, borderpad=.5)
#                         # ax.set_ylim(140, 490)
#                         # ax.set_xlim(90, 360)
#                         ax.set_ylabel('Delta (2-6 Hz) amplitude, μV', fontsize=18, family='Cambria')
#                         ax.tick_params(axis='both', labelsize=14)
#                         ax.text(300, 470, 'A', fontsize=24)
                        
#                         ax = axes.flatten()[1]
#                         colors2 = {1: 'gold', 0: 'darkcyan'}
#                         stage_map2 = {1: 'REM', 0: 'Wake'}
#                         for c in range(2):
#                             ax.scatter(rt_shr[labels1d==c], multiDd[labels1d==c], c=colors2[c], alpha=.5, s=10, label=stage_map2[c]) # полный кластер с окрашенным горизонтальным разделением
#                         ax.legend(fontsize=14, handletextpad=0, borderpad=.5)
#                         # ax.set_ylim(140, 490)
#                         # ax.set_xlim(90, 360)
#                         ax.tick_params(axis='both', labelsize=14)
#                         ax.text(300, 470, 'B', fontsize=24)
                        
#                         plt.tight_layout()
#                         plt.savefig(r'G:\LA\2024\qual_figs\vertical_separation.png', dpi=300, bbox_inches='tight')
#                         plt.show()
#                         return
                        
        
                print(f'{len(vote_dict[fname][eeg_idx][day][0])} theta-delta combinations in total\n')
            # for paper
#                 colors = {'Wake': 'darkcyan', 'NREM': 'purple', 'REM': 'gold'}
#                 stage_map = {0: 'Wake', 1: 'NREM', 2: 'REM'}
#                 fig = plt.figure(figsize=(4.15, 4.15))
#                 to_ave_plotx = np.vstack([el[0].reshape((1, -1)) for el in TO_AVE[fname][eeg_idx][day]]).mean(axis=0)
#                 to_ave_ploty = np.vstack([el[1].reshape((1, -1)) for el in TO_AVE[fname][eeg_idx][day]]).mean(axis=0)
#                 hpno = expert_hypnos[fname][eeg_idx][day][:len(nr_mask)][nr_mask] if not (expert_hypnos is None) else np.zeros(len(TO_AVE[fname][eeg_idx][day][0][0]))
#                 for c in np.unique(hpno):
#                     plt.scatter(to_ave_plotx[hpno == c], to_ave_ploty[hpno == c], c=colors[stage_map[c]], label=stage_map[c], s=10, alpha=.5)
#                 plt.legend(fontsize=14, handletextpad=0, borderpad=.5, loc='upper left')
#                 plt.tick_params(axis='both', labelsize=14)
                
#                 for rnd_idx in random_ids[:5]:
#                     plt.scatter(to_ave_plotx[rnd_idx], to_ave_ploty[rnd_idx], marker='s', c='k', s=12)
#                     for prime in TO_AVE[fname][eeg_idx][day]:
#                         plt.plot([to_ave_plotx[rnd_idx], prime[0][rnd_idx]], [to_ave_ploty[rnd_idx], prime[1][rnd_idx]], lw=.5, c='k')
#                         plt.scatter(prime[0][rnd_idx], prime[1][rnd_idx], marker='*', c='grey', s=12)
#                 plt.ylim(np.percentile(to_ave_ploty, 0), np.percentile(to_ave_ploty, 99))
#                 plt.xlim(np.percentile(to_ave_plotx, 1), np.percentile(to_ave_plotx, 99.5))
#                 plt.ylabel('Average delta amplitude, μV', fontsize=18, family='Cambria')
#                 plt.xlabel('Average theta amplitude, μV', fontsize=18, family='Cambria')
#                 plt.savefig(r'G:\LA\2024\qual_figs\theta_delta_random_scatter.png', dpi=300, bbox_inches='tight')
#                 plt.show()

         # for paper compare clusters
                # colors = {'Wake': 'darkcyan', 'NREM': 'purple', 'REM': 'gold'}
                # stage_map = {0: 'Wake', 1: 'NREM', 2: 'REM'}
                # fig, axes = plt.subplots(1, 3, figsize=(8.3, 3))
                # hpno = expert_hypnos[fname][eeg_idx][day][:len(nr_mask)][nr_mask] if not (expert_hypnos is None) else np.zeros(len(TO_AVE[fname][eeg_idx][day][0][0]))
                # for i, (x, y) in enumerate(TO_AVE[fname][eeg_idx][day][:3]):
                #     ax = axes.flatten()[i]
                #     for c in np.unique(hpno):
                #         ax.scatter(x[hpno == c], y[hpno == c], c=colors[stage_map[c]], label=stage_map[c], s=10, alpha=.5)
                #     for j, rnd_idx in enumerate(random_ids[:9]):
                #         ax.text(x[rnd_idx], y[rnd_idx], str(j+1), c='k')
                #     ax.legend(fontsize=12, handletextpad=0, borderpad=.01, loc='upper left')
                #     ax.tick_params(axis='both', labelsize=14)
                #     # if i == 0: ax.set_ylabel(f'Delta ({delta_theta_combs[i][0][0]}-{delta_theta_combs[i][0][1]} Hz) amplitude, μV', fontsize=18, family='Cambria')
                #     ax.set_ylabel(f'{delta_theta_combs[i][0][0]}-{delta_theta_combs[i][0][1]} Hz', fontsize=12, family='Cambria')
                #     ax.set_xlabel(f'{delta_theta_combs[i][1][0]}-{delta_theta_combs[i][1][1]} Hz', fontsize=12, family='Cambria')
                # fig.text(0.5, -.01, 'Theta amplitude, μV', ha='center', fontsize=18, family='Cambria')
                # fig.text(-.01, .2, 'Delta amplitude, μV', ha='center', fontsize=18, family='Cambria', rotation=90)
                # plt.tight_layout()
                # plt.savefig(r'G:\LA\2024\qual_figs\theta_delta\theta_delta_random_scatter_compare{}.png'.format(str(datetime.now()).replace(':', '_')), dpi=300, bbox_inches='tight')
                # plt.show()
                    
    return vote_dict, scoring

def spindle_check(hypno, full_spec, freqs):
    print('Checking spindles')
    # spindle_feats = [np.sum(full_spec[:, (freqs >= band[0])&(freqs <= band[1])], axis=1) for band in spindles]
    # for spf in spindle_feats:
    #     plt.rcParams['figure.figsize'] = (30, 4)
    #     ax = plt.subplot(111)
    #     ax.plot(scoring.last_hypno == 2, c='orange')
    #     ax = ax.twinx()
    #     ax.plot(spf, c='lightblue')
    #     ax.axhline(np.mean(spf), c='g')
    #     ax.axhline(np.mean(spf)+np.std(spf), c='r')
    #     plt.show()
    # bsln = [(np.mean(ft), np.std(ft)) for ft in spindle_feats]
    
    # find dominant theta frequency in REM
    theta_dom = []
    freq_slc = (freqs > 5) & (freqs < 10)
    freq_th = freqs[freq_slc]
    plt.rcParams['figure.figsize'] = (10, 4)
    theta_chunks = []
    for i in range(len(hypno)):
        if hypno[i] == 2: 
            theta_spec = full_spec[i, freq_slc]
            theta_chunks.append(theta_spec[None, :])
            plt.plot(freq_th, theta_spec, c='grey', alpha=.1)
            theta_dom.append(freq_th[np.argmax(theta_spec)])
    plt.plot(freq_th, np.mean(np.vstack(theta_chunks), axis=0))
    dom_theta = np.mean(theta_dom)
    print(f'Typical theta for REM is {dom_theta:.1f}')
    plt.axvline(dom_theta, c='r')
    plt.xlabel('Frequency, Hz')
    plt.ylabel('~Power')
    plt.show()
    
    # then use (dom-1 Hz, dom+1 Hz) and it x2 harmonic as the spindle criterion
    spindles = [(dom_theta-1, dom_theta+1), (2*dom_theta-2, 2*dom_theta+2)]
    spindle_feats = np.hstack([np.sum(full_spec[:, (freqs >= band[0])&(freqs <= band[1])], axis=1)[:, None] for band in spindles])
    rem_thr = np.mean(spindle_feats[hypno == 2], axis=0).flatten()
    
    sample = zscore(spindle_feats, axis=0)
    kd = KD(kernel='epanechnikov').fit(sample)
    dens_pred = kd.score_samples(sample)
    thr_ca = -6
    plt.rcParams['figure.figsize'] = (30, 8)
    ax = plt.subplot(121)
    ax.hist(dens_pred, 50)
    ax.set_xlabel('Log-likelihood probability density', fontsize=18)
    ax.set_ylabel('Count', fontsize=18)
    ax.axvline(thr_ca)
    ax.tick_params(axis='both', labelsize=13)

    ax = plt.subplot(122)
    ax.set_xlabel(f'Theta ({spindles[0][0]:.1f} - {spindles[0][1]:.1f}) Hz', fontsize=18)
    ax.set_ylabel(f'Theta ({spindles[1][0]:.1f} - {spindles[1][1]:.1f}) Hz', fontsize=18)
    scatter = ax.scatter(*sample.T, c=dens_pred < thr_ca)
    legend1 = ax.legend(scatter.legend_elements()[0], ('Normal', 'Spindle'), prop=font)
    ax.tick_params(axis='both', labelsize=13)
    ax.add_artist(legend1)
    plt.show()
    
    hypno[dens_pred < thr_ca] = 4
    for i in np.argwhere(hypno == 4):        
        print(f'A spindle found at {i*10}-{(i+1)*10} s')
        # plt.rcParams['figure.figsize'] = (10, 4)
        # plt.plot(freqs[freqs < 25], full_spec[i, freqs < 25])
        # for band in spindles: 
        #     plt.axvline(band[0], c='black', ls='--', marker=2)
        #     plt.axvline(band[1], c='black', ls='--', marker=1)
        # plt.xlabel('Frequency, Hz')
        # plt.ylabel('~Power')
        # plt.show()

    # now look for spindles in every state
#     for i in range(len(hypno)):
#         # plt.rcParams['figure.figsize'] = (10, 4)
#         # plt.plot(freqs[freqs < 25], full_spec[i, freqs < 25])
#         # for band in spindles: 
#         #     plt.axvline(band[0], c='r')
#         #     plt.axvline(band[1], c='r')
#         # plt.show()
#         # check = [spindle_feats[j][i] > (bsln[j][0] + 2*bsln[j][1]) for j in range(len(spindle_feats))]
#         check = [spindle_feats[i][j] > 2*rem_thr[j] for j in range(spindle_feats.shape[1])]
#         check = all(check)
#         if check:
#             print(f'A spindle found at {i*10}-{(i+1)*10} s')
#             plt.plot(freqs[freqs < 25], full_spec[i, freqs < 25])
#             for band in spindles: 
#                 plt.axvline(band[0], c='black', ls='--', marker=2)
#                 plt.axvline(band[1], c='black', ls='--', marker=1)
#             plt.show()
            
#             hypno[i] = 4
#             # mark all continuous REM episode as a spindle
#             # c = 0
#             # cur_state = hypno[i]
#             # while ((i-c) >= 0) and (hypno[i-c] == cur_state):
#             #     hypno[i-c] = 4
#             #     c += 1
#             # c = 0
#             # while ((i+c) < len(hypno)) and (hypno[i+c] == cur_state):
#             #     hypno[i+c] = 4
#             #     c += 1
    return hypno
        

LOG = None
fh = None
def start_log(fname):
    global LOG, fh 
    # init_time = str(datetime.now().strftime("%Y_%m_%dT%H_%M_%S_%f"))
    # logging.basicConfig(filename=init_time + '.txt', level=logging.DEBUG, format='%(message)s')
    LOG = logging.getLogger()
    LOG.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename=fname + '.txt')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(message)s')
    fh.setFormatter(formatter)
    LOG.addHandler(fh)
    
def kill_log():
    global LOG, fh
    LOG.removeHandler(fh)
    # LOG.shutdown() 
    del LOG, fh

def in_hull(p, hull):
    # if not isinstance(hull,Delaunay):
    #     hull = Delaunay(hull)
    return hull.find_simplex(p)>=0
    
def end_scoring(vote_dict, fnames, scoring, rule, cluster_artifact, n_back, n_hours_cycle_bars, minutes, hours_fragm, red_mask, n_hours_cycle_rem, expert_hypnos, verbose, mask_option, single_nr, single_wake, single_rem1_2_0, single_rem1_2_1, save_dir='./', save_spec_params={}, spindles=None, spec_max = (), ep=[None]*8, nr_epanech=[-4.5]*8, rem_epanech=[-3.6]*8, w_epanech=[-4]*8):
    # %matplotlib inline
    
    to_interactive = {}
    hypnos = {}
    fragms = {}
    fracs, exp_fracs = {}, {}
    saved_specs = {}
    for fn, fname in enumerate(fnames):
        print('\n\nFile \"'+fname+'\"')
       # LOG.info('File \"'+fname+'\"')
        to_interactive[fname] = {}
        fracs[fname] = []
        exp_fracs[fname] = []
        fragms[fname] = []
        hypnos[fname] = []
        saved_specs[fname] = {}
        for eeg_idx in vote_dict[fname]:
            print('Animal '+str(eeg_idx))
            n_days = len(vote_dict[fname][eeg_idx])
            to_interactive[fname][eeg_idx] = []
            saved_specs[fname][eeg_idx] = []
            for day in range(n_days):
                print('Day '+str(day))
    # ГОЛОСОВАНИЕ
                to_vote, failed, mask1, art_mask, red_to_interactive = vote_dict[fname][eeg_idx][day][:5]
                to_vote_ = [mset[mask_option[eeg_idx][optn]] for optn, mset in enumerate(to_vote)]
                turn_on = np.argwhere(np.array(mask_option[eeg_idx]).flatten() != -1).flatten()
                if len(turn_on) < len(to_vote_):
                    to_vote = []
                    for stay_idx in turn_on:
                        to_vote.append(to_vote_[stay_idx])
                else: to_vote = to_vote_
                print(f"voting using {len(to_vote)} clusters. rule = {rule[eeg_idx]-failed}")
                
                mask = rem_vote(to_vote, rule=rule[eeg_idx]-failed) # ПРАВИЛО ГОЛОСОВАНИЯ rule МОЖЕТ БЫТЬ ЦЕЛЫМ ЧИСЛОМ, КОТОРОЕ БУДЕТ ОЗНАЧАТЬ МИНИМАЛЬНОЕ КОЛИЧЕСТВО ГОЛОСОВ ДЛЯ ПРИНЯТИЯ РЕШЕНИЯ REM
                # СЕЙЧАС ВСЕГО 16 РАЗНЫХ КЛАСТЕРИЗАЦИЙ, ЗНАЧИТ, rule=8 ДЕТЕКТИРУЕТ REM, КОГДА ХОТЯ БЫ ПОЛОВИНА КЛАСТЕРИЗАЦИЙ ОПРЕДЕЛИЛА REM; 
                # ВСЕГДА ОТНИМАЕМ КОЛ-ВО ПРОВАЛЕННЫХ КЛАСТЕРИЗАЦИЙ failed
                
                ########## check 4th stage
                # scoring.txthypno = np.array(expert_hypnos[fname][eeg_idx][day])
                # th_bckp, lhbck = np.array(scoring.txthypno), np.array(mask)
                # scoring.txthypno[scoring.txthypno != 2] = 0
                # scoring.txthypno[scoring.txthypno == 2] = 1
                # scoring.last_hypno, scoring.txthypno = mask, scoring.txthypno
                # scoring.compare(None)
                # scoring.txthypno, mask = lhbck, th_bckp
                # return
                ##########
                
                # далее используется экземпляр scoring последнего дня последнего животного в последнем файле. что-то может быть некорректным
                if red_mask:   
                    scoring.last_hypno = combine_masks(mask, mask1, int(n_hours_cycle_rem*3600/scoring.window_sec2), art_mask)
                else:
                    # print(mask.shape, mask1.shape, art_mask.shape)
                    scoring.last_hypno = combine_masks(mask, mask1, int(n_hours_cycle_rem*3600/scoring.window_sec2), art_mask)
                
                if cluster_artifact:
                    nr_mask, art_rms, (full_spec, freqs) = vote_dict[fname][eeg_idx][day][5:-1]
                    band = (2, 10)
                    theta_210 = np.sum(full_spec[:, (freqs >= band[0])&(freqs <= band[1])], axis=1)
                    if cluster_artifact == 'outside_ridges': cluster = np.hstack((art_rms[np.logical_not(nr_mask), None], theta_210[np.logical_not(nr_mask), None]))
                    elif cluster_artifact == 'everywhere': cluster = np.hstack((art_rms[:, None], theta_210[:, None]))

                    sample = zscore(cluster, axis=0)
                    kd = KD(kernel='epanechnikov').fit(sample)
                    dens_pred = kd.score_samples(sample)
                    if ep[eeg_idx] is None: thr_ca = -5
                    else: thr_ca = ep[eeg_idx]
                    if verbose > 0:
                        plt.rcParams['figure.figsize'] = (30, 8)
                        ax = plt.subplot(121)
                        ax.hist(dens_pred, 50)
                        ax.set_xlabel('Log-likelihood probability density', fontsize=18)
                        ax.set_ylabel('Count', fontsize=18)
                        ax.axvline(thr_ca)
                        ax.tick_params(axis='both', labelsize=13)
                        
                        ax = plt.subplot(122)
                        ax.set_xlabel('Raw signal RMS', fontsize=18)
                        ax.set_ylabel('Theta 2-10 Hz PSD', fontsize=18)
                        scatter = ax.scatter(*sample.T, c=dens_pred < thr_ca)
                        legend1 = ax.legend(scatter.legend_elements()[0], ('Normal', 'Outlier'), prop=font)
                        ax.tick_params(axis='both', labelsize=13)
                        ax.add_artist(legend1)
                        plt.show()
                        plt.rcParams['figure.figsize'] = (30, 5)
                        plt.plot(art_rms)

                    slice_mask = np.logical_not(nr_mask) if cluster_artifact == 'outside_ridges' else np.full(len(nr_mask), True)
                    masked = np.array(art_rms[slice_mask])
                    masked[dens_pred > thr_ca] = None
                    to_plot = np.array(art_rms)
                    to_plot[np.logical_not(slice_mask)] = None
                    to_plot[slice_mask] = masked
                    
                    if verbose > 0:
                        plt.plot(to_plot)
                        plt.ylabel('Raw signal RMS', fontsize=18)
                        plt.xlabel('Epoch #', fontsize=18)
                        plt.tick_params(axis='both', labelsize=13)
                        plt.show()

                    _mask = np.array(np.isnan(to_plot))
                    scoring.last_hypno = cluster_art(scoring.last_hypno, _mask)
                
                
                
                if single_nr: scoring.last_hypno = no_singles(scoring.last_hypno, 1) # 0-1-0 -> 0-0-0; 2-1-2 -> 2-2-2
                if single_wake: scoring.last_hypno = no_singles(scoring.last_hypno, 0) # 1-0-1 -> 1-1-1; 2-0-2 -> 2-2-2
                
                scoring.no_rems_in_wake(n_back) # 0-2-...-2-0 -> 0-0-...-0-0 и 0-2-...-2-1 -> 0-0-...-0-1, неважно сколько двоек
                # scoring.no_single_a_between_b_and_c(0, 1, 2) # 1-0-2 -> 1-1-2, повторяет строку 607, так что бесполезно
                # scoring.no_single_a_between_b_and_c(1, 2, 2) # 2-1-2 -> 2-2-2, повторяет строку 605, так что бесполезно
                
                if single_rem1_2_0: scoring.no_single_a_between_b_and_c(2, 1, 0) #  1-2-0 -> 1-1-0
                if single_rem1_2_1: scoring.no_single_a_between_b_and_c(2, 1, 1) # 1-2-1 -> 1-1-1
                
                artifacts = np.logical_and(art_mask, _mask)
                # spindles
                if spindles:
                    scoring.last_hypno = spindle_check(scoring.last_hypno, full_spec, freqs)
                stages = np.unique(scoring.last_hypno)
                colors = {0: 'g', 1: 'b', 2: 'r', 4: 'purple'}
                
                # clustering metrics
                hypno_labels = np.array(scoring.last_hypno)[nr_mask].reshape((-1, 1))
                for combidx, (th_x, del_y) in enumerate(TO_AVE[fname][eeg_idx][day]):
                    points = zscore(np.hstack((th_x.reshape((-1, 1)), del_y.reshape((-1, 1)))), axis=0)
                    LOG.info(f'Theta - {CLUSTER_NAMES[combidx][0]}, delta - {CLUSTER_NAMES[combidx][1]}')
                    # print(f'Silhoutte = {silhs(points, hypno_labels):.3f}\nCalinski-Harabasz = {chs(points, hypno_labels):.3f}\nDavies-Bouldin = {dbs(points, hypno_labels):.3f}')
                    LOG.info(f'Silhoutte = {silhs(points, hypno_labels):.3f}\nCalinski-Harabasz = {chs(points, hypno_labels):.3f}\nDavies-Bouldin = {dbs(points, hypno_labels):.3f}')
                    # plt.rcParams['figure.figsize'] = (5, 5)
                    # plt.scatter(*points.T, c=hypno_labels, alpha=.5, s=5)
                    # plt.xlabel(CLUSTER_NAMES[combidx][0])
                    # plt.ylabel(CLUSTER_NAMES[combidx][1])
                    # plt.show()
                #
                ## red circle
                hypno_labels = hypno_labels.flatten()
                av_clust = np.hstack((np.vstack([el[0].reshape((1, -1)) for el in TO_AVE[fname][eeg_idx][day]]).mean(axis=0)[:, None], np.vstack([el[1].reshape((1, -1)) for el in TO_AVE[fname][eeg_idx][day]]).mean(axis=0)[:, None]))
                av_clust_rem = av_clust[hypno_labels == 2]
                
                # используем только наиболее плотную часть REM-кластера
                kd_rem_thr = rem_epanech[eeg_idx]
                av_clust_rem_z = zscore(av_clust_rem, axis=0)
                kd_rem = KD(kernel='epanechnikov').fit(av_clust_rem_z)
                dens_pred_rem = kd_rem.score_samples(av_clust_rem_z)
                av_clust_rem = av_clust_rem[dens_pred_rem > kd_rem_thr]
                
                # граница NR-кластера для удаления FP REM
                kd_nr_thr = nr_epanech[eeg_idx]
                av_clust_nr = av_clust[hypno_labels == 1]
                av_clust_nr_z = zscore(av_clust_nr, axis=0)
                kd_nr = KD(kernel='epanechnikov').fit(av_clust_nr_z)
                dens_pred_nr = kd_nr.score_samples(av_clust_nr_z)
                av_clust_nr = av_clust_nr[dens_pred_nr > kd_nr_thr]
                
                # граница W-кластера для удаления FP REM
                kd_w_thr = w_epanech[eeg_idx]
                av_clust_w = av_clust[hypno_labels == 0]
                av_clust_w_z = zscore(av_clust_w, axis=0)
                kd_w = KD(kernel='epanechnikov').fit(av_clust_w_z)
                dens_pred_w = kd_nr.score_samples(av_clust_w_z)
                dens_pred_w[np.abs(dens_pred_w) == np.inf] = -10
                av_clust_w = av_clust_w[dens_pred_w > kd_w_thr]
                
                
                plt.rcParams['figure.figsize'] = (13, 4)
                ax = plt.subplot(131)
                ax.hist(dens_pred_rem, 30)
                ax.axvline(kd_rem_thr, c='k')
                ax.set_title('REM')
                
                ax = plt.subplot(132)
                ax.hist(dens_pred_nr, 30)
                ax.axvline(kd_nr_thr, c='k')
                ax.set_title('NR')
                
                ax = plt.subplot(133)
                ax.hist(dens_pred_w, 30)
                ax.axvline(kd_w_thr, c='k')
                ax.set_title('W')
                plt.show()
       
                # коррекция красным кружком
                # remify = gm_rem.predict_proba(av_clust)
                expand = 1 #.2
                
                mean_rem = np.mean(av_clust_rem, axis=0)
                centrize = lambda x: expand * (x - mean_rem)
                
                REMhull = Delaunay(av_clust_rem)
                remify = np.array([in_hull(p, REMhull) for p in av_clust])

                # remify = remify.sum(axis=1)
                # print('REM before = ', np.where(hypno_labels == 2)[0].size)
                hypno_labels[remify] = 2
                # print('REM after REM Epanech. = ', np.where(hypno_labels == 2)[0].size)
                                
                
                NRhull = Delaunay(av_clust_nr)
                nrfy = np.array([in_hull(p, NRhull) for p in av_clust])
                hypno_labels[nrfy & (hypno_labels == 2)] = 1
                # print('REM after NR Epanech. = ', np.where(hypno_labels == 2)[0].size)
                
                Whull = Delaunay(av_clust_w)
                wfy = np.array([in_hull(p, Whull) for p in av_clust])
                hypno_labels[wfy & (hypno_labels == 2)] = 0
                # print('REM after W Epanech. = ', np.where(hypno_labels == 2)[0].size)
                
                # строгие пороги
                left_rem = np.percentile(av_clust_rem[:, 0], 10)
                updwn_av = np.mean(av_clust_rem[:, 1])
                hypno_labels[(av_clust[:, 0] > left_rem) & (av_clust[:, 1] < updwn_av)] = 2
                # hypno_labels[(av_clust[:, 0] > 720) & (av_clust[:, 1] < 490)] = 2
                # print('REM after yellow = ', np.where(hypno_labels == 2)[0].size)
                
                scoring.last_hypno = np.array(scoring.last_hypno)
                scoring.last_hypno[nr_mask] = hypno_labels

                plt.rcParams['figure.figsize'] = (7, 7)
                plt.scatter(*av_clust.T, c=hypno_labels, alpha=.5)
                plt.xlabel('Av. theta')
                plt.ylabel('Av. delta')
                plt.scatter(*av_clust_rem.T, c='k', marker='x', alpha=.6, s=8)
                plt.scatter(*av_clust_nr.T, c='orange', marker='x', alpha=.6, s=8)
                
                if not (expert_hypnos is None):
                    nr_exp_hypn = expert_hypnos[fname][eeg_idx][day][:len(nr_mask)]
                    nr_exp_hypn = nr_exp_hypn[nr_mask]
                    fp_rem = (hypno_labels == 2) & (nr_exp_hypn != 2)
                    fn_rem = (hypno_labels != 2) & (nr_exp_hypn == 2)
                
                    plt.scatter(*av_clust[fp_rem].T, c='b', marker='x')
                    plt.scatter(*av_clust[fn_rem].T, c='r', marker='x')
                plt.ylim(np.percentile(av_clust[:, 1], 1), np.percentile(av_clust[:, 1], 99))
                plt.xlim(np.percentile(av_clust[:, 0], 1), np.percentile(av_clust[:, 0], 99))
                plt.show()
                ##
                
                
                to_interactive[fname][eeg_idx].append((scoring.last_hypno, red_to_interactive))
                if verbose > 0: scoring.bars_by_cycles(n_hours_cycle=n_hours_cycle_bars) # соотношение стадий в циклах заданной длины
                if verbose > 0: 
                    plt.rcParams['figure.figsize'] = (21, 1)
                    if expert_hypnos is None: max_len = len(scoring.last_hypno)
                    else: max_len = max(len(scoring.last_hypno), len(expert_hypnos[fname][eeg_idx][day]))
                    if not (expert_hypnos is None):
                        plt.plot(expert_hypnos[fname][eeg_idx][day])
                        plt.xlim(0, max_len)
                        plt.yticks(sorted(list(scoring.stage_map.keys()))[:3], labels=[scoring.stage_map[k] for k in sorted(list(scoring.stage_map.keys()))[:3]])
                        plt.ylabel('State', fontsize=15)
                        plt.xlabel('Epoch #', fontsize=15)
                        plt.show()

#                     plt.plot(mask1.astype(int), lw=1, c='grey')
#                     plt.xlim(0, len(mask1))
#                     plt.yticks(np.arange(2), ['', 'NREM'])
#                     plt.scatter(np.argwhere(mask1 != 0), np.ones(len(np.where(mask1 != 0)[0])), c=colors[1], s=.7)
#                     plt.ylabel('State', fontsize=15)
#                     plt.xlabel('Epoch #', fontsize=15)
#                     plt.ylim(0, 1.25)
#                     plt.show()

#                     plt.plot(mask.astype(int)*2, lw=1, c='grey')
#                     plt.xlim(0, len(mask))
#                     plt.yticks([0, 2], ['', ' REM'])
#                     plt.scatter(np.argwhere(mask != 0), np.ones(len(np.where(mask != 0)[0]))*2, c=colors[2], s=.7)
#                     plt.ylabel('State', fontsize=15)
#                     plt.xlabel('Epoch #', fontsize=15)
#                     plt.ylim(0, 2.25)
#                     plt.show()
                    
                    ax = plt.subplot(111)
                    ax.plot(scoring.last_hypno, lw=1, c='grey')
                    for stage in stages:
                        dum_stage = np.full(len(scoring.last_hypno), stage)
                        dum_stage[scoring.last_hypno != stage] = None
                        ax.plot(dum_stage, lw=2, c=colors[stage])
                        ax.scatter(np.arange(len(dum_stage)), dum_stage, c=colors[stage], s=.5)
                    ax.set_yticks(sorted(list(scoring.stage_map.keys()))[:len(stages)])
                    ax.set_yticklabels([scoring.stage_map[k] for k in sorted(list(scoring.stage_map.keys()))[:len(stages)]])
                    ax.set_ylabel('State', fontsize=15)
                    ax.set_xlim(0, len(scoring.last_hypno))
                    ax.set_xlabel('Epoch #', fontsize=15)
                    ax.set_ylim(0, 2.25)
                    plt.show()
                    
                # если есть гипнограмма, можно посмотреть точности
                # scoring.compare(os.path.join(dir_, fname[:-3]+'txt'))
                
                # save specs
                saved_specs[fname][eeg_idx].append({})
                av = save_spec_params['window_sec'] // scoring.window_sec1 # по умолчанию ориентируемся на окно анализа дельты
                unit = str(scoring.data.analogsignals[0].units).split()[-1]
                for b, band in enumerate(save_spec_params['ranges']):
                    saved_specs[fname][eeg_idx][-1][band] = {}
                    spec_save = np.sum(full_spec[:, (freqs >= band[0])&(freqs <= band[1])]**2, axis=1)
                    for n in range(ceil(len(scoring.last_hypno)/av)):
                        art_chunk = artifacts[av*n: av*(n+1)]
                        hypno_chunk = scoring.last_hypno[av*n: av*(n+1)]
                        spec_chunk = spec_save[av*n: av*(n+1)]
                        for stage in stages:
                            saved_specs[fname][eeg_idx][-1][band].setdefault(stage, [])
                            # print(np.min(spec_chunk), np.max(spec_chunk), np.unique(hypno_chunk), np.unique(art_chunk))
                            slc = (hypno_chunk == stage) & art_chunk
                            if np.any(slc): saved_specs[fname][eeg_idx][-1][band][stage].append(np.mean(spec_chunk[slc]))
                            else: saved_specs[fname][eeg_idx][-1][band][stage].append(0)
                    plt.rcParams['figure.figsize'] = (21, 3)
                    plt.ylabel(f'Power at {band[0]}-{band[1]} Hz [${unit}^2$/{freqs[1]-freqs[0]:.2f} Hz]')
                    plt.xlabel('Epoch #', fontsize=15)
                    ax = plt.subplot(111)
                    for stage in stages:
                        stage_mask = scoring.last_hypno == stage
                        ax.bar(np.argwhere(stage_mask).flatten(), spec_save[stage_mask], width=5, color=colors[stage])
                        if (stage == 1) and (spec_max[eeg_idx][b] is None): ax.set_ylim(ax.get_ylim()[0], np.max(spec_save[stage_mask])*1.1)
                        else: ax.set_ylim(0, spec_max[eeg_idx][b])
                    ax.bar(np.argwhere(np.logical_not(artifacts)).flatten(), spec_save[np.logical_not(artifacts)], width=5, color='yellow')
                    ax.legend([scoring.stage_map[stage] for stage in stages]+['Artifacts'])
                    # ax.set_ylim(0, np.max(spec_save)*1.1)
                    # print(np.min(spec_save), np.max(spec_save), np.argmax(spec_save), scoring.last_hypno[np.argmax(spec_save)], spec_save[np.argmax(spec_save)])
                    ax.set_xlim(0, len(spec_save))
                    #  ax.axvline(artidx, c='orange', alpha=.5, lw=1)
                    plt.show()
                #
                fracs[fname].extend(scoring.stage_fraction(minutes))
                fragms[fname].extend(scoring.stage_duration(hours_fragm))
                hypnos[fname].append(scoring.last_hypno) 
                if not (expert_hypnos is None):
                    scoring.txthypno = expert_hypnos[fname][eeg_idx][day][:min(len(expert_hypnos[fname][eeg_idx][day]), len(scoring.last_hypno))]
                    scoring.compare(None)
                    scoring.last_hypno = scoring.txthypno
                    exp_fracs[fname].extend(scoring.stage_fraction(minutes))
                    if verbose > 0:
                        dct, dct_exp = fracs[fname][-1], exp_fracs[fname][-1]
                        plt.rcParams['figure.figsize'] = (20, 4)
                        for i, stage in enumerate(sorted(list(dct.keys()))):
                            ax = plt.subplot(100 + len(dct)*10 + i + 1)
                            ax.plot(np.array(dct[stage])*100, label=f'{scoring.stage_map[int(stage)]} auto')
                            ax.plot(np.array(dct_exp[stage])*100, label=f'{scoring.stage_map[int(stage)]} expert')
                            ax.legend(prop=font)
                            ax.set_ylabel('Stage presence, %', fontsize=18)
                            ax.set_xlabel('Hour', fontsize=18)
                            ax.tick_params(axis='both', labelsize=13)
                        plt.tight_layout()
                        plt.show()

    for fname in fnames:
        scoring.stage_fraction_report(os.path.join(save_dir, fname[:-4]+'_fracs.csv'), minutes, fracs[fname], col_names=[f'_an{an}_d{d}' for an in vote_dict[fname] for d in range(len(vote_dict[fname][an]))]) # можно указать папку в начале имени файла вместо ./
        scoring.stage_duration_report(os.path.join(save_dir, fname[:-4]+'_fragms.csv'), fragms[fname])
        pd.concat([pd.DataFrame({'hypno'+str(i): el}) for i, el in enumerate(hypnos[fname])], axis=1).to_csv(os.path.join(save_dir, fname[:-4]+'_hypnos.csv'), index=False)
        # specs
        df = {}
        for eeg_idx in sorted(list(saved_specs[fname])): # str
            for day in range(len(saved_specs[fname][eeg_idx])): # int
                for band in sorted(list(saved_specs[fname][eeg_idx][day])): # tuple
                    for stage in sorted(list(saved_specs[fname][eeg_idx][day][band])): # int
                        df[f'An#{eeg_idx}_d{day}_{band[0]}-{band[1]}Hz_{scoring.stage_map[stage]}'] = saved_specs[fname][eeg_idx][day][band][stage]
        pd.DataFrame(df).to_csv(os.path.join(save_dir, fname[:-4]+'_specs.csv'), index=False)
    return to_interactive

def points_in_hull(p, hull, tol=1e-12):
    return np.all(hull.equations[:,:-1] @ p.T + np.repeat(hull.equations[:,-1][None,:], len(p), axis=0).T <= tol, 0)


def in_ellipse(x, y, g_ell_center, g_ell_width, g_ell_height, angle):
    cos_angle = np.cos(np.radians(180.-angle))
    sin_angle = np.sin(np.radians(180.-angle))
    xc = x - g_ell_center[0]
    yc = y - g_ell_center[1]
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 
    rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)
    return np.argwhere(rad_cc <= 1).flatten()

def make_ellipses(gmm, ax, ellipse_magnify=1):
    import matplotlib as mpl
    n=0
    covariances = gmm.covariances_[n][:2, :2]
    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v) * ellipse_magnify #
    ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color='brown')
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(.5)
    ax.scatter(*gmm.means_[n, :2], marker='*', color='brown')
    ax.add_artist(ell)
    # ax.set_aspect("equal", "datalim")
    return gmm.means_[n, :2], v[0], v[1], angle
        
        
def sel_scatter(df, manual=True, ellipse_magnify=1, selection=None):
    ###
    if manual:
        points = df.hvplot.scatter(x="Theta", y="~Delta", c="state", width=700, height=500, cmap='plasma', colorbar=False).opts(tools=["lasso_select", "box_select"], active_tools=["lasso_select"])
        sel = hv.streams.Selection1D(source=points)
        if not (selection is None):
            remarr = df[['Theta', '~Delta']].values[selection.index]
            coords = ConvexHull(remarr)
            xs, ys = [], []
            for simplex in coords.vertices:
                # print(simplex)
                xs.append(remarr[simplex, 0])
                ys.append(remarr[simplex, 1])
            # print(xs)
            # print(ys)
            points = points * hv.Polygons([{'x': xs, 'y': ys}]).opts(alpha=.5, color='r')
    else:
        X = df[['Theta', '~Delta']].values
        y = df['state'].values
        
        # plt.figure(figsize=(6, 6))
        # ax = plt.subplot(111)
        
        ## Naive gaussian 3-class Bayes
        # model = GaussianNB(priors=[np.where(y == c)[0].size/len(y) for c in np.unique(y)]).fit(X, y)
        # from sklearn.inspection import DecisionBoundaryDisplay
        # DecisionBoundaryDisplay.from_estimator(
        #         model, X, cmap='plasma', alpha=0.5, ax=ax, eps=0.5
        #     )
        # sel = np.argwhere(model.predict(X) == 2).flatten()
        
        ## REM Gauss 1D
        model = GM(n_components=1, covariance_type='full', tol=1e-6).fit(X[y==2])
        
        # ax.scatter(*X.T, c=y, alpha=.6)
        # ell_params = make_ellipses(model, ax, ellipse_magnify) # None
        # sel = in_ellipse(X[:, 0], X[:, 1], *ell_params)
        # ax.set_ylim(60, 180)
        # ax.set_xlim(100, 250)
        
        # sel = np.argwhere(model.predict_proba(X) >= .5).flatten() 
                
        # plt.show()
        
        
        hypno_ = y#expert_hypnos[fname][eeg_idx][day][:len(nr_mask)][nr_mask]
        stage_map = {0: 'Wake', 1: 'NREM', 2: 'REM'}
        colors = {'Wake': 'darkcyan', 'NREM': 'purple', 'REM': 'gold'}
        fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.7), sharey=True)
        fig.text(0.5, -.01, 'Average theta amplitude, μV', ha='center', fontsize=18, family='Cambria')
        ax = axes.flatten()[0]
        for c in range(3):
            ax.scatter(X[hypno_ == c, 0], X[hypno_ == c, 1], alpha=.5, s=10, label=stage_map[c], c=colors[stage_map[c]])                
        ax.legend(fontsize=14, handletextpad=0, borderpad=.5)
        # ax.set_ylim(140, 600)
        # ax.set_xlim(90, 360)
        ax.set_ylabel('Average delta amplitude, μV', fontsize=18, family='Cambria')
        ax.tick_params(axis='both', labelsize=14)
        # ax.text(400, 1470, 'A', fontsize=24)

        ax = axes.flatten()[1]
        colors2 = {1: 'purple', 0: 'grey'}
        stage_map2 = {1: 'NREM', 0: 'Wake+REM'}
        for c in range(3):
            ax.scatter(X[hypno_ == c, 0], X[hypno_ == c, 1], alpha=.5, s=10, label=stage_map[c], c=colors[stage_map[c]])  
        ax.legend(fontsize=14, handletextpad=0, borderpad=.5)
        ell_params = make_ellipses(model, ax, ellipse_magnify) # None
        sel = in_ellipse(X[:, 0], X[:, 1], *ell_params)
        # ax.set_ylim(140, 600)
        # ax.set_xlim(90, 360)
        ax.tick_params(axis='both', labelsize=14)
        # ax.text(400, 1470, 'B', fontsize=24)

        plt.tight_layout()
        # plt.savefig(r'G:\LA\2024\qual_figs\ellipse_before_after.png', dpi=300, bbox_inches='tight')
        plt.show()
        
#         plt.figure(figsize=(6, 6))
#         ax = plt.subplot(111)
#         elcol = np.zeros(len(X))
#         elcol[sel] = 1
#         ax.scatter(*X.T, c=elcol, alpha=.6)
#         plt.show()
        
        points = None
    return points, sel

def correct_selection(selection, scoring):
    nrids = np.argwhere(scoring.nr_mask).flatten()
    while selection.index[-1] >= len(nrids):
        del selection.index[-1]
    return selection

REM_MASK = []

def end_scoring_selection(vote_dict, fname, eeg_idx, day, fnames, scoring, rule, cluster_artifact, n_back, red_mask, expert_hypnos, mask_option, single_nr, single_wake, single_rem1_2_0, single_rem1_2_1, n_hours_cycle_rem, verbose, save_dir='./', spindles=None, ep=[None]*8, manual_contour=True, vote_pass=slice(None, None), ellipse_magnify=1, selection=None):
    global REM_MASK
    hypnos = {}
    # # print('\n\nFile \"'+fname+'\"')
    # LOG.info('File \"'+fname+'\"')
    hypnos[fname] = []
    print('Animal '+str(eeg_idx))
    # LOG.info('Animal '+str(eeg_idx))
    n_days = len(vote_dict[fname][eeg_idx])
    print('Day '+str(day))
    # LOG.info('Day '+str(day))
    # if not (selection is None):
    #     selection = correct_selection(selection, scoring)
    # ГОЛОСОВАНИЕ
    to_vote, failed, mask1, art_mask, red_to_interactive = vote_dict[fname][eeg_idx][day][:5]
    to_vote = [to_vote[i] for i in vote_pass]
    print(f'len(to_vote) = {len(to_vote)}')
    to_vote_ = [mset[mask_option[eeg_idx][optn]] for optn, mset in enumerate(to_vote)]
    turn_on = np.argwhere(np.array(mask_option[eeg_idx]).flatten() != -1).flatten()
    if len(turn_on) < len(to_vote_):
        to_vote = []
        for stay_idx in turn_on:
            to_vote.append(to_vote_[stay_idx])
    else: to_vote = to_vote_
    print(f"voting using {len(to_vote)} clusters. rule = {rule[eeg_idx]-failed}")
    # LOG.info(f"voting using {len(to_vote)} clusters. rule = {rule[eeg_idx]-failed}")
    mask = rem_vote(to_vote, rule=rule[eeg_idx]-failed) # ПРАВИЛО ГОЛОСОВАНИЯ rule МОЖЕТ БЫТЬ ЦЕЛЫМ ЧИСЛОМ, КОТОРОЕ БУДЕТ ОЗНАЧАТЬ МИНИМАЛЬНОЕ КОЛИЧЕСТВО ГОЛОСОВ ДЛЯ ПРИНЯТИЯ РЕШЕНИЯ REM
    REM_MASK = mask
    # далее используется экземпляр scoring последнего дня последнего животного в последнем файле. что-то может быть некорректным
    if red_mask:   
        scoring.last_hypno = combine_masks(mask, mask1, int(n_hours_cycle_rem*3600/scoring.window_sec2), art_mask)
    else:
        # print(mask.shape, mask1.shape, art_mask.shape)
        scoring.last_hypno = combine_masks(mask, mask1, int(n_hours_cycle_rem*3600/scoring.window_sec2), art_mask)

    # Epanechnikov
    
# cut
#####
            
        # for paper
#         plt.figure(figsize=(8.3, 7))
#         ax = plt.subplot(2, 2, 2)
#         ax.hist(dens_pred, 50, color='grey')
#         ax.set_xlabel('Log-likelihood\nprobability density', fontsize=18, family='Cambria')
#         ax.set_ylabel('Count', fontsize=18, family='Cambria')
#         ax.axvline(thr_ca, c='k')
#         ax.tick_params(axis='both', labelsize=14)
#         ax.text(-8.5, 150, 'B', fontsize=24)
#         ax.set_ylim(0, 300)

#         ax = plt.subplot(2, 2, 1)
#         ax.set_xlabel('Raw signal RMS', fontsize=18, family='Cambria')
#         ax.set_ylabel('Theta (2-10 Hz)\namplitude, μV', fontsize=18, family='Cambria')
#         colors = {False: 'g', True: 'r'}
#         labels = {False: 'Normal', True: 'Outlier'}
#         for c in (True, False):
#             ax.scatter(*sample[(dens_pred < thr_ca) == c].T, c=colors[c], s=5, alpha=.5, label=labels[c])
#         ax.tick_params(axis='both', labelsize=14)
#         ax.legend(fontsize=14, handletextpad=0, borderpad=.5, prop={'family': 'Cambria', 'size': 14})   
#         ax.text(-2, 2.5, 'A', fontsize=24)
        
        
        
#         ax = plt.subplot(2, 1, 2)
#         ax.plot(art_rms, lw=.3, c=colors[False], label=labels[False])
#         ax.plot(to_plot, lw=1.2, c=colors[True], label=labels[True])
#         ax.legend(fontsize=14, handletextpad=0.5, borderpad=.5, prop={'family': 'Cambria', 'size': 14})
#         ax.set_ylabel('Raw signal RMS', fontsize=18, family='Cambria')
#         ax.set_xlabel('Epoch #', fontsize=18, family='Cambria')
#         ax.tick_params(axis='both', labelsize=14)
#         ax.text(250, .09, 'C', fontsize=24)
#         ax.set_xlim(0, len(to_plot))
#         # ax.set_ylim(.1, .27)
#         plt.tight_layout()
#         plt.savefig(r'G:\LA\2024\qual_figs\epan_result.png', dpi=300, bbox_inches='tight')
#         plt.show()
            
        # _mask = np.array(np.isnan(to_plot))
        # scoring.last_hypno = cluster_art(scoring.last_hypno, _mask)

    # Контекстуальные правила
    if single_nr: 
        nr_before = np.argwhere(np.array(scoring.last_hypno) == 1).size
        scoring.last_hypno = no_singles(scoring.last_hypno, 1, single_nr) # 0-1-0 -> 0-0-0; 2-1-2 -> 2-2-2
        print(f'{nr_before-np.argwhere(np.array(scoring.last_hypno) == 1).size} {int(single_nr)}-long or shorter NREM epochs corrected')
    if single_wake:
        w_before = np.argwhere(np.array(scoring.last_hypno) == 0).size
        scoring.last_hypno = no_singles(scoring.last_hypno, 0, single_wake) # 1-0-1 -> 1-1-1; 2-0-2 -> 2-2-2
        print(f'{w_before-np.argwhere(np.array(scoring.last_hypno) == 0).size} {int(single_wake)}-long or shorter Wake epochs corrected')
    # if single_rem1_2_0: 
    #     scoring.no_single_a_between_b_and_c(2, 1, 0) #  1-2-0 -> 1-1-0
    # if single_rem1_2_1: scoring.no_single_a_between_b_and_c(2, 1, 1) # 1-2-1 -> 1-1-1
    
    # spindles
    if spindles:
        scoring.last_hypno = spindle_check(scoring.last_hypno, full_spec, freqs)
    stages = np.unique(scoring.last_hypno)
    colors = {0: 'g', 1: 'b', 2: 'r', 4: 'purple'}

    ## red circle
    nr_mask = vote_dict[fname][eeg_idx][day][5]
    hypno_labels = np.array(scoring.last_hypno)[nr_mask].reshape((-1, 1))
    for stcode, stname in zip((0, 1, 2), ('Wake', 'NREM', 'REM')):
        print(f'{stname} - {np.argwhere(hypno_labels.flatten() == stcode).size}; ', end='')
    print()
#     hypno_labels = hypno_labels.flatten()
    av_clust = np.hstack((np.vstack([el[0].reshape((1, -1)) for el in TO_AVE[fname][eeg_idx][day]]).mean(axis=0)[:, None], np.vstack([el[1].reshape((1, -1)) for el in TO_AVE[fname][eeg_idx][day]]).mean(axis=0)[:, None]))

    df = pd.DataFrame(np.hstack((av_clust, hypno_labels)), columns=('Theta', '~Delta', 'state'))
    return sel_scatter(df, manual_contour, ellipse_magnify, selection), hypno_labels, scoring.last_hypno

def val_ridges(scoring, hypno, exp_hypno=None):
    fig = plt.figure(figsize=(25, 5))
    # print()
    # minlen = min(len(scoring.sm_art_rms), len(scoring.art_mask))
    # to_plot = scoring.sm_art_rms[:minlen]
    # art_epan_mask = scoring.epan_mask[scoring.first_art_mask]
    # to_plot = scoring.sm_art_rms[art_epan_mask] #[scoring.art_mask]
    to_plot = np.full(len(scoring.art_rms), np.nan)
    to_plot[scoring.first_art_mask] = scoring.sm_art_rms
    plt.plot(to_plot[scoring.art_mask], c='grey', ls='--') #art_mask
    plt.title('Валидация вырезания горбов')
    plt.xlabel('Epoch #')
    plt.ylabel('Amplitude, a.u.')
    chunksize = 1 / len(scoring.ridge_thrs)
    for i, rthr in enumerate(scoring.ridge_thrs):
        plt.axhline(rthr, xmin=i * chunksize, xmax=(i+1)*chunksize, c='k')
    # print(len(scoring.sm_art_rms), len(hypno))
    minlen = min(len(scoring.art_rms), len(hypno)) # sm_
    # print(minlen, to_plot.shape, hypno.shape)
    
    rem_dumm = np.array(to_plot)[:minlen] #scoring.sm_art_rms
    rem_dumm = rem_dumm[scoring.art_mask[:minlen]]
    
    hypno = hypno[:minlen]
    hypno = hypno[scoring.art_mask[:minlen]]
    
    nanmask = hypno != 2
    rem_dumm[nanmask] = np.nan
    plt.scatter(np.arange(len(rem_dumm)), rem_dumm, c='gold', alpha=1, lw=2, edgecolor='gold', s=65) #rem_dumm
    
    not_rem = np.array(to_plot)[:minlen] #scoring.sm_art_rms)[:minlen]
    not_rem = not_rem[scoring.art_mask[:minlen]]
    art_nr_mask = np.array(scoring.nr_mask[:minlen])[scoring.art_mask[:minlen]]
    not_rem[art_nr_mask] = np.nan #scoring.nr_mask
    plt.scatter(np.arange(len(not_rem)), not_rem, c='darkred', alpha=1, s=10)
    
    if not (exp_hypno is None):
        exp_hypno = exp_hypno[scoring.art_mask[:minlen]]
        lh = scoring.last_hypno[scoring.art_mask[:minlen]]
        x_fp_rem = np.argwhere((exp_hypno != 2) & (lh == 2)).flatten()
        to_plot_ = to_plot[scoring.art_mask[:minlen]]
        plt.scatter(x_fp_rem, to_plot_[x_fp_rem], marker='x', c='b', s=25)
        
        x_fn_rem = np.argwhere((exp_hypno == 2) & (lh != 2)).flatten()
        plt.scatter(x_fn_rem, to_plot_[x_fn_rem], marker='X', c='r', s=35)
    plt.xlim(0, len(to_plot)) #scoring.sm_art_rms
    plt.show()
    return rem_dumm, not_rem

FIGSAVE = OD()
def end_scoring_final(vote_dict, fnames, scoring, n_hours_cycle_bars, minutes, hours_fragm, red_mask, n_hours_cycle_rem, expert_hypnos, verbose, hypno_labels_dct, last_hypnos, n_back, wake_thr, single_nrem2_1_2, single_wake2_0_2, single_r, n_back_old, save_spec_params={}, spindles=None, spec_max = (), get_particular=False, save_dir='./', clear_start_rem=False, cut_left=True, rnr_edges=False):
    # %matplotlib inline
    
    to_interactive = {}
    hypnos = {}
    fragms = {}
    fracs, exp_fracs = {}, {}
    saved_specs = {}
    for fn, fname in enumerate([get_particular[0]] if get_particular else fnames):
        # print('\n\nFile \"'+fname+'\"')
       # LOG.info('File \"'+fname+'\"')
        to_interactive[fname] = {}
        fracs[fname] = []
        exp_fracs[fname] = []
        fragms[fname] = []
        hypnos[fname] = []
        saved_specs[fname] = {}
        for eeg_idx in ([get_particular[1]] if get_particular else vote_dict[fname]):
            # print('Animal '+str(eeg_idx))
            n_days = len(vote_dict[fname][eeg_idx])
            to_interactive[fname][eeg_idx] = []
            saved_specs[fname][eeg_idx] = []
            for day in ([get_particular[2]] if get_particular else range(n_days)):
                # print('Day '+str(day))
                nr_mask = vote_dict[fname][eeg_idx][day][5]
                #########################################
                hypno_labels = hypno_labels_dct[fname][eeg_idx][day].flatten()
                scoring.last_hypno = np.array(last_hypnos[fname][eeg_idx][day])
                scoring.last_hypno[nr_mask] = hypno_labels
                
                if not (expert_hypnos is None): 
                    val_ridges(scoring, scoring.last_hypno, expert_hypnos[fname][eeg_idx][day][:len(nr_mask)])
                    print('Before contextual rules:')
                    scoring.txthypno = expert_hypnos[fname][eeg_idx][day][:min(len(expert_hypnos[fname][eeg_idx][day]), len(scoring.last_hypno))]
                    scoring.compare(None)
                else: val_ridges(scoring, scoring.last_hypno)
                
                
                # вычищаем сомнительный РЕМ в самом начале (20 эпох) 
                if clear_start_rem:
                    lh_backup = np.array(scoring.last_hypno)
                    scoring.last_hypno = scoring.last_hypno[:20]
                    rems_20 = np.argwhere(scoring.last_hypno == 2).size
                    scoring.no_rems_in_wake(1, 0)
                    print(f'{rems_20-np.argwhere(scoring.last_hypno == 2).size} REMs turned to Wake in the very beginning')
                    lh_backup[:20] = scoring.last_hypno
                    scoring.last_hypno = lh_backup
                
                rems_total = np.argwhere(scoring.last_hypno == 2).size
                
                # сложный n_bcak (при наличии wake_thr эпох Wake среди n_back эпох перед REM отменяем этот REM)
                scoring.no_rems_in_wake(n_back, wake_thr) # 
                print(f'{rems_total-np.argwhere(scoring.last_hypno == 2).size} REMs turned to Wake by analyzing preceding states')
                
                # удаляем фрагментацию REM единичными эпохами других состояний
                if single_nrem2_1_2: 
                    scoring.no_single_a_between_b_and_c(1, 2, 2) #  
                if single_wake2_0_2: 
                    scoring.no_single_a_between_b_and_c(0, 2, 2)
                
                # удаляем единичный (или длинее) REM
                if single_r: 
                    r_before = np.argwhere(scoring.last_hypno == 2).size
                    scoring.last_hypno = np.array(no_singles(scoring.last_hypno, 2, single_r))
                    print(f'{r_before-np.argwhere(np.array(scoring.last_hypno) == 2).size} {int(single_r)}-long or shorter REM epochs corrected')
                
                # удаляем REM, если перед ним как минимум n_back_old эпох Wake
                if n_back_old: 
                    r_before = np.argwhere(scoring.last_hypno == 2).size
                    scoring.no_rems_in_wake_old(n_back_old)
                    print(f'{r_before-np.argwhere(np.array(scoring.last_hypno) == 2).size} REM epochs removed due to at least {int(n_back_old)} preceding Wake epochs (old n_back)')
                
                hypno_labels = scoring.last_hypno[nr_mask]
                
                # усредненный с ошибками
                av_clust = np.hstack((np.vstack([el[0].reshape((1, -1)) for el in TO_AVE[fname][eeg_idx][day]]).mean(axis=0)[:, None], np.vstack([el[1].reshape((1, -1)) for el in TO_AVE[fname][eeg_idx][day]]).mean(axis=0)[:, None]))
                # plt.rcParams['figure.figsize'] = (7, 7)
                fig = plt.figure(figsize=(7, 7))
                # plt.scatter(*av_clust.T, c=hypno_labels, alpha=.5)
                ##
                hypno2rems = np.zeros(len(scoring.nr_mask))
                hypno2rems[scoring.nr_mask] = hypno_labels
                hypno2rems[(hypno2rems == 2) & (np.arange(len(hypno2rems)) > (len(hypno2rems)//2))] = -2
                hypno2rems = hypno2rems[scoring.nr_mask]
                if isinstance(rnr_edges, int):
                    edgemask = np.zeros(len(hypno2rems))
                    curidx = 0 
                    for code, lst in groupby(hypno2rems):
                        lst = list(lst)
                        if code == 1:
                            edgemask[max(0, curidx+len(lst)-rnr_edges):curidx+len(lst)] = 1
                        elif abs(code) == 2:
                            edgemask[curidx: curidx+rnr_edges] = 1
                        curidx += len(lst)
                    colors = {0: 'purple', 1: 'darkcyan', 2: 'yellow', -2: 'goldenrod'}
                    for code in np.unique(hypno2rems):
                        plt.scatter(*av_clust[hypno2rems == code].T, c=colors[code], alpha=.5)
                    for code in np.unique(hypno2rems):
                        plt.scatter(*av_clust[(hypno2rems == code) & (edgemask == 1)].T, facecolors="none", edgecolors='k')
                ##
                plt.xlabel('Av. theta')
                plt.ylabel('Av. delta')
                
                if not (expert_hypnos is None):
                    nr_exp_hypn = expert_hypnos[fname][eeg_idx][day][:len(nr_mask)]
                    nr_exp_hypn = nr_exp_hypn[nr_mask]
                    fp_rem = (hypno_labels == 2) & (nr_exp_hypn != 2)
                    fn_rem = (hypno_labels != 2) & (nr_exp_hypn == 2)
                
                    plt.scatter(*av_clust[fp_rem].T, c='b', marker='x')
                    plt.scatter(*av_clust[fn_rem].T, c='r', marker='x')
                plt.ylim(np.percentile(av_clust[:, 1], 0), np.percentile(av_clust[:, 1], 99))
                plt.xlim(np.percentile(av_clust[:, 0], 1), np.percentile(av_clust[:, 0], 99.5))
                # plt.savefig(os.path.join(save_dir, f'{fname[:-4]}_an{eeg_idx}_d{day}_clustering.png'), dpi=300, bbox_inches='tight')
                tmpfile = BytesIO()
                fig.savefig(tmpfile, format='png', bbox_inches='tight')
                FIGSAVE[f'{fname[:-4]}_an{eeg_idx}_d{day}_clustering'] = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                plt.show()
                ##
                
                red_to_interactive = vote_dict[fname][eeg_idx][day][4]
                to_interactive[fname][eeg_idx].append((scoring.last_hypno, red_to_interactive))
                if verbose > 0: 
                    cyclefigs = scoring.bars_by_cycles(n_hours_cycle=n_hours_cycle_bars, save_dir=os.path.join(save_dir, f'{fname[:-4]}_an{eeg_idx}_d{day}_fractions.png')) # соотношение стадий в циклах заданной длины
                    for cclfg in range(len(cyclefigs)): FIGSAVE[f'{fname[:-4]}_an{eeg_idx}_d{day}_cycle#{cclfg+1}'] = cyclefigs[cclfg]
                if verbose > 0: 
                    # plt.rcParams['figure.figsize'] = (21, 1)
                    
                    if expert_hypnos is None: max_len = len(scoring.last_hypno)
                    else: max_len = max(len(scoring.last_hypno), len(expert_hypnos[fname][eeg_idx][day]))
                    if not (expert_hypnos is None):
                        fig = plt.figure(figsize=(21, 1))
                        plt.plot(expert_hypnos[fname][eeg_idx][day], lw=1)
                        plt.xlim(0, max_len)
                        plt.yticks(sorted(list(scoring.stage_map.keys()))[:3], labels=[scoring.stage_map[k] for k in sorted(list(scoring.stage_map.keys()))[:3]])
                        plt.ylabel('State', fontsize=15)
                        plt.xlabel('Epoch #', fontsize=15)
                        # plt.savefig(os.path.join(save_dir, f'{fname[:-4]}_an{eeg_idx}_d{day}_expert_hypnogram.png'), dpi=300, bbox_inches='tight')
                        tmpfile = BytesIO()
                        fig.savefig(tmpfile, format='png', bbox_inches='tight')
                        FIGSAVE[f'{fname[:-4]}_an{eeg_idx}_d{day}_expert_hypnogram'] = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                        plt.show()
                    
                    fig = plt.figure(figsize=(21, 1))
                    ax = plt.subplot(111)
                    ax.plot(scoring.last_hypno, lw=1, c='grey')
                    stages = np.unique(scoring.last_hypno)
                    colors = {0: 'darkcyan', 1: 'purple', 2: 'goldenrod', 4: 'purple'}
                    # for stage in stages:
                    #     dum_stage = np.full(len(scoring.last_hypno), stage)
                    #     dum_stage[scoring.last_hypno != stage] = None
                    #     ax.plot(dum_stage, lw=2, c=colors[stage]) #
                        # ax.scatter(np.arange(len(dum_stage)), dum_stage, c=colors[stage], s=.5)
                    ax.set_yticks(sorted(list(scoring.stage_map.keys()))[:len(stages)])
                    ax.set_yticklabels([scoring.stage_map[k] for k in sorted(list(scoring.stage_map.keys()))[:len(stages)]])
                    ax.set_ylabel('State', fontsize=15)
                    ax.set_xlim(0, len(scoring.last_hypno))
                    ax.set_xlabel('Epoch #', fontsize=15)
                    ax.set_ylim(-.1, 2.25)
                    # plt.savefig(os.path.join(save_dir, f'{fname[:-4]}_an{eeg_idx}_d{day}_hypnogram.png'), dpi=300, bbox_inches='tight')
                    tmpfile = BytesIO()
                    fig.savefig(tmpfile, format='png', bbox_inches='tight')
                    FIGSAVE[f'{fname[:-4]}_an{eeg_idx}_d{day}_hypnogram'] = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                    plt.show()
                    
                # если есть гипнограмма, можно посмотреть точности
                # scoring.compare(os.path.join(dir_, fname[:-3]+'txt'))
                
                # save specs
                saved_specs[fname][eeg_idx].append({})
                av = save_spec_params['window_sec'] // scoring.window_sec1 # по умолчанию ориентируемся на окно анализа дельты
                unit = str(scoring.data.analogsignals[0].units).split()[-1]
                for b, band in enumerate(save_spec_params['ranges']):
                    saved_specs[fname][eeg_idx][-1][band] = {}
                    full_spec, freqs = vote_dict[fname][eeg_idx][day][7]
                    spec_save = np.sum(full_spec[:, (freqs >= band[0])&(freqs <= band[1])]**2, axis=1)
                    artifacts = scoring.art_mask
                    for n in range(ceil(len(scoring.last_hypno)/av)):
                        art_chunk = artifacts[av*n: av*(n+1)]
                        hypno_chunk = scoring.last_hypno[av*n: av*(n+1)]
                        spec_chunk = spec_save[av*n: av*(n+1)]
                        for stage in stages:
                            saved_specs[fname][eeg_idx][-1][band].setdefault(stage, [])
                            # print(np.min(spec_chunk), np.max(spec_chunk), np.unique(hypno_chunk), np.unique(art_chunk))
                            slc = (hypno_chunk == stage) & art_chunk
                            if np.any(slc): saved_specs[fname][eeg_idx][-1][band][stage].append(np.mean(spec_chunk[slc]))
                            else: saved_specs[fname][eeg_idx][-1][band][stage].append(0)
                    fig = plt.figure(figsize=(21, 3))
                    # plt.rcParams['figure.figsize'] = (21, 3)
                    colors = {0: 'darkcyan', 1: 'purple', 2: 'r', 4: 'purple'}
                    ax = plt.subplot(111)
                    ax.set_ylabel(f'Power at {band[0]}-{band[1]} Hz [${unit}^2$/{freqs[1]-freqs[0]:.2f} Hz]')
                    ax.set_xlabel('Epoch #', fontsize=15)
                    for stage in stages:
                        stage_mask = scoring.last_hypno == stage
                        ax.bar(np.argwhere(stage_mask).flatten(), spec_save[stage_mask], width=5, color=colors[stage])
                        if (stage == 1) and (spec_max[eeg_idx][b] is None): ax.set_ylim(ax.get_ylim()[0], np.max(spec_save[stage_mask])*1.1)
                        else: ax.set_ylim(0, spec_max[eeg_idx][b])
                    ax.bar(np.argwhere(np.logical_not(artifacts)).flatten(), spec_save[np.logical_not(artifacts)], width=5, color='yellow')
                    ax.legend([scoring.stage_map[stage] for stage in stages]+['Artifacts'])
                    # ax.set_ylim(0, np.max(spec_save)*1.1)
                    # print(np.min(spec_save), np.max(spec_save), np.argmax(spec_save), scoring.last_hypno[np.argmax(spec_save)], spec_save[np.argmax(spec_save)])
                    ax.set_xlim(0, len(spec_save))
                    #  ax.axvline(artidx, c='orange', alpha=.5, lw=1)
                        # plt.savefig(os.path.join(save_dir, f'{fname[:-4]}_an{eeg_idx}_d{day}_{band}power.png'), dpi=300, bbox_inches='tight')
                    tmpfile = BytesIO()
                    fig.savefig(tmpfile, format='png', bbox_inches='tight')
                    FIGSAVE[f'{fname[:-4]}_an{eeg_idx}_d{day}_{band}power.png'] = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                    plt.show()
                
                fracs[fname].extend(scoring.stage_fraction(minutes))
                fragms[fname].extend(scoring.stage_duration(hours_fragm))
                hypnos[fname].append(scoring.last_hypno) 
                
                if not (expert_hypnos is None): found_rem, ridge_removed = val_ridges(scoring, scoring.last_hypno, expert_hypnos[fname][eeg_idx][day][:len(nr_mask)])
                else: found_rem, ridge_removed = val_ridges(scoring, scoring.last_hypno)
                refined_ridges_mask_ = np.zeros(len(found_rem))
                refined_ridges_mask_[~np.isnan(found_rem)] = 1
                refined_ridges_mask_[~np.isnan(ridge_removed)] = 2
                refined_ridges_mask = np.zeros(len(scoring.last_hypno))
                refined_ridges_mask[scoring.art_mask] = refined_ridges_mask_
                
                refined_ridges = np.full(len(refined_ridges_mask), True)
                refined_ridges[refined_ridges_mask == 2] = False
                seqs012 = [(el[0], len(list(el[1]))) for el in groupby(refined_ridges_mask)]
                # print(found_rem)
                # print(ridge_removed)
                # print(seqs012)
                curidx = seqs012[0][1]
                for i in range(1, len(seqs012)-1):
                    code, lenlst = seqs012[i]
                    if cut_left:
                        if (code == 0) and all([el in (1, 2) for el in (seqs012[i-1][0], seqs012[i+1][0])]) and (seqs012[i-1][0] != seqs012[i+1][0]):
                            refined_ridges[curidx: curidx+lenlst] = False
                    else:
                        if (code == 0) and (seqs012[i-1][0] == 1) and (seqs012[i+1][0] == 2):
                            refined_ridges[curidx: curidx+lenlst] = False
                    curidx += lenlst
                print(f'{np.argwhere(~refined_ridges).size-np.argwhere(refined_ridges_mask == 2).size} more points will be removed between detected REM and ridge-removed points')
                
                if not (expert_hypnos is None):
                    scoring.txthypno = expert_hypnos[fname][eeg_idx][day][:min(len(expert_hypnos[fname][eeg_idx][day]), len(scoring.last_hypno))]
                    scoring.compare(None)
                    scoring.last_hypno = scoring.txthypno
                    exp_fracs[fname].extend(scoring.stage_fraction(minutes))
                    if verbose > 0:
                        dct, dct_exp = fracs[fname][-1], exp_fracs[fname][-1]
                        # plt.rcParams['figure.figsize'] = (20, 4)
                        fig, axes = plt.subplots(3, 1, figsize=(21, 7), sharex=True)
                        for i, stage in enumerate(sorted(list(dct.keys()))):
                            ax = axes.flatten()[i] #plt.subplot(100 + len(dct)*10 + i + 1)
                            ax.plot(np.array(dct[stage])*100, label=f'{scoring.stage_map[int(stage)]} auto')
                            ax.plot(np.array(dct_exp[stage])*100, label=f'{scoring.stage_map[int(stage)]} expert')
                            ax.legend(fontsize=14, handletextpad=.5, borderpad=.5) #fontname='Cambria', 
                            ax.set_ylabel('Stage presence, %', fontsize=18)
                            ax.tick_params(axis='both', labelsize=14)
                        ax.set_xlim(0, len(dct[stage])-1)
                        ax.set_xlabel('Hour', fontsize=18)
                        plt.tight_layout()
                        # plt.savefig(os.path.join(save_dir, f'{fname[:-4]}_an{eeg_idx}_d{day}_profiles.png'), dpi=300, bbox_inches='tight')
                        tmpfile = BytesIO()
                        fig.savefig(tmpfile, format='png', bbox_inches='tight')
                        FIGSAVE[f'{fname[:-4]}_an{eeg_idx}_d{day}_profiles.png'] = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                        
                        plt.show()
                
                # fig = plt.figure(figsize=(25, 5))
                # plt.plot(scoring.sm_art_rms, c='grey', ls='--')
                # plt.title('Валидация вырезания горбов')
                # plt.xlabel('Epoch #')
                # plt.ylabel('Amplitude, a.u.')
                # chunksize = 1 / len(scoring.ridge_thrs)
                # for i, rthr in enumerate(scoring.ridge_thrs):
                #     plt.axhline(rthr, xmin=i * chunksize, xmax=(i+1)*chunksize, c='k')
                # minlen = min(len(scoring.sm_art_rms), len(scoring.last_hypno))
                # rem_dumm = np.array(scoring.sm_art_rms)[:minlen]
                # rem_dumm[scoring.last_hypno[:minlen] != 2] = np.nan
                # plt.plot(rem_dumm, c='gold', alpha=1, lw=2)
                # plt.xlim(0, len(scoring.sm_art_rms))
                # plt.show()

    # for fname in ([get_particular[0]] if get_particular else fnames):
    #     fnsuff = f'an{get_particular[1] if get_particular else "All"}_d{get_particular[2] if get_particular else "All"}'
    #     scoring.stage_fraction_report(os.path.join(save_dir, fname[:-4]+f'_{fnsuff}_fracs.csv'), minutes, fracs[fname], col_names=[f'_an{an}_d{d}' for an in vote_dict[fname] for d in range(len(vote_dict[fname][an]))]) # можно указать папку в начале имени файла вместо ./
    #     scoring.stage_duration_report(os.path.join(save_dir, fname[:-4]+f'_{fnsuff}_fragms.csv'), fragms[fname])
    #     pd.concat([pd.DataFrame({'hypno'+str(i): el}) for i, el in enumerate(hypnos[fname])], axis=1).to_csv(os.path.join(save_dir, fname[:-4]+f'_{fnsuff}_hypnos.csv'), index=False)
    #     # specs
    #     df = {}
    #     for eeg_idx in sorted(list(saved_specs[fname])): # str
    #         for day in range(len(saved_specs[fname][eeg_idx])): # int
    #             for band in sorted(list(saved_specs[fname][eeg_idx][day])): # tuple
    #                 for stage in sorted(list(saved_specs[fname][eeg_idx][day][band])): # int
    #                     df[f'An#{eeg_idx}_d{day}_{band[0]}-{band[1]}Hz_{scoring.stage_map[stage]}'] = saved_specs[fname][eeg_idx][day][band][stage]
    #     if len(df) > 0: pd.DataFrame(df).to_csv(os.path.join(save_dir, fname[:-4]+f'_{fnsuff}_specs.csv'), index=False)
    return to_interactive, fracs, fragms, hypnos, saved_specs, refined_ridges

def arange_dicts(fracs, fragms, hypnos, saved_specs):
    resfrac, resfragm, reshypn, ressasp, res_ids = {}, {}, {}, {}, {}
    for fname in fracs:
        resfrac[fname], resfragm[fname], reshypn[fname], ressasp[fname] = [], [], [], {}
        res_ids[fname] = {}
        for eeg_idx in sorted(list(fracs[fname].keys())):
            ressasp[fname][eeg_idx] = []
            res_ids[fname][eeg_idx] = 0
            for day in sorted(list(fracs[fname][eeg_idx].keys())):
                resfrac[fname].extend(fracs[fname][eeg_idx][day])
                resfragm[fname].extend(fragms[fname][eeg_idx][day])
                reshypn[fname].append(hypnos[fname][eeg_idx][day])
                ressasp[fname][eeg_idx].append(saved_specs[fname][eeg_idx][day])
                res_ids[fname][eeg_idx]  += 1
    return resfrac, resfragm, reshypn, ressasp, res_ids             
                
def save_results(fnames, scoring, fracs, fragms, hypnos, saved_specs, minutes, res_ids, suffixes={}, save_dir='./'):
    for fname in fnames:
        an_names = [str(el) + f'{suffixes.get(fname, {}).get(el, "")}' for el in sorted(list(res_ids[fname]))]
        an_ds = [f'_an{an}_d{d}' for an, eeg_idx in zip(an_names, sorted(list(res_ids[fname]))) for d in range(res_ids[fname][eeg_idx])]# saved_specs[fname]
        scoring.stage_fraction_report(os.path.join(save_dir, fname[:-4]+'_fracs.csv'), minutes, fracs[fname], col_names=an_ds) 
        # можно указать папку в начале имени файла вместо ./
        scoring.stage_duration_report(os.path.join(save_dir, fname[:-4]+'_fragms.csv'), fragms[fname], nums=[el[1:] for el in an_ds])
        hypncols = ['hypno'+el for el in an_ds]
        hdf = pd.DataFrame(OD([(hypncol, pd.Series(el)) for hypncol, el in zip(hypncols, hypnos[fname])])) #(OD([('hypno'+str(i), pd.Series(el)) for i, el in enumerate(hypnos[fname])]))
        hdf.to_csv(os.path.join(save_dir, fname[:-4]+'_hypnos.csv'), index=False)
        # specs
        df = {}
        for eeg_idx, an_name in zip(sorted(list(saved_specs[fname])), an_names): # str
            for day in range(len(saved_specs[fname][eeg_idx])): # int
                for band in sorted(list(saved_specs[fname][eeg_idx][day])): # tuple
                    for stage in sorted(list(saved_specs[fname][eeg_idx][day][band])): # int
                        df[f'An#{an_name}_d{day}_{band[0]}-{band[1]}Hz_{scoring.stage_map[stage]}'] = saved_specs[fname][eeg_idx][day][band][stage] #eeg_idx
        if len(df) > 0: pd.DataFrame(df).to_csv(os.path.join(save_dir, fname[:-4]+f'_specs.csv'), index=False)
        html = f'{fname} plots<br />\n'
        for picname, encoded in FIGSAVE.items():
            if not (fname[:-4] in picname): continue
            html += picname + "<br />\n" +  '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + "<br />\n"

        with open(os.path.join(save_dir, f'{fname[:-4]}_plots.html'),'w') as f:
            f.write(html)

def nearest_w_nr(selection, df):
    centroids = {}
    relabel = []
    for c in np.unique(df['state']):
        centroids[c] = np.mean(df[['Theta', '~Delta']][df['state'] == c].values, axis=0)
    for idx in selection.index:
        if df['state'].iloc[idx] == 2:
            dists = {c: np.linalg.norm(v-df[['Theta', '~Delta']].iloc[idx].values) for c, v in centroids.items()}
            if dists[0] < dists[1]:
                # print(0, df[['Theta', '~Delta']].iloc[idx])
                relabel.append(0)
            else:
                # print(1, df[['Theta', '~Delta']].iloc[idx])
                relabel.append(1)
        else:
            relabel.append(df['state'].iloc[idx])
    return np.array(relabel)          

def refine_ellipses(vote_dict, to_interactive_first, ellipse_percentage, fnames, scoring, rule, cluster_artifact, n_back, n_hours_cycle_bars, minutes, hours_fragm, red_mask, n_hours_cycle_rem, expert_hypnos, verbose, save_dir='./'):
    # %matplotlib inline
    to_interactive = {}
    hypnos = {}
    fragms = {}
    fracs, exp_fracs = {}, {}
    for fn, fname in enumerate(fnames):
        print('File \"'+fname+'\"')
        to_interactive[fname] = {}
        fracs[fname] = []
        exp_fracs[fname] = []
        fragms[fname] = []
        hypnos[fname] = []
        for eeg_idx in range(len(vote_dict[fname])):
            print('Animal '+str(eeg_idx))
            n_days = len(vote_dict[fname][eeg_idx])
            to_interactive[fname][eeg_idx] = []
            
            for day in range(n_days):
                print('Day '+str(day))
    # ГОЛОСОВАНИЕ
                to_vote, failed, mask1, art_mask, red_to_interactive = vote_dict[fname][eeg_idx][day][:5]
                first_hypno = to_interactive_first[fname][eeg_idx][day][0]
                rts = vote_dict[fname][eeg_idx][day][-1]
                nr_mask = vote_dict[fname][eeg_idx][day][5]
                new_to_vote = []
                for rt_c, rt in enumerate(rts):
                    sample = zscore(rt[first_hypno[nr_mask] == 2], axis=0)
                    full_sample = zscore(rt, axis=0)
                    kd = KD(kernel='epanechnikov').fit(sample)
                    dens_pred = kd.score_samples(sample)
                    thr_ca = -7
                    percent_mask = dens_pred > thr_ca
                    while np.where(percent_mask)[0].size/len(sample) > (ellipse_percentage/100):
                        # print(np.where(percent_mask)[0].size/len(sample), ellipse_percentage/100)
                        thr_ca += .1
                        percent_mask = dens_pred > thr_ca
                    thr_ca -= .1   
                    points = sample[dens_pred >= thr_ca]
                    hull = ConvexHull(points)
                    if verbose > 0:
                        plt.rcParams['figure.figsize'] = (30, 8)
                        ax = plt.subplot(121)
                        ax.hist(dens_pred, 50)
                        ax.set_xlabel('Log-likelihood probability density', fontsize=18)
                        ax.set_ylabel('Count', fontsize=18)
                        ax.axvline(thr_ca)
                        ax.tick_params(axis='both', labelsize=13)
                        
                        ax = plt.subplot(122)
                        ax.set_xlabel(f'Theta {rt_c}', fontsize=18)
                        ax.set_ylabel('Delta', fontsize=18)
                        ax.scatter(full_sample[:, 0], full_sample[:, 1], c=first_hypno[nr_mask])
                        # scatter = ax.scatter(*sample.T, c=dens_pred < thr_ca)
                        # legend1 = ax.legend(scatter.legend_elements()[0], ('Normal', 'Outlier'), prop=font)
                        points2 = full_sample[first_hypno[nr_mask] == 2]
                        points2 = points2[dens_pred >= thr_ca]
                        for simplex in hull.simplices:
                            ax.plot(points2[simplex, 0], points2[simplex, 1], 'c')
                        ax.tick_params(axis='both', labelsize=13)
                        # ax.add_artist(legend1)
                        ax.set_ylim(np.percentile(full_sample[:, 1], .5), np.percentile(full_sample[:, 1], 95))
                        ax.set_xlim(np.percentile(full_sample[:, 0], .5), np.percentile(full_sample[:, 0], 99.5))
                        plt.show()
                        
                        plt.rcParams['figure.figsize'] = (10, 8)
                        # kmeans = KMeans(n_clusters=2).fit(points)
                        try:
                            # labels = GM(n_components=2, covariance_type='full').fit_predict(points)
                            labels = SC(n_clusters=2).fit(points).labels_
                            # labels = gm.labels_
                            print(f'Silhoutte = {silhs(points, labels):.3f}; Calinski-Harabasz = {chs(points, labels):.3f};  Davies-Bouldin = {dbs(points, labels):.3f}')
                            left, right = points[labels == 0, 0], points[labels == 1, 0]
                            if np.mean(left) > np.mean(right):
                                labels = np.abs(labels - 1)
                        except:
                            labels = np.ones(len(points))
                        
                        plt.rcParams['figure.figsize'] = (16.4, 4)
                    
                        ax = plt.subplot(121)
                        ax.scatter(*points.T, c=labels)
                        ax = plt.subplot(122)
                        ax.scatter(rt[:, 0], rt[:, 1], c=first_hypno[nr_mask])
                        ax.set_ylim(np.percentile(rt[:, 1], .5), np.percentile(rt[:, 1], 95))
                        ax.set_xlim(np.percentile(rt[:, 0], .5), np.percentile(rt[:, 0], 99.5))
                        plt.show()
                    
                    
#                     # REM by outlier percentage of ellipses
#                     mask = np.zeros(len(art_mask))
#                     sample = rt[first_hypno[nr_mask] == 2]
#                     points = sample[dens_pred >= thr_ca]
#                     hull = ConvexHull(points)
                    
#                     mid1 = mask[nr_mask]
#                     mid1[points_in_hull(rt, hull)] = 1
#                     mask[nr_mask] = mid1
#                     new_to_vote.append(mask)
                    
                    # points by clustering
                    mask = np.zeros(len(art_mask))
                    mid1 = mask[nr_mask]
                    mid2 = mid1[first_hypno[nr_mask] == 2]
                    mid3 = mid2[dens_pred >= thr_ca]
                    mid3 = labels
                    mid2[dens_pred >= thr_ca] = mid3
                    mid1[first_hypno[nr_mask] == 2] = mid2
                    mask[nr_mask] = mid1
                    new_to_vote.append(mask)
                print('Дальше всё по-старому, но с результатами голосования уже по эллипсам')
                mask = rem_vote(new_to_vote, rule=rule-failed) # ПРАВИЛО ГОЛОСОВАНИЯ rule МОЖЕТ БЫТЬ ЦЕЛЫМ ЧИСЛОМ, КОТОРОЕ БУДЕТ ОЗНАЧАТЬ МИНИМАЛЬНОЕ КОЛИЧЕСТВО ГОЛОСОВ ДЛЯ ПРИНЯТИЯ РЕШЕНИЯ REM

                # СЕЙЧАС ВСЕГО 16 РАЗНЫХ КЛАСТЕРИЗАЦИЙ, ЗНАЧИТ, rule=8 ДЕТЕКТИРУЕТ REM, КОГДА ХОТЯ БЫ ПОЛОВИНА КЛАСТЕРИЗАЦИЙ ОПРЕДЕЛИЛА REM; 
                # ВСЕГДА ОТНИМАЕМ КОЛ-ВО ПРОВАЛЕННЫХ КЛАСТЕРИЗАЦИЙ failed

                # далее используется экземпляр scoring последнего дня последнего животного в последнем файле. что-то может быть некорректным
                if red_mask:   
                    scoring.last_hypno = combine_masks(mask, mask1, int(n_hours_cycle_rem*3600/scoring.window_sec2), art_mask)
                else:
                    scoring.last_hypno = combine_masks(mask, mask1, int(n_hours_cycle_rem*3600/scoring.window_sec2), art_mask)
                if cluster_artifact:
                    nr_mask, art_rms, (full_spec, freqs) = vote_dict[fname][eeg_idx][day][5:-1]
                    band = (2, 10)
                    theta_210 = np.sum(full_spec[:, (freqs >= band[0])&(freqs <= band[1])], axis=1)
                    if cluster_artifact == 'outside_ridges': cluster = np.hstack((art_rms[np.logical_not(nr_mask), None], theta_210[np.logical_not(nr_mask), None]))
                    elif cluster_artifact == 'everywhere': cluster = np.hstack((art_rms[:, None], theta_210[:, None]))

                    sample = zscore(cluster, axis=0)
                    kd = KD(kernel='epanechnikov').fit(sample)
                    dens_pred = kd.score_samples(sample)
                    thr_ca = -5
                    if verbose > 0:
                        plt.rcParams['figure.figsize'] = (30, 8)
                        ax = plt.subplot(121)
                        ax.hist(dens_pred, 50)
                        ax.set_xlabel('Log-likelihood probability density', fontsize=18)
                        ax.set_ylabel('Count', fontsize=18)
                        ax.axvline(thr_ca)
                        ax.tick_params(axis='both', labelsize=13)
                        
                        ax = plt.subplot(122)
                        ax.set_xlabel('Raw signal RMS', fontsize=18)
                        ax.set_ylabel('Theta 2-10 Hz PSD', fontsize=18)
                        scatter = ax.scatter(*sample.T, c=dens_pred < thr_ca)
                        legend1 = ax.legend(scatter.legend_elements()[0], ('Normal', 'Outlier'), prop=font)
                        ax.tick_params(axis='both', labelsize=13)
                        ax.add_artist(legend1)
                        plt.show()
                        plt.rcParams['figure.figsize'] = (30, 5)
                        plt.plot(art_rms)

                    slice_mask = np.logical_not(nr_mask) if cluster_artifact == 'outside_ridges' else np.full(len(nr_mask), True)
                    masked = np.array(art_rms[slice_mask])
                    masked[dens_pred > thr_ca] = None
                    to_plot = np.array(art_rms)
                    to_plot[np.logical_not(slice_mask)] = None
                    to_plot[slice_mask] = masked
                    
                    if verbose > 0:
                        plt.plot(to_plot)
                        plt.ylabel('Raw signal RMS', fontsize=18)
                        plt.xlabel('Epoch #', fontsize=18)
                        plt.tick_params(axis='both', labelsize=13)
                        plt.show()

                    _mask = np.array(np.isnan(to_plot))
                    scoring.last_hypno = cluster_art(scoring.last_hypno, _mask)
                scoring.last_hypno = no_singles(scoring.last_hypno, 1) # 0-1-0 -> 0-0-0; 2-1-2 -> 2-2-2
                scoring.last_hypno = no_singles(scoring.last_hypno, 0) # 1-0-1 -> 1-1-1; 2-0-2 -> 2-2-2
                scoring.no_rems_in_wake(n_back) # 0-2-...-2-0 -> 0-0-...-0-0 и 0-2-...-2-1 -> 0-0-...-0-1, неважно сколько двоек
                # scoring.no_single_a_between_b_and_c(0, 1, 2) # 1-0-2 -> 1-1-2, повторяет строку 607, так что бесполезно
                # scoring.no_single_a_between_b_and_c(1, 2, 2) # 2-1-2 -> 2-2-2, повторяет строку 605, так что бесполезно
                scoring.no_single_a_between_b_and_c(2, 1, 0) #  1-2-0 -> 1-1-0
                scoring.no_single_a_between_b_and_c(2, 1, 1) # 1-2-1 -> 1-1-1
                to_interactive[fname][eeg_idx].append((scoring.last_hypno, red_to_interactive))
                if verbose > 0: scoring.bars_by_cycles(n_hours_cycle=n_hours_cycle_bars) # соотношение стадий в циклах заданной длины
                if verbose > 0: scoring.plot_rms(scoring.last_hypno)
                # если есть гипнограмма, можно посмотреть точности
                # scoring.compare(os.path.join(dir_, fname[:-3]+'txt'))
                fracs[fname].extend(scoring.stage_fraction(minutes))
                fragms[fname].extend(scoring.stage_duration(hours_fragm))
                hypnos[fname].append(scoring.last_hypno) 
                if not (expert_hypnos is None):
                    scoring.txthypno = expert_hypnos[fname][eeg_idx][day][:min(len(expert_hypnos[fname][eeg_idx][day]), len(scoring.last_hypno))]
                    scoring.compare(None)
                    scoring.last_hypno = scoring.txthypno
                    exp_fracs[fname].extend(scoring.stage_fraction(minutes))
                    if verbose > 0:
                        dct, dct_exp = fracs[fname][-1], exp_fracs[fname][-1]
                        plt.rcParams['figure.figsize'] = (20, 4)
                        for i, stage in enumerate(sorted(list(dct.keys()))):
                            ax = plt.subplot(100 + len(dct)*10 + i + 1)
                            ax.plot(np.array(dct[stage])*100, label=f'{scoring.stage_map[int(stage)]} auto')
                            ax.plot(np.array(dct_exp[stage])*100, label=f'{scoring.stage_map[int(stage)]} expert')
                            ax.legend(prop=font)
                            ax.set_ylabel('Stage presence, %', fontsize=18)
                            ax.set_xlabel('Hour', fontsize=18)
                            ax.tick_params(axis='both', labelsize=13)
                        plt.tight_layout()
                        plt.show()

    for fname in fnames:
        scoring.stage_fraction_report(os.path.join(save_dir, fname[:-4]+'_fracs.csv'), minutes, fracs[fname], col_names=[f'_an{an}_d{d}' for an in vote_dict[fname] for d in range(len(vote_dict[fname][an]))]) # можно указать папку в начале имени файла вместо ./
        scoring.stage_duration_report(os.path.join(save_dir, fname[:-4]+'_fragms.csv'), fragms[fname])
        pd.concat([pd.DataFrame({'hypno'+str(i): el}) for i, el in enumerate(hypnos[fname])], axis=1).to_csv(os.path.join(save_dir, fname[:-4]+'_hypnos.csv'), index=False)
    return to_interactive

METRICS = {}

def metrics_summary():
    for metric, lst in METRICS.items():
        if 'matrix' in metric:
            mats = np.dstack([mat[..., None] for mat in lst])
            print('Averaged confusion matrix:\n', mats.mean(axis=-1))
            print('Confusion matrix std.:\n', mats.std(axis=-1))
        else:
            lst = np.array(lst)
            if lst.ndim == 1:
                lst = lst[:, None]
            print(f'{metric}: ', end='')
            for i in range(lst.shape[1]): print(f'{np.mean(lst[:, i]):.2f}±{np.std(lst[:, i]):.2f}', end='; ')

class Scorer:
    """
    """
    def __init__(self, filename, window_sec1, n_hours_cycle, window_sec2=None, delta=(3, 4), theta=(6.5, 9), filt_params={'length': 1501, 'transition': 0.6, 'window': 'kaiser'}, add_bands=(), delta_mode='band', theta_mode='one', pre_filt=None, depr=[0, 0], spindles=None, cluster_strictness=0.5, verbose=0):
        # add **kwargs
        self.spindles = spindles
        self.delta_refs_copy = None
        self.depr = depr
        self.pre_filt = pre_filt
        self.delta_mode = delta_mode
        self.theta_mode = theta_mode
        self.fname = filename
        self.window_sec1 = window_sec1
        self.cluster_w_nr_rule = cluster_strictness
        if window_sec2 is None:
            self.window_sec2 = window_sec1
        else:
            self.window_sec2 = window_sec2
        self.n_hours_cycle = n_hours_cycle
        self.thr1_cycles, self.thr2_cycles = [], []
        self.delta_plot_count = 0
        self.extension = filename.split('.')[-1]
        self.reader_map = {'smr': neo.io.Spike2IO, 'plx': neo.io.PlexonIO}

        reader = neo.io.Spike2IO(self.fname, try_signal_grouping=False)  # describe how to change for any other reader
        self.data = reader.read(lazy=False)[0].segments[0]
#         print(len(reader.read(lazy=False)[0].segments))
        self.eeg_count = 0
        self.eeg_idxs = []
        self.bands = {'delta': delta, 'theta': theta}
        self.fir_params = filt_params
        self.duration_ranges = ((0, 30), (30, 120), (120, 300), (300, 600), (600, 5000))
        self.stage_map = {0: 'Wake', 1: 'NREM', 2: 'REM', 8:'A', 4: 'x_spindle'}
        self.archived = {}
        self.additional_bands = add_bands
        self.additional_rmss = []
        self.points_per_cycle = int(self.n_hours_cycle*3600/self.window_sec1)
        # self.points_per_cycle2 = int(self.window_sec1*self.n_hours_cycle*3600/(self.window_sec2**2))
        self.cluster_voting = None
        self.txthypno = None
        self.eeg_map = []
        self.force_tht_ratio, self.force_ratio = None, None
        self.verbose=verbose
        if self.extension in self.reader_map:
            reader = self.reader_map[self.extension](self.fname, try_signal_grouping=False)  # describe how to change for any other reader
            self.data = reader.read(lazy=False)[0].segments[0]

            for i in range(len(reader.raw_annotations['blocks'][0]['segments'][0]['signals'])):
                ann = reader.raw_annotations['blocks'][0]['segments'][0]['signals'][i]
                # print(ann)
                chs = ann['__array_annotations__']['channel_names']
    #             print(chs)
                for j in range(len(chs)):
                    if ('eeg' in chs[j].lower()) or ('ch' in chs[j].lower()):
                        self.eeg_map.append((i, j))
        elif self.extension == 'hdf':
            f = h5py.File(self.fname, 'r')
            self.data = f['eeg']
            self.sf = float(f['eeg'].attrs['sf'][0])
            chs = f['eeg'].attrs['ch_names']
            for j, ch in enumerate(chs):
                if ('eeg' in ch.lower()) or ('ch' in ch.lower()):
                    self.eeg_map.append((j, j))

        # self.eeg_map = [(0, 0), (0, 2), (0, 4), (0, 6), (0, 8)]
        self.eeg_count = len(self.eeg_map)
        s = 's' if self.eeg_count > 1 else ''
        # print('The file contains {} EEG channel{}.'.format(self.eeg_count, s))
    
    def fir(self, freqs, mode):
        b = ss.firwin(self.fir_params['length'], freqs, pass_zero=mode, fs=self.sf, width=self.fir_params['transition'], window=self.fir_params['window'])
        return ss.filtfilt(b, 1, self.eeg)
    
    def plot_filter(self, freqs, mode):
        b = ss.firwin(self.fir_params['length'], freqs, pass_zero=mode, fs=self.sf, width=self.fir_params['transition'], window=self.fir_params['window'])
        w, h = ss.freqz(b, worN=2048)
        plt.rcParams['figure.figsize'] = (10, 5)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.sf*w / (2*np.pi), 20 * np.log10(np.maximum(abs(h), 1e-5)))
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude [dB]')
        ax.grid(which='both', axis='both')
        if type(freqs) in (int, float):
            ax.axvline(freqs, c='k')
            ax.set_xlim(0, freqs+5)
        else:
            ax.axvline(freqs[0], c='k')
            ax.axvline(freqs[1], c='k')
            ax.set_xlim(0, freqs[1]+5)
        plt.show()
    
    def set_thrs(self, thr1, thr2):
        if thr1 is not None:
            if type(thr1) in (int, float):
                self.thr1_cycles = [thr1]*self.n_cycles
            else:
                self.thr1_cycles = thr1
        if thr2 is not None:
            if type(thr2) in (int, float):
                self.thr2_cycles = [thr2]*self.n_cycles
            else:
                self.thr2_cycles = thr2
    
    def pick_eeg(self, eeg_idx, day_n=0, skip_minutes=0):
        ids = self.eeg_map[eeg_idx]
        if self.extension in self.reader_map:
            self.eeg = np.array(self.data.analogsignals[ids[0]])[:, ids[1]].flatten()

            self.sf = float(self.data.analogsignals[ids[0]].sampling_rate)
        else:
            # print(self.sf)
            self.sf /= 5
            self.eeg = np.array(self.data)[::5, ids[1]].flatten()

        self.eeg = self.eeg[day_n*round(self.sf*3600*24):round((day_n+1)*self.sf*3600*24)]#self.eeg[day_n*round(self.sf*3600*24):(day_n+1)*round(self.sf*3600*24)]
        if self.sf % 1 != 0:
            print(f'Non-integer sampling rate {self.sf:.4f} correction started...', end=' ')
            info = mne.create_info(['EEG1'], sfreq=self.sf, ch_types=['eeg'])
            raw = mne.io.RawArray(self.eeg.reshape((1, -1)), info)
            self.sf = round(self.sf)
            raw.resample(self.sf, npad='auto', n_jobs=-1) 
            self.eeg = raw.get_data().flatten()
            print('Done.')
        # print(f'Starting from {skip_minutes}th minute')
        self.eeg = self.eeg[int(self.sf*60*skip_minutes):]   
#        self.n_cycles = ceil(len(self.eeg)/self.sf/3600/self.n_hours_cycle)
        # self.eeg = self.eeg[30000*int(self.sf):]
        self.n_cycles = round(len(self.eeg)/self.sf/3600/self.n_hours_cycle)
        
        print('{:.2f} hours of data loaded. {} cycles of {} hours each, sf = {}'.format(len(self.eeg)/self.sf/3600, self.n_cycles, self.n_hours_cycle, self.sf))
        self.actual_hours = len(self.eeg)/self.sf/3600
    
    def rms(self, arr, ws=None):
        if ws is None:
            ws = self.window_sec1
        fs_out = 1/ws
        arr = np.array(arr.flatten())
        window = int(self.sf / fs_out)
        arr_outp = []
        for i in range(int(len(arr) / window)):
            arr_outp.append(np.sqrt(np.mean(arr[i*window:(i+1)*window]**2)))
        return np.array(arr_outp).flatten()
    
    def rms_smooth(self, series, window_sec=50, ws=None): # series - 'delta' or 'ratio'
        if isinstance(series, np.ndarray):
            arr = series
        elif series == 'delta':
            arr = self.deltarms
        elif series == 'theta':
            arr = self.ratio
        else:
            arr = series
        if ws is None:
            ws = self.window_sec1
        fs = 1 / ws
        window = round(window_sec * fs)
        arr = np.array(arr.flatten())
        arr_outp, i = [], 0
        while i < len(arr):
            arr_outp.append(np.sqrt(np.mean(arr[max(0, i-window): min(len(arr)-1, i+window)]**2)))
            i += 1
        arr = np.array(arr_outp).flatten()
        if isinstance(series, np.ndarray):
            return arr
        elif series == 'delta':
            self.archived.setdefault('delta', []).append(np.array(self.deltarms))
            self.deltarms = arr
        elif series == 'theta':
            self.archived.setdefault('ratio', []).append(np.array(self.deltarms))
            self.ratio = arr 
        else:
            return arr
    
#     @nb.jit()
#     def rms_smooth_delta(self, series, window_sec=50, ws=None):
#         if ws is None:
#             ws = self.window_sec1
#         fs = 1 / ws
#         window = round(window_sec * fs)
#         arr = np.array(series.flatten())
#         arr_outp, i = [], 0
#         while i < len(arr):
#             arr_outp.append(np.sqrt(np.mean(arr[max(0, i-window): min(len(arr)-1, i+window)]**2)))
#             i += 1
#         return np.array(arr).flatten()
    
    def plot_rms(self, series, thr=None, ymax=None, ws=None, mask=None):# series - 'delta' or 'ratio'
        if series == 'delta':
            seq = self.deltarms
            if ws is None:
                ws = self.window_sec1
        elif series == 'theta':
            if ws is None:
                ws = self.window_sec1
            if self.theta_mode == 'one':
                seq = self.ratio
            else:
                seqs = self.thetas
        else:
            seq = series
        if ws is None:
            ws = self.window_sec1
#         if (series == 'delta') or (self.theta_mode == 'one'):
#         plt.rcParams['figure.figsize'] = (9.5, 3)
        fig = plt.figure(figsize=(30, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.arange(len(seq))*ws/3600, seq)
        if mask is not None:
            masked = np.array(seq)
            masked[mask == 0] = None
            plt.plot(np.arange(len(seq))*ws/3600, masked, c='lime', lw=3)
        plt.xlim(0, len(seq)*ws/3600)
        if ymax is not None:
            plt.ylim(np.min(seq), ymax)
        plt.xlabel('Время, час', fontsize=15)
        plt.ylabel('Значение параметра', fontsize=15)
        if type(thr) in (float, int):
            plt.axhline(thr, ls='--', c='k')
        elif thr is not None:
            for i, t in enumerate(thr):
                plt.axhline(thr[i], ls='--', xmin=i/len(thr), xmax=(i+1)/len(thr), c='k')
        plt.tight_layout()
        plt.show()
                
    def set_ratio(self, idx):
        self.ratio = self.thetas[idx]
    
    def band_diff(self, band, ws=None):
        if ws is None:
            ws = self.window_sec1
        lowerf, upperf = self.fir(band[0],'lowpass'), self.fir(band[1],'lowpass')
        return self.rms(lowerf, ws), self.rms(upperf, ws)
    
    def fft_feats(self, freqs=((1, 2), (2, 3), (3, 4), (4, 5)), log = False, use_welch = False, manual=False, use_MT=False):
        if use_welch:
            freq, spec_abs = ss.welch(self.raw_epochs, axis=-1, fs=self.sf, nperseg=512, nfft=1024)
        elif use_MT:
            # print('using MT')
            frequency_range = [1, 16]  # Limit frequencies 
            time_bandwidth = 2  # Set time-half bandwidth
            num_tapers = 5  # Set number of tapers (optimal is time_bandwidth*2 - 1)
            window_params = [4, 1]  # Window size is 4s with step size of 1s
            min_nfft = 0  # No minimum nfft
            detrend_opt = 'constant'  # detrend each window by subtracting the average

            multiprocess = True  # use multiprocessing
            n_jobs = -1  # use 3 cores in multiprocessing

            weighting = 'unity'  # weight each taper at 1
            clim_scale = False # do not auto-scale colormap
            xyflip = False  # do not transpose spect output matrix
            
            res = []
            for i in range(len(self.raw_epochs)):
                spect, stimes, freq = multitaper_spectrogram(self.raw_epochs[i], self.sf, frequency_range, time_bandwidth, num_tapers, window_params, min_nfft, detrend_opt, multiprocess, n_jobs, weighting, False, False, clim_scale, False, xyflip)
                res.append(spect.sum(axis=1))
            # print('done')
            spec_abs = np.vstack([el[None] for el in res])
    
        else:
            
            fft = np.fft.fft(self.raw_epochs, axis=-1)
            freq = np.fft.fftfreq(self.raw_epochs.shape[-1], 1/self.sf)
            spec_abs = np.abs(fft[:,freq >= 0])
        if log:
            spec_abs = np.log10(spec_abs)
            if np.any(spec_abs < 0): spec_abs -= np.min(spec_abs, axis=1)[:, None]
            
        # freq, spec_abs = ss.welch(self.raw_epochs, axis=-1, fs=self.sf, nperseg=512, nfft=1024)
        freq = freq[freq >= 0]
        if manual:
            return spec_abs, freq
        return [spec_abs[:, (freq >= band[0])&(freq <= band[1])] for band in freqs]
    
    def prepare_rms(self, delta_mode='bandpass', theta_mode='diff', theta_res='ratio', add_mode='diff', smooth_fft=False, w=20):
        print('Preparing data...', end='')
        self.sf = int(self.sf)
        self.raw_epochs = np.vstack([self.eeg[i*self.sf*self.window_sec1:(i+1)*self.sf*self.window_sec1].reshape((1, -1)) for i in range(len(self.eeg)//(self.sf*self.window_sec1))])
        if self.pre_filt is not None:
            print('pre-filtering... ', end='')
            self.eeg = self.fir(self.pre_filt, 'highpass')
        if self.delta_mode == 'band':
            if delta_mode == 'bandpass':
                self.deltaf = self.fir(self.bands['delta'], 'bandpass')
                self.deltarms = self.rms(self.deltaf)
            else:
                dlowerrms, dupperrms = self.band_diff(self.bands['delta'])
                self.deltarms = dupperrms-dlowerrms
        else:
            # self.delta_clusters = self.fft_feats(self.bands['delta'], use_MT=False)#[zscore(np.log10(el), axis=0) for el in ]
            # if smooth_fft:
            #     print('smoothing deltas... ', end='')
            #     self.delta_clusters = [np.apply_along_axis(self.rms_smooth, 0, band, *(w, )) for band in self.delta_clusters]
            spec_abs, freq = self.fft_feats([], manual=True)#[zscore(np.log10(el), axis=0) for el in ]
            minld = np.min([el[0] for el in self.bands['delta']])
            maxld = np.max([el[1] for el in self.bands['delta']])
            if smooth_fft:
                print('smoothing deltas... ', end='')
                slc = (freq >= minld) & (freq <= maxld)
                t1 = perf_counter()
                spec_abs[:, slc] = np.apply_along_axis(self.rms_smooth, 0, spec_abs[:, slc], *(w, ))
                print('Done')
                # print(perf_counter()-t1)
            self.delta_clusters = [spec_abs[:, (freq >= band[0])&(freq <= band[1])] for band in self.bands['delta']]
        if self.theta_mode =='one':
            if theta_mode == 'diff':
                tlowerrms, tupperrms = self.rms(self.fir((0, self.bands['theta'][0]),'bandpass'), self.window_sec2), self.rms(self.fir((0, self.bands['theta'][1]),'bandpass'), self.window_sec2)#self.band_diff(self.bands['theta'])
                if theta_res == 'ratio':
                    self.ratio = tupperrms/tlowerrms
                elif theta_res == 'diff':
                    self.ratio = tupperrms-tlowerrms
            else: 
                self.thetaf = self.fir(self.bands['theta'], 'bandpass')
                self.thetarms = self.rms(self.thetaf, self.window_sec2)
        elif self.theta_mode =='cluster':
            self.thetas = []
            self.ratios = []
            for band in []: #self.bands['theta']:
                thetaf = self.fir((band-1.5, band), 'bandpass')
#                 self.thetas.append(thetaf)
                self.ratios.append(self.rms_smooth(self.rms(thetaf, self.window_sec1), 10, ws=self.window_sec1))
            self.ratio = [] #self.ratios[0]
            # self.delta_ref = self.rms_smooth(self.rms(self.fir((4, 5), 'bandpass'), self.window_sec2), 20, ws=self.window_sec2)
            
            self.delta_refs = []
            for db in self.delta_ref_bands: # db = delta_bands[i]
                # print(db)
                self.delta_refs.append([]) #self.rms_smooth(self.rms(self.fir(db, 'bandpass'), self.window_sec1), 20, ws=self.window_sec1))
            self.delta34 = [] #self.rms_smooth(self.rms(self.fir((3, 4), 'bandpass'), self.window_sec1), 30, ws=self.window_sec1)
        else:
            self.thetas = []
            if theta_mode == 'diff':
                theta_pairs = []
                for band in self.bands['theta']:
                    theta_pairs.append(self.band_diff(band, self.window_sec2))
                for pair in theta_pairs:
                    if theta_res == 'ratio':
                        self.thetas.append(pair[1]/pair[0])
                    elif theta_res == 'diff':
                        self.thetas.append(pair[1]-pair[0])
            else:
                for band in self.bands['theta']:
                    thetaf = self.fir(band, 'bandpass')
                    self.thetas.append(self.rms(thetaf, self.window_sec2))
            self.thetas = [self.thetas[0], self.thetas[1]*self.thetas[2]]
                
        self.additional_rmss = []
        for band in self.additional_bands:
            print(band)
            if add_mode == 'diff':
                lowerrms, upperrms = self.band_diff(band)
                self.additional_rmss.append(upperrms-lowerrms)
            else:
                self.additional_rmss.append(self.rms(self.fir(band, 'bandpass')))
#         lowerrms, upperrms = self.band_diff((0.1, 1))
#         self.art_rms = upperrms-lowerrms
        print('smoothing raw RMS... ', end='')
        self.art_rms = self.rms_smooth(self.rms(self.eeg), 20)
        print('Done.')
        # if self.spindles is not None:
        #     self.spindle_feats = [np.sum(el, axis=1) for el in self.fft_feats(self.spindles)]
    
    def set_thr2_auto(self, ratio, ymax=None, plot=False, mask=None, w_hump=150, zbefore=10, oafter=10, ridge_thrs=None, w_humps_rises=150, rise_len_thr=30, refined_ridges=None):
        if isinstance(refined_ridges, np.ndarray):
            self.nr_mask = refined_ridges
            return
        
        plt.rcParams['figure.figsize'] = (30, 5)
        rt = np.array(ratio)
        rng = np.arange(1, 90.1, .5)
        thrs = []
        #
        sm = self.rms_smooth(self.art_rms[self.art_mask], w_hump)
        # smshort = self.rms_smooth(self.art_rms[self.art_mask], w_humps_rises)
        # smdiff = np.insert(np.sign(np.diff(smshort)), 0, 0)
        
        if (self.verbose > 0) and (not (self.txthypno is None)):
            hypno = np.array(self.txthypno)
            min_len = min(len(hypno), len(self.art_mask))
            hypno = hypno[:min_len]
            hypno = hypno[self.art_mask[:min_len]]
            for stage, color in zip(range(3), ('purple', 'darkcyan', 'gold')):
                ins = np.array(sm)
                ins[hypno != stage] = None
                plt.plot(ins, c=color)
                plt.xlim(0, len(ins))
            plt.show()
        self.sm_art_rms = sm #self.rms_smooth(self.art_rms, w_hump) #sm
        if ridge_thrs is None:
            gms = GM(n_components=2, covariance_type='full', tol=1e-6).fit_predict(sm.reshape(-1, 1)).astype(float)
            centr0, centr1 = np.mean(sm[gms==0.]), np.mean(sm[gms==1.])
            if centr0 > centr1:
                gms -= 0.5
                gms *= -1
                gms += .5
                gms = gms.astype(int)
        else:
            if np.all(np.array(ridge_thrs) == None):
                chunk = len(sm) // len(ridge_thrs)
                gms, fake_thrs = [], []
                for i in range(len(ridge_thrs)):
                    smchunk = sm[i*chunk:(i+1)*chunk]
                    cur_gms = GM(n_components=2, covariance_type='full', tol=1e-6).fit_predict(smchunk.reshape(-1, 1)).astype(float).flatten()
                    centr0, centr1 = np.mean(smchunk[cur_gms==0.]), np.mean(smchunk[cur_gms==1.])
                    if centr0 > centr1: 
                        cur_gms = (cur_gms - 1) * (-1)
                        centr1, centr0 = centr0, centr1
                    # lower the thr
                    cur_thr = np.max(smchunk[cur_gms==0.])
                    cur_percent = (cur_thr - centr0) / (centr1 - centr0)
                    if cur_percent > .1:
                        print(f'{i+1}th threshold too high {cur_thr:.2f} (lower mean = {centr0:.3f}, upper mean = {centr1:.3f}, range = {centr1-centr0:.3f}), lowering to ', end='')
                        cur_thr = centr0 + .1 * (centr1 - centr0)
                        print(f'{cur_thr:.3f}')
                        cur_gms = (smchunk > cur_thr).astype(int)                    
                    #
                    fake_thrs.append(cur_thr) # np.max(smchunk[cur_gms==0.])
                    gms.append(cur_gms)
                ridge_thrs = fake_thrs
                gms = np.concatenate(gms)[:len(sm)]
                #
                if len(gms) < len(sm): 
                    if (len(sm) - len(gms)) == 1:
                        gms = np.concatenate((gms, [gms[-1]]))
                    else:
                        extra_chunk = sm[-(len(sm)-len(gms)):]
                        extra_gms = GM(n_components=2, covariance_type='full', tol=1e-6).fit_predict(extra_chunk.reshape(-1, 1)).astype(float).flatten()
                        centr0, centr1 = np.mean(extra_chunk[extra_gms==0.]), np.mean(extra_chunk[extra_gms==1.])
                        if centr0 > centr1: extra_gms = (extra_gms - 1) * (-1)
                        gms = np.concatenate((gms, extra_gms))
            else:
                gms = []
                chunk = len(sm) // len(ridge_thrs)
                for i, thr in enumerate(ridge_thrs):
                    gms.append((sm[i*chunk:(i+1)*chunk] >= thr).astype(float))
                gms = np.concatenate(gms)[:len(sm)]
                if len(gms) < len(sm): gms = np.concatenate((gms, (sm[-(len(sm)-len(gms)):] >= thr).astype(float)))
                
            self.ridge_thrs = ridge_thrs
        plt.rcParams['figure.figsize'] = (30, 6)        
        # gms_thr = np.array(gms)
        del_ridges = np.argwhere(gms == 0).size
        
        grpby = [(el[0], len(list(el[1]))) for el in groupby(gms.astype(int))]
        cur = 0
        i = 0
        # выбрасываем точки после пересечения с порогом
        inter_mask = np.ones(len(gms))
        while i < (len(grpby)-1):
            el = grpby[i]
            cur += el[1]
            if (el[0] == 0) and (el[1] >= zbefore):
                el_ = grpby[i+1]
                if el_[1] >= oafter:
                    # nr_mask[cur: cur+oafter] = False
                    gms[cur: cur+oafter] = 0
                    # inter_mask[cur: cur+oafter] = 0
            i += 1
        
        del_inter = np.argwhere(gms == 0).size - del_ridges
        
        # удаление длинных восходящих участков
        # rise_mask = np.ones(len(gms))
#         ids = np.arange(len(gms))
#         cur = 0
#         # gms_before_rise = np.array(gms)
#         for el in groupby(smdiff):
#             lst = list(el[1])
#             if el[0] != 1: 
#                 cur += len(lst)
#                 continue
#             if len(lst) >= rise_len_thr: 
#                 gms[cur: cur + len(lst)] = 0
#                 # rise_mask[cur: cur+len(lst)] = 0
#             cur += len(lst)
        
#         del_rises = np.argwhere(gms == 0).size - del_ridges - del_inter
        
        if ridge_thrs is None: 
            art_thr = np.max(sm[gms==0])#np.mean((np.min(self.art_rms[gms==1]), np.max(self.art_rms[gms==0])))
            nr_mask = gms.astype(bool) #sm > art_thr
            # print(art_thr, np.min(sm[nr_mask]))
            self.ridge_thrs = [art_thr]
        else:
            nr_mask = gms.astype(bool)
        
        
        
        dummy = np.full(len(self.art_mask), False)
        dummy[self.art_mask] = nr_mask
        nr_mask = dummy
        if self.verbose > 1:
            # print(sm, np.unique(gms))
            for sm_ in (sm,): #, smshort
                ins, outs = np.array(sm_), np.array(sm_)
                ins[gms == 0], outs[gms == 1] = None, None
                plt.plot(ins, c='g')
                plt.plot(np.arange(len(outs)), outs, c='r')
                if ridge_thrs is None: plt.axhline(art_thr)
                else:
                    ratio_chunk = 1 / len(ridge_thrs)
                    for i, thr in enumerate(ridge_thrs):
                        plt.axhline(thr, xmin=i * ratio_chunk, xmax=(i+1)*ratio_chunk)
                plt.xlim(0, len(ins))
                plt.show()
        self.nr_mask = np.logical_and(nr_mask, self.art_mask)
        
        print(f'{del_ridges} wakes removed by threshold (ridges)')
        print(f'{del_inter} epochs removed after threshold intersection')
        # print(f'{del_rises} epochs removed on the left rising slopes of ridges')
        print(f'{np.argwhere(gms == 0).size} epochs removed in total by ridges')
        
    def cluster_vote(self, deltas, cycle_slice, art_mask):
        gms = []
        fishers = []
        to_ave = []
    
        for i, delta in enumerate(deltas):
            gm_model = GM(n_components=2, covariance_type='full', reg_covar=1e-3, weights_init=self.weights).fit(delta)
            # print(gm_model.weights_)
            gm_preds = gm_model.predict(delta)
            if gm_model.means_[0, 0] > [gm_model.means_[1, 0]]:
                gm_preds = ((gm_preds-.5)*(-2)+1)/2
            gms.append(gm_preds)
            fishers.append(np.var(delta)/(np.var(delta[gm_preds==1]) + np.var(delta[gm_preds==1])))
            if self.visualize_clusters and (self.verbose > 0):
                plt.rcParams['figure.figsize'] = (25, 5)
                if self.txthypno is None:
                    y = np.zeros(np.where(art_mask)[0].size)
                else:
                    y = self.txthypno[cycle_slice][art_mask]
                    y[y > 1] = 0
                np.nan_to_num(delta, copy=False)
                comps = PCA(n_components=2, svd_solver='arpack').fit_transform(delta)
                to_ave.append(comps)
                ax = plt.subplot(121)
                p01, p099, p11, p199 = np.percentile(comps[:, 0], .5), np.percentile(comps[:, 0], 99.5), np.percentile(comps[:, 1], .5), np.percentile(comps[:, 1], 99.5)
                ax.set_xlim(p01, p099)
                ax.set_ylim(p11, p199)
                cs = np.array(gms[-1])
                cs[y == 3] = 2
                ax.scatter(comps[:, 0], comps[:, 1], c=y, alpha=.4) #len(delta_)//2
                
                ax.set_title('True labels, {}-{} Hz'.format(*self.bands['delta'][i]))
                
                ax = plt.subplot(122)
                ax.set_xlim(p01, p099)
                ax.set_ylim(p11, p199)
                ax.scatter(comps[:, 0], comps[:, 1], c=gms[-1], alpha=.4)
                ax.set_title('GM labels, {}-{} Hz'.format(*self.bands['delta'][i]))
                plt.show()
                
                ridges = np.sum(delta, axis=1)
                plt.scatter(np.arange(len(comps)), ridges, c=cs, s=8)
                plt.plot(ridges, c='grey', lw=.8)
                plt.title('GM labels, {}-{} Hz'.format(*self.bands['delta'][i]))
                plt.show()
                print()
                
                # for paper
                # print('{}-{} Hz'.format(*self.bands['delta'][i]))
#                 stage_map = {0: 'Wake+REM', 1: 'NREM'}
#                 colors = {'Wake+REM': 'darkcyan', 'NREM': 'purple'}
#                 # 1D
#                 fig, axes = plt.subplots(2, 1, figsize=(8.3, 5), sharex=True)
#                 fig.text(.002, .27, f'Delta amplitude {self.bands["delta"][i][0]}-{self.bands["delta"][i][1]} Hz, μv', ha='center', fontsize=18, family='Cambria', rotation=90)
#                 ax = axes.flatten()[0]
#                 ax.plot(ridges, c='grey', lw=.8)
#                 for c in np.unique(y):
#                     ax.scatter(np.arange(len(ridges))[y == c], ridges[y == c], alpha=.5, s=10, label=stage_map[c], c=colors[stage_map[c]])
#                 ax.legend(fontsize=14, handletextpad=0, borderpad=.5, loc='lower right')
#                 ax.tick_params(axis='both', labelsize=14)
#                 ax.text(55, 510, 'A', fontsize=24)
                
#                 ax = axes.flatten()[1]
#                 ax.set_xlabel('Epoch #', fontsize=18, family='Cambria')
#                 ax.tick_params(axis='both', labelsize=14)
#                 ax.plot(ridges, c='grey', lw=.8)
#                 for c in np.unique(gms[-1]):
#                     ax.scatter(np.arange(len(ridges))[gms[-1] == c], ridges[gms[-1] == c], alpha=.5, s=10, label=stage_map[c], c=colors[stage_map[c]])
#                 ax.legend(fontsize=14, handletextpad=0, borderpad=.5, loc='lower right')
#                 ax.tick_params(axis='both', labelsize=14)
#                 ax.text(55, 510, 'B', fontsize=24)
#                 plt.tight_layout()
#                 # 
#                 if self.bands['delta'][i] == (2, 6): plt.savefig(r'G:\LA\2024\qual_figs\delta_cluster_1d.png', dpi=300, bbox_inches='tight')
#                 plt.show()
                
#                 # scatters
#                 plt.rcParams['font.family'] = 'Cambria'
#                 # hypno_ = expert_hypnos[fname][eeg_idx][day][:len(nr_mask)][nr_mask]
                
#                 fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.7), sharey=True)
#                 fig.text(0.5, -.01, 'Principal component 1, a.u.', ha='center', fontsize=18, family='Cambria')
#                 ax = axes.flatten()[0]
#                 # ax.set_title('True labels, {}-{} Hz'.format(*self.bands['delta'][i]), fontsize=18)
#                 for c in np.unique(y):
#                     ax.scatter(*comps[y == c].T, alpha=.5, s=10, label=stage_map[c], c=colors[stage_map[c]])                
#                 ax.legend(fontsize=14, handletextpad=0, borderpad=.5)
#                 # ax.set_xlim(p01, p099)
#                 # ax.set_ylim(p11, p199)
#                 ax.set_ylabel('Principal component 2, a.u.', fontsize=18, family='Cambria')
#                 ax.tick_params(axis='both', labelsize=14)
#                 ax.text(65, 30, 'A', fontsize=24)

#                 ax = axes.flatten()[1]
#                 colors2 = {1: 'purple', 0: 'darkcyan'}
#                 stage_map2 = {1: 'NREM', 0: 'Wake+REM'}
#                 for c in np.unique(gms[-1]):
#                     ax.scatter(*comps[gms[-1] == c].T, c=colors2[c], alpha=.5, s=10, label=stage_map2[c]) # полный кластер с окрашенным горизонтальным разделением
#                 # ax.legend(fontsize=14, handletextpad=0, borderpad=.5)
#                 # ax.set_xlim(p01, p099)
#                 # ax.set_ylim(p11, p199)
#                 # ax.set_title('GM labels, {}-{} Hz'.format(*self.bands['delta'][i]), fontsize=18)
#                 ax.tick_params(axis='both', labelsize=14)
#                 ax.legend(fontsize=14, handletextpad=0, borderpad=.5)
#                 ax.text(65, 30, 'B', fontsize=24)

#                 plt.tight_layout()
#                 # 
#                 # if self.bands['delta'][i] == (2, 6): plt.savefig(r'G:\LA\2024\qual_figs\delta_cluster.png', dpi=300, bbox_inches='tight')
#                 plt.show()
                
        # for paper
#         plt.rcParams['font.family'] = 'Cambria'
#         stage_map = {0: 'Wake+REM', 1: 'NREM'}
#         colors = {'Wake+REM': 'darkcyan', 'NREM': 'purple'}
#         fig, ax = plt.subplots(1, 1, figsize=(4.15, 3.7), sharey=True)
#         ax.set_xlabel('Average principal component 1, a.u.', fontsize=18, family='Cambria')
#         ax.set_ylabel('Average principal component 2, a.u.', fontsize=18, family='Cambria')
#         comps = np.dstack([el[..., None] for el in to_ave]).mean(axis=-1)
#         for c in np.unique(y):
#             ax.scatter(*(comps[y == c]).T, alpha=.5, s=10, label=stage_map[c], c=colors[stage_map[c]])                
#         ax.legend(fontsize=14, handletextpad=0, borderpad=.5)
        
#         for rnd_idx in np.random.choice(np.arange(len(comps)), 5):
#             ax.scatter(comps[rnd_idx, 0], comps[rnd_idx, 1], marker='s', c='k', s=12)
#             for prime in to_ave:
#                 ax.plot([comps[rnd_idx, 0], prime[rnd_idx, 0]], [comps[rnd_idx, 1], prime[rnd_idx, 1]], lw=.5, c='k')
#                 ax.scatter(prime[rnd_idx, 0], prime[rnd_idx, 1], marker='*', c='grey', s=12)
        
#         ax.tick_params(axis='both', labelsize=14)
#         plt.tight_layout()
#         plt.savefig(r'G:\LA\2024\qual_figs\delta_cluster_ave_scatter.png', dpi=300, bbox_inches='tight')
#         plt.show()
        #
    # for paper compare clusters
        # stage_map = {0: 'Wake+REM', 1: 'NREM'}
        # colors = {'Wake+REM': 'darkcyan', 'NREM': 'purple'}
        # fig, axes = plt.subplots(1, 3, figsize=(8.3, 3.1), sharey=True)
        # hpno = y#expert_hypnos[fname][eeg_idx][day][:len(nr_mask)][nr_mask] if not (expert_hypnos is None) else np.zeros(len(TO_AVE[fname][eeg_idx][day][0][0]))
        # random_ids = np.random.choice(np.arange(len(to_ave[0])), 9)
        # for i, (clust) in enumerate(to_ave[:3]):
        #     x, y = clust.T
        #     ax = axes.flatten()[i]
        #     for c in np.unique(hpno):
        #         ax.scatter(x[hpno == c], y[hpno == c], c=colors[stage_map[c]], label=stage_map[c], s=10, alpha=.5)
        #     for j, rnd_idx in enumerate(random_ids):
        #         ax.text(x[rnd_idx], y[rnd_idx], str(j+1), c='k')
        #     ax.legend(fontsize=12, handletextpad=0, borderpad=.5, loc='upper left')
        #     ax.tick_params(axis='both', labelsize=14)
        #     # if i == 0: ax.set_ylabel(f'Delta ({delta_theta_combs[i][0][0]}-{delta_theta_combs[i][0][1]} Hz) amplitude, μV', fontsize=18, family='Cambria')
        #     ax.set_title(f'{self.bands["delta"][i][0]}-{self.bands["delta"][i][1]} Hz', fontsize=14, family='Cambria')
        #     # ax.set_xlabel(f'{delta_theta_combs[i][1][0]}-{delta_theta_combs[i][1][1]} Hz', fontsize=12, family='Cambria')
        # fig.text(0.5, -.01, 'Principal component 1, a.u.', ha='center', fontsize=18, family='Cambria')
        # fig.text(-.01, .05, 'Principal component 2, a.u.', ha='center', fontsize=18, family='Cambria', rotation=90)
        # plt.tight_layout()
        # plt.savefig(r'D:\LA\2024\qual_figs\deltas\delta_random_scatter_compare{}.png'.format(str(datetime.now()).replace(':', '_')), dpi=300, bbox_inches='tight')
        # plt.show()
                
        cluster_preds = np.hstack([predict.reshape((-1, 1)) for predict in gms])
        vote = cluster_preds.mean(axis=1)
        # print('Fisher =', np.mean(fishers))
        return (vote > self.cluster_w_nr_rule).astype(int)#.round()
    
    
    def get_theta_nan(self, delta, rt):
        rem_mask = [0]
        thr = np.mean(rt)
        d_bigger_t = delta[0] > rt[0]
        i = 1
        while i < len(rt):
            if (rt[i] < thr) or (delta[i] < thr):
                rem_mask.append(0)
                d_bigger_t = delta[i] > rt[i]
                i += 1
                continue
            if d_bigger_t:
                if delta[i] < rt[i]:
                    j = i
                    while j < len(rt):
                        if (rt[j] < thr):
                            d_bigger_t = False
#                                 break 
                        if (rt[j] < delta[j]):
                            d_bigger_t = True
                            break
                        rem_mask.append(1)
                        j += 1
                    i = j
                    continue
            if delta[i] > rt[i]:
                d_bigger_t = True
            rem_mask.append(0)
            i += 1

#             print(rem_mask)
        rem_mask = np.array(rem_mask).astype(bool)
        theta_nan = np.array(rt)
        theta_nan[np.logical_not(rem_mask)] = None
        return theta_nan, rem_mask
    
    def pick_delta(self, theta, deltas, plot=False, ylim=None, xlim=None, normalize=False, cycle_hours=24):
        means = []
        self.delta_refs_copy = list(map(np.array, self.delta_refs))
        tm = np.mean(theta)
        tms = []
        if normalize:
            ppc = int(cycle_hours*3600/self.window_sec1)
            for i in range(int(len(theta)/ppc)):
                theta_chunk = theta[i*ppc:(i+1)*ppc]
                gms = GM(n_components=2, covariance_type='full').fit_predict(theta_chunk.reshape((-1, 1)))
                tms.append(sorted([np.mean(theta_chunk[gms==0]), np.mean(theta_chunk[gms==1])]))
        for i, d in enumerate(deltas):
            cur_means = []
            if normalize:
#                 print(int(len(d)/ppc), len(tms), len(d), len(theta), ppc)
                for j in range(int(len(d)/ppc)):
                    delta_chunk = d[j*ppc:(j+1)*ppc]
                    gms = GM(n_components=2, covariance_type='full').fit_predict(delta_chunk.reshape((-1, 1)))
                    cur_means.append(sorted([np.mean(delta_chunk[gms==0]), np.mean(delta_chunk[gms==1])]))
                    zeros, ones = delta_chunk[gms == 0], delta_chunk[gms == 1]
                    if np.mean(zeros) > np.mean(ones):
                        gms = gms*(-1) + 1
                    delta_chunk[gms==0] -= np.mean(delta_chunk[gms==0]) - tms[j][0]
                    delta_chunk[gms==1] -= np.mean(delta_chunk[gms==1]) - tms[j][1]
                    d[j*ppc:(j+1)*ppc] = delta_chunk
            means.append(np.abs(np.mean(d)-tm))
            if (self.verbose > 1) and (self.delta_plot_count == 0):

                plt.rcParams['figure.figsize'] = (25, 5)
                plt.plot(theta, lw=.7)
                plt.plot(d, lw=.7)
                plt.plot(self.get_theta_nan(d, theta)[0], c='r', lw=.7)
                plt.axhline(np.mean(d), c='r', ls='--')
                plt.axhline(np.mean(theta), c='k', ls='--')
                if ylim is not None:
                    plt.ylim(*ylim)
                if xlim is not None:
                    plt.xlim(*xlim)
                plt.legend(('Theta (6.5-8 Hz)', 'Delta ({}-{} Hz)'.format(*self.delta_ref_bands[i])), fontsize=14)
                plt.show()
        self.delta_plot_count = 1
        idx = np.argmin(means).flatten()[0]
#         print('Best matching delta is # {}'.format(idx))
        self.delta_ref = deltas[idx]
        self.ratios = [theta]
        
    def ratio_ee(self, ratios, cs, mask):
        masks, thrs = [], []
        rng = np.arange(1) #, 25.1, .5)
        plt.rcParams['figure.figsize'] = (30, 10)
        delta = self.delta_ref[cs][mask]
        # print('in ee')
        # print(len(ratios))
        for rt in ratios:
#             print('ratio', rt.shape)
            gms = GM(n_components=2, covariance_type='full', tol=1e-8).fit_predict(rt.reshape((-1, 1)))
            zeros, ones = rt[gms == 0], rt[gms == 1]
#             print(np.mean(zeros), np.mean(ones), np.max(zeros), np.min(ones))
#             print(zeros, ones)
            if np.mean(zeros) > np.mean(ones):
#                 print('inversion')
                zeros, ones = ones, zeros
                gms = gms*(-1) + 1
            thr = np.mean(rt)
#             
            if self.force_ratio is None:
                black_ratio = rt/ self.delta34[cs][mask]
            else:
                black_ratio = np.array(self.force_ratio)[cs][mask]
            
            theta_nan, rem_mask = self.get_theta_nan(delta, rt)
            
            rt = rt[gms == 1]
            dlt = delta[gms == 1]
            
            
            if self.priority_theta_nan:
                theta_nan = np.array(self.current_theta_nan).astype(float)
#                 print(np.unique(theta_nan))
            else:
                self.theta_nan.append(np.array(theta_nan))
                self.theta_nan[-1][np.isnan(self.theta_nan[-1])] = 0
                self.theta_nan[-1] = self.theta_nan[-1].astype(bool)
#             ax.plot(theta_nan, c='r')
#             ax1 = ax.twinx()
            if self.force_ratio is None:
                brt = black_ratio-np.convolve(black_ratio, np.ones(40)/40, mode='same')#self.rms_smooth(black_ratio, 20)-np.convolve(black_ratio, np.ones(40)/40, mode='same')
            else:
                brt = self.force_ratio[cs][mask]
            if self.force_thr_ratio is None:
                thrbrt = np.mean(brt)+np.std(brt)
            else:
                thrbrt = self.force_thr_ratio
            brt[brt < thrbrt] = 0
            theta_nan[brt == 0] = None
            theta_nan[np.isnan(theta_nan)] = 0
            theta_nan[theta_nan != 0] = 1
            masks.append(theta_nan.reshape((-1, 1)))
            lofs = []
        preds = np.mean(np.hstack(masks), axis=1)
#         preds[preds > 0] = 0
#         preds[preds < 0] = 1
        return preds, thrs     
    
    def thr_artifacts(self, ylims=None):
        n, b = np.histogram(self.art_rms, 150)
        gms_art = np.zeros(len(self.art_rms))
        gms_art[0] = 1
        nmax = np.argmax(n)
        art_sample = np.array(self.art_rms)
        if not self.force_art_thr:
            counter = 0
            while abs(1 - len(art_sample[gms_art == 0])/len(art_sample[gms_art == 1])) > .9:         
                gms_art = GM(n_components=2, covariance_type='full', weights_init=(.5, .5), tol=1e-6).fit_predict(art_sample.reshape((-1, 1)))
                if counter > 5:
                    art_sample = art_sample[art_sample < np.percentile(art_sample, 95)]
                    gms_art = gms_art[:len(art_sample)]
                if counter > 15: break
                counter += 1
            means = {cod: np.mean(art_sample[gms_art == cod]) for cod in np.unique(gms_art)}
            means = sorted([(j, i) for i, j in means.items()])
            bigger = art_sample[gms_art == means[-1][1]]
            smaller = art_sample[gms_art == means[0][1]]
            art_thr = np.mean(bigger)+ 4 * np.std(bigger)
            art_thr1 = np.mean(smaller)- 4 * np.std(smaller) 
        else:
            art_thr1, art_thr = self.force_art_thr
        art_mask = (self.art_rms < art_thr) & (self.art_rms > art_thr1) 
        art_mask = art_mask
        # self.art_mask = art_mask
        
        art_mask = np.array(art_mask).astype(int)
        d = np.diff(art_mask)
        for i in range(len(d)):
            idx = d[i]
            if idx == 1:
                art_mask[i+1] = False
            if idx == -1:
                art_mask[i] = False
        art_mask = art_mask.astype(bool)
        self.art_mask = art_mask
        self.thr_arts_count = np.argwhere(~art_mask).size
        if self.verbose > 0:
            plt.rcParams['figure.figsize'] = (30, 6)
            plt.plot(self.art_rms)
            plt.axhline(art_thr, c='r')
            plt.axhline(art_thr1, c='r')
            plt.plot(self.art_rms[art_mask])
            plt.xlim(0, len(self.art_mask))
            plt.ylim(art_thr1 - .1*abs(art_thr), art_thr + .1*abs(art_thr))
            plt.show()
        
    
    
    def scoring(self, n_back=1, series1=None, series2=None, ylims=None, n_back_rule='old'): 
#         print(n_back)
        if self.delta_mode == 'cluster':
            try:
                self.deltarms = np.zeros(len(self.ratio))
                table = np.hstack((self.deltarms.reshape((-1, 1)), self.ratio.reshape((-1, 1))))
            except:
                self.deltarms = np.zeros(len(self.delta_clusters[0])) #ratios[0]
                table = np.hstack((self.deltarms.reshape((-1, 1)), self.delta_clusters[0][:, 0].reshape((-1, 1)))) #self.ratios[0].reshape((-1, 1))
        
        
        if series1 is not None:
            table[:, 0] = series1.flatten()
        if series2 is not None:
            table[:, 1] = series2.flatten()    
        result_mask_cycles = []
        
        art_masks = [self.art_mask[i*self.points_per_cycle:(i+1)*self.points_per_cycle] for i in range(self.n_cycles)]
        for i in range(self.n_cycles):
            # print('Cycle #{}'.format(i+1), end=' -> ')
            # self.visualize_clusters = True
            
            self.weights = None#[.4, .6] if self.depr[i] == 0 else [.05, .95]
            # print(self.weights)
            cycle_slice = slice(i*self.points_per_cycle, (i+1)*self.points_per_cycle)
            if self.delta_mode == 'band':
                thr1 = self.thr1_cycles[i]
                mask1 = table[cycle_slice, 0][art_masks[i]] > thr1
            else:
                mask1 = self.cluster_vote([cl[cycle_slice][art_masks[i]] for cl in  self.delta_clusters], cycle_slice, art_masks[i])
            mask1 = mask1.astype(int)
            result_mask_cycles.extend(mask1)
        self.mask1 = np.array(result_mask_cycles).flatten()   
        print()
    
    def rem_scoring(self, n_hours_cycle, n_back=2):
        self.points_per_cycle2 = int(n_hours_cycle*3600/self.window_sec2)
#         print(self.points_per_cycle2)
        if self.theta_mode == 'cluster':
            n_cycles = round(len(self.ratios[0])/self.points_per_cycle2)
        else:
            n_cycles = round(len(self.ratio)/self.points_per_cycle2)
#         print(n_cycles)
        # print(self.points_per_cycle2)
        if self.window_sec1 != self.window_sec2:
            factor = int(self.window_sec1/self.window_sec2)
#             print(factor)
            self.art_mask2 = np.zeros(len(self.art_mask)*factor)
#             print(self.art_mask.shape, self.art_mask2.shape)
            for i in range(factor):
                self.art_mask2[i:][::factor] = self.art_mask
            art_masks = [self.art_mask2[i*self.points_per_cycle2:(i+1)*self.points_per_cycle2].astype(bool) for i in range(n_cycles)]
        else:
            factor = 1
            art_masks = [self.art_mask[i*self.points_per_cycle2:(i+1)*self.points_per_cycle2] for i in range(n_cycles)]
        rem_mask = []
#         print([el.shape for el in art_masks])
        for i in range(n_cycles):
#             print('Cycle #{}'.format(i+1), end=' -> ')
            cycle_slice = slice(i*self.points_per_cycle2, (i+1)*self.points_per_cycle2)
            if self.priority_theta_nan:
                self.current_theta_nan = self.theta_nan[i]
            if self.theta_mode == 'cluster':
                # print(cycle_slice, art_masks[i])
                # mask2, cur_thrs = self.ratio_ee([cl[cycle_slice][art_masks[i]] for cl in  self.ratios], cycle_slice, art_masks[i])
                mask2, cur_thrs = self.ratio_ee([cl[cycle_slice][art_masks[i]] for cl in  self.ratios], cycle_slice, art_masks[i])
                self.ratio_thrs.append(cur_thrs)
            else:
                try:
                    mask2 = self.ratio[cycle_slice][art_masks[i]] > self.force_thr_ratio[i]
                except:
                    mask2 = self.ratio[cycle_slice][art_masks[i]] > self.force_thr_ratio#mask2 = self.ratio > thr2
            mask2 = mask2.astype(int)
            rem_mask.extend(mask2)
#             print(np.sum(rem_mask)/len(rem_mask))
        # print(len(rem_mask))
        if self.window_sec1 != self.window_sec2:
            # print(factor)
            rm = []
            for i in range(int(len(rem_mask)/factor)):
                rm.append(np.mean(rem_mask[i*factor:(i+1)*factor]))
            rm = np.array(rm)
            rm[rm == 1] = 1
            rm[rm < 1] = 0
            rem_mask = np.array(rm).round()
        
        rem_mask = 2*np.array(rem_mask)    
        res_mask = self.mask1 + rem_mask
#         print(len(res_mask), len(self.art_mask), np.where(self.art_mask)[0].size)
        res_mask[res_mask > 2] = 2
        dummy = np.zeros(self.points_per_cycle2*n_cycles)
        dummy[self.art_mask[:len(dummy)]] = res_mask
        
        j = 0
        not_mask = np.argwhere(np.logical_not(self.art_mask)).flatten()
        while j < len(not_mask):
            idx = not_mask[j]
            k = idx
            while (k-idx)<2:
                dummy[k] = dummy[idx-2]
                k += 1
                if k >= len(not_mask):
                    break
            j = k
        
        if self.spindles is not None:
            bsln = [(np.mean(ft[:len(dummy)][dummy == 0]), np.std(ft[:len(dummy)][dummy == 0])) for ft in self.spindle_feats]
            # print(bsln)
            for i in range(len(dummy)):
                if dummy[i] == 2:
#                     print(bsln[0][0] + 2*bsln[0][1], bsln[1][0] + 2*bsln[1][1])
                    check = [self.spindle_feats[j][i] > (bsln[j][0] + 2*bsln[j][1]) for j in range(len(self.spindle_feats))]
#                     print(check, i)
                    check = all(check)
                    if check:
                        c = 0
#                         print(c, dummy[i-c])
                        while (c >= 0) and (dummy[i-c] == 2):
                            dummy[i-c] = 4
                            c += 1
                        c = 0
                        while (c < len(dummy)) and (dummy[i+c] == 2):
                            dummy[i+c] = 4
                            c += 1
        self.last_hypno, self.raw_hypno = dummy, dummy
#         print()
        
    def cluster_clear(self, theta, delta, plot=False):
        rt = np.hstack((theta[self.nr_mask][:, None], delta[self.nr_mask][:, None]))
        for cont_ in np.arange(20, 30.1, .5):
            cont = cont_/100
            clf = EE(contamination=cont)
            y = clf.fit_predict(rt.reshape((-1, 2))) 
            kurt = kurtosis(rt[y == 1], axis=0)
            if np.all(kurt < 0):
                break
        if plot: print(cont)
        thr = np.min(rt[y==1][:, 1])
        gmm = GM(n_components=2, covariance_type='full', tol=1e-6).fit(rt[y != 1][:, 0].reshape(-1, 1))
        gms = gmm.predict(rt[y != 1][:, 0].reshape(-1, 1))
        centr0, centr1 = np.mean(rt[y != 1][:, 0][gms==0.]), np.mean(rt[y != 1][:, 0][gms==1.])
        inversion = False
        if centr0 > centr1:
            inversion = True
            gms = gms.astype(float)
            gms -= 0.5
            gms *= -1
            gms += .5
            gms = gms.astype(int)
        centr = np.mean(rt[y != 1][:, 0][gms==1])
        thr2 = np.min(rt[y != 1][:, 0][gms==1])
        delt = (np.max(rt[y != 1][:, 0][gms==1])-np.min(rt[y != 1][:, 0][gms==1]))/50
        step = float(delt)
        while gmm.predict_proba([[centr - step]])[0, 1-int(inversion)] > .1:
            step += delt
        thr2 = centr-step
        if plot:
            plt.rcParams['figure.figsize'] = (12, 6)
            plt.scatter(theta[self.nr_mask], delta[self.nr_mask], c=self.last_hypno[self.nr_mask]==2, alpha=.7)
            plt.axhline(thr)
            plt.axvline(thr2, ymax=0.25)
            plt.show()
        
        aut = self.last_hypno[self.nr_mask]
        aut[(theta[self.nr_mask] < thr2) & (delta[self.nr_mask] < thr)] = 0 # 
        self.last_hypno[self.nr_mask] = aut
        
   
    def no_rems_after_wake(self, n_back):
        res_mask = self.last_hypno
        result_mask_cycles_ = [el for el in res_mask[:n_back]]
        for i in range(n_back, len(res_mask)):
            if (res_mask[i] == 2) and (np.sum(result_mask_cycles_[i-n_back:i]) == 0): result_mask_cycles_.append(0)
            else: result_mask_cycles_.append(res_mask[i])
        self.last_hypno = result_mask_cycles_
    
    def plot_last_hypno(self, plot_also=['spec'], misc_rms=None, thr=None, xlims=[None, None], expert_stage=None, ax2ylims=None, percents=(1, 99), hypno=True, ax1ylims=None): 
#         plt.rcParams['figure.figsize'] = (30, 6)
        plt.rcParams['figure.dpi'] = 96
        ws = None
#         fig, ax1 = plt.subplots()
#         ax1.set_ylim(-0.2, 2.2)
        fig = plt.figure(figsize=(9.5, 2.5))
        ax1 = fig.add_subplot(1, 1, 1)
        # ax1.set_ylim(-0.2, 2.2)
        ax2 = ax1.twinx()
        # np.arange(len(self.last_hypno))*self.window_sec/3600,
        for i in range(1, self.n_cycles):
            plt.axvline(i * self.points_per_cycle, c='goldenrod', lw=.7, ls='--') # self.window_sec */ 3600,
        if (xlims[0] is None) and (xlims[1] is None):
            try:
                xlims = (0, len(self.ratio))
            except:
                xlims = (0, self.actual_hours*3600/self.window_sec1) # 8640
        ax1.set_xlim(xlims[0], xlims[1])
        if type(thr) in (int, float, np.float64):
            ax2.axhline(thr, ls='--', c='g', lw=.7)
        elif thr is not None:
            for i, t in enumerate(thr):
                if (((i+1) * self.points_per_cycle <= xlims[1]) and ((i+1) * self.points_per_cycle >= xlims[0])) and (i * self.points_per_cycle <= xlims[0]):
                    left_frac, right_frac = 0, ((i+1) * self.points_per_cycle - xlims[0])/(xlims[1]-xlims[0])
                elif ((i * self.points_per_cycle <= xlims[1]) and (i * self.points_per_cycle >= xlims[0])) and ((i+1) * self.points_per_cycle >= xlims[1]):
                    left_frac, right_frac = (i * self.points_per_cycle - xlims[0])/(xlims[1]-xlims[0]), 1
                elif ((i * self.points_per_cycle <= xlims[1]) and (i * self.points_per_cycle >= xlims[0])) and (((i+1) * self.points_per_cycle <= xlims[1]) and ((i+1) * self.points_per_cycle >= xlims[0])):
                    left_frac, right_frac = (i * self.points_per_cycle - xlims[0])/(xlims[1]-xlims[0]), ((i+1) * self.points_per_cycle - xlims[0])/(xlims[1]-xlims[0])
                else:
                    left_frac, right_frac = 0, 0
                ax2.axhline(thr[i], ls='--', xmin=left_frac, xmax=right_frac, c='k', lw=.7)
        if 'rationan' in plot_also:
            ser = np.array(self.ratio)
            ser[self.txthypno != 2] = np.nan
            if hypno: ax2.plot(np.arange(len(self.ratio)), ser, color='orange', lw=.7)
            ws = self.window_sec2
        if 'ratiosmnan' in plot_also:
            ser = np.array(self.ratiosm)
            ser[self.txthypno != 2] = np.nan
            ax2.plot(np.arange(len(self.ratiosm)), ser, color='orange', lw=.7)
            ws = self.window_sec2
        if 'delta' in plot_also:
            ax2.plot(np.arange(len(self.delta)), self.deltarms, color='magenta', lw=.7) # *self.window_sec/3600
            ws = self.window_sec1
        if 'ratio' in plot_also:
            ax2.plot(np.arange(len(self.ratio)), self.ratio, color='orange', lw=.7) # *self.window_sec/3600
            ws = self.window_sec2
        if 'ratiosm' in plot_also:
            ax2.plot(np.arange(len(self.ratiosm)), self.ratiosm, color='orange', lw=.7) # *self.window_sec/3600
        if (expert_stage is not None) and (self.txthypno is not None):
            stage = np.array(self.txthypno)
            if expert_stage == 0:
                stage[stage == 0] = 3
                expert_stage = 3
            stage[stage != expert_stage] = 0
            if stage[0] == expert_stage: stage[0] -= 1
            d = np.diff(stage)
            l, r = np.argwhere(d>0).flatten()+1, np.argwhere(d<0).flatten()+1
            for i in range(min(len(l), len(r))):
                plt.axvspan(l[i], r[i], color='g', alpha=0.3)
        if 'raweegrms' in plot_also:
            ws = self.window_sec1
            ser = self.rms_smooth(self.rms(self.eeg), 20)
            ax1.plot(ser, c='r', lw=.7)
            ax1.scatter(np.arange(len(ser)), ser, lw=1, c='k', s=5)
            # if ax1ylims is not None:
            #     ax1.set_ylim(ax1ylims[0], ax1ylims[1])
        if 'raw' in plot_also:
            try:
                ax1.plot(np.linspace(0, len(self.ratio), len(self.eeg))[::20], self.eeg[::20], lw=.5, c='r')
            except:
                ax1.plot(np.linspace(0, self.actual_hours*3600/self.window_sec1, len(self.eeg))[::20], self.eeg[::20], lw=.5, c='r') #len(self.ratios[0])
#             ax1.set_xlim(xlims[0], xlims[1])
            # if hypno: ax2.plot(self.last_hypno, c='darkblue', lw=.7) 
#             print('1')
        if misc_rms is not None:
            colors = ('k', 'grey','brown',  'magenta')
            for i, _rms in enumerate(misc_rms):
                # print(colors[i], _rms)
                ax2.plot(_rms, c=colors[i], lw=.9)
            # if ax1ylims is not None:
            #     ax1.set_ylim(ax1ylims[0], ax1ylims[1])
        if 'spec' in plot_also:
            if ws is None:
                ws = self.window_sec1
            n_points_epoch = len(self.eeg)//(ws*self.sf)
            # epochs = np.vstack([self.eeg[i*n_points_epoch:(i+1)*n_points_epoch] for i in range(n_points_epoch)])
            f, t, Zxx = ss.stft(self.eeg, fs=self.sf, window='triang', nperseg=512, noverlap=200, nfft=1024, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
            t /= ws
            if ax1ylims is None:
                ax1ylims = (0, self.sf//2)
            power = 10*np.log(np.abs(Zxx[(f <= ax1ylims[1])&(f >= ax1ylims[0])])) 
            f = f[(f <= ax1ylims[1])&(f >= ax1ylims[0])]
            vmin, vmax = np.percentile(power, percents[0]), np.percentile(power, percents[1])
            ax1.pcolormesh(t, f, power, shading='gouraud', cmap='Greys', vmin=vmin, vmax=vmax, rasterized=True) #
            # ax1.set_xlim(xlims[0], xlims[1])
            # ax1.set_ylim(ax1ylims[0], ax1ylims[1])
            # ax1.tick_params(axis='both', left=False, labelleft=False)
            # if hypno: ax2.plot(self.last_hypno, c='r', lw=.7) 
            # ax1.set_xlim(xlims[0], xlims[1])
        # elif not ('raw'  in plot_also):
#             print('2')
        if ax1ylims is not None:
            ax1.set_ylim(ax1ylims[0], ax1ylims[1])
        if ax2ylims is not None:
            ax2.set_ylim(ax2ylims[0], ax2ylims[1])
        if hypno: 
            newy = list(ax1.get_ylim())
            newy[0], newy[1] = .95*newy[0], .95*newy[1]
            ax1.plot(scale_to_range(self.last_hypno, *newy), lw=.7)
                # ax1.set_ylim(-0.2, 2.2)
            # ax1.set_xlim(xlims[0], xlims[1])
#         plt.show()
        try:
            sldr = IntSlider(min=0, max=len(self.ratio)-100, step=100, description='Start', continuous_update=False, layout=Layout(width='900px'))
        except:
            sldr = IntSlider(min=0, max=(self.actual_hours*3600/self.window_sec1) - 100, step=100, description='Start', continuous_update=False, layout=Layout(width='900px'))

        def wupdate(w=500):
            sldr.min = 0
            try:
                sldr.max = len(self.ratio)-w
            except:
                sldr.max = (self.actual_hours*3600/self.window_sec1) - w
            sldr.step = 100
        win = IntText(value=500, step=100, description='Window', disabled=False, layout=Layout(width='150px'))
        plt.xlim(0, win.value)
        plt.tight_layout()
        iw = interact(wupdate, w=win)
        def update(w=0):
            plt.xlim(w, w+win.value)
            fig.canvas.draw_idle()
        i = interact(update, w=sldr)
        
    def bars_by_cycles(self, n_hours_cycle=12, save_dir=False):
        border = int(n_hours_cycle*60*(60 / self.window_sec1))
        points_per_cycle = int(n_hours_cycle*3600/self.window_sec1)
        suffix = ['st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th']
        n_cycles = self.window_sec1*len(self.last_hypno)/3600/n_hours_cycle
        if n_cycles%1 > 0:
            n_cycles = int(n_cycles) + 1
        else:
            n_cycles = int(n_cycles)
        res = []
        for i in range(n_cycles):
            halfday = np.array(self.last_hypno[i*points_per_cycle:(i+1)*points_per_cycle]).astype(int)
            nr, w, r = [len(np.where(halfday == code)[0]) for code in (1, 0, 2)]
            # plt.rcParams['figure.figsize'] = (10, 5)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title('{}{} 12 hours'.format(i+1, suffix[i]))
            bar_container = ax.bar(range(3), [w, nr, r])
            plt.bar_label(bar_container, [ w,nr, r])
            ax.set_xticks(range(3))
            ax.set_xticklabels(('Wake','Non-REM',  'REM'))
            # if save_dir: plt.savefig(save_dir[:-4]+f'_cycle#{i+1}', dpi=300, bbox_inches='tight')
            tmpfile = BytesIO()
            fig.savefig(tmpfile, format='png', bbox_inches='tight')
            plt.show()
            res.append(base64.b64encode(tmpfile.getvalue()).decode('utf-8'))
        if save_dir: return res
        # добавить сохранение в файл
        
    def prepare_rem_borders(self, n_conv=5, thr=None, xlims=[None, None], series=None, dc=250, detected=False, expert_stages=[]):
        
        if series is None:
            self.ratiosm = np.convolve(self.ratio, np.ones(5)/5, mode='same')
        else:
            self.ratiosm = series
#         self.ratiosm = dc_remove(self.ratiosm, dc)
#         self.ratiosm = self.rms_smooth(self.ratiosm, 20)
        plt.rcParams['figure.figsize'] = (30, 6)
        plt.plot(self.ratiosm)
        if thr is not None: plt.axhline(thr, ls='--', c='k') # порог для отображения
        if not all(map(lambda x: x is None, xlims)):
            plt.xlim(xlims[0], xlims[1]) # границы по оси Х, чтобы разглядеть все нужное
        else:
            plt.xlim(0, len(self.ratiosm))
        if self.txthypno is not None:
            colors = ['r', 'g']
            for stage in expert_stages:
                txthypno = np.array(self.txthypno)
                txthypno[txthypno != stage] = 0
                d = np.diff(txthypno)
                l, r = np.argwhere(d>0).flatten(), np.argwhere(d<0).flatten()
                for i in range(min(len(l), len(r))):
                    plt.axvspan(l[i], r[i], color=colors[stage-1], alpha=0.3)
        if detected and (self.raw_hypno is not None):
            dtctd = np.array(self.raw_hypno)
            dtctd[dtctd != 2] = 0
            d = np.diff(dtctd)
            l, r = np.argwhere(d>0).flatten(), np.argwhere(d<0).flatten()
            for i in range(min(len(l), len(r))):
                plt.axvspan(l[i], r[i], color='g', alpha=0.3, hatch='*')
        plt.ylim(np.percentile(self.ratiosm, 1), np.percentile(self.ratiosm, 99.7))
        print(np.percentile(self.ratiosm, 1), np.percentile(self.ratiosm, 99))
        plt.show()
        
    def continuous_rems(self, thr, ser=None):
        if ser is None:
            ser = [self.ratiosm, self.raw_hypno]
        c_mask = ser[0] > thr
        c_mask = c_mask.astype(int) * 2
        mask = np.array(ser[1])
        mask[mask != 2] = 0     
        rems, c_rems = self.rem_epochs_bounds(mask), self.rem_epochs_bounds(c_mask)
        t_rems = []
        for bounds in rems:
            for c_bounds in c_rems:
                if ((bounds[0] >= c_bounds[0]) and (bounds[0] <= c_bounds[1])) or ((bounds[1] >= c_bounds[0]) and (bounds[1] <= c_bounds[1])) or ((bounds[1] >= c_bounds[1]) and (bounds[0] <= c_bounds[0])) or ((bounds[1] <= c_bounds[1]) and (bounds[0] >= c_bounds[0])):
                    t_rems.append(c_bounds)
        t_scoring = np.zeros(len(mask)).astype(int)
        for bounds in t_rems:
            t_scoring[bounds[0]:bounds[1]] = 2
        mask = np.array(ser[1])
        mask += t_scoring
        mask[mask > 2] = 2
#         mask[mask == 4] = 2
        return mask 
    
    def multi_refine(self, n_conv=None, thr_mode='same'):
        refined = []
        chunk_pts = int(len(self.ratios[0]) / len(self.ratio_thrs))
        for c in range(len(self.ratio_thrs)):
            mask = self.raw_hypno[c*chunk_pts:(c+1)*chunk_pts]
            refined.append([])
            for i in range(len(self.ratio_thrs[c])):
                rt = self.ratios[i]
                if n_conv is not None:
                    rtc = self.rms_smooth(rt[c*chunk_pts:(c+1)*chunk_pts], n_conv, ws=self.window_sec2)
                else:
                    rtc = rt[c*chunk_pts:(c+1)*chunk_pts]
                if thr_mode == 'same':
                    thr = self.ratio_thrs[c][i][0]
                elif 'std' in thr_mode:
                    # print(c, i)
                    thr = self.ratio_thrs[c][i][1] + float(thr_mode.split('*')[0])*self.ratio_thrs[c][i][2]
                # print(thr, self.ratio_thrs[c][i][0])
                refined[-1].append(self.continuous_rems(thr, [rtc, mask]).reshape((-1, 1)))
        self.last_hypno = np.concatenate([np.mean(np.hstack(cycle), axis=1).round().astype(int) for cycle in refined])
        
                
    
    def rem_plus_one(self,):
        mask = np.array(self.raw_hypno)
        for i, st in enumerate(self.raw_hypno[:-1]):
            if st == 2:
                mask[i-1], mask[i+1] = 2, 2
                # mask[i-1] = 2
        self.last_hypno = mask
        
    def rem_epochs_bounds(self, hypno):
        scoring = [(int(el[0]), len(list(el[1]))) for el in groupby(hypno)]
        cur_idx, i = 0, 0
        rems = []
        while i < len(scoring):
            if scoring[i][0] == 2:
                rems.append((cur_idx, cur_idx+scoring[i][1])) 
            cur_idx += scoring[i][1]
            i += 1
        return rems

    def continuous_events(self, p_mask, exp_mask):
        p_rems, exp_rems = self.rem_epochs_bounds(p_mask), self.rem_epochs_bounds(exp_mask)
        inter = 0
        exclude = []
        for i, boundaries in enumerate(p_rems):
            for exp_bounds in exp_rems:
                if ((boundaries[0] >= exp_bounds[0]) and (boundaries[0] <= exp_bounds[1])) or ((boundaries[1] >= exp_bounds[0]) and (boundaries[1] <= exp_bounds[1])):
                    inter += 1
                    exclude.append(i)
        for boundaries in exp_rems:
            for i, p_bounds in enumerate(p_rems):
                if ((boundaries[0] >= p_bounds[0]) and (boundaries[0] <= p_bounds[1])) or ((boundaries[1] >= p_bounds[0]) and (boundaries[1] <= p_bounds[1])):
                    if i not in exclude: inter += 1
        print('Automatic scoring detected {} out of {} continuous REM-epochs identified by the expert, i.e., {:.2f}%'.format(inter, len(exp_rems), 100*inter/len(exp_rems)))
    
    def load_txthypno(self, fname):
        with open(fname, 'r') as f: # открываем гипнограмму эксперта
            self.txthypno = np.array([int(el) for el in f.readlines()]).flatten()
        self.txthypno[self.txthypno > 4] = 0
        try:
            self.txthypno = self.txthypno[:min(len(self.txthypno), len(self.ratio))]
        except:
            self.txthypno = self.txthypno[:min(len(self.txthypno), len(self.ratios[0]))]
    
    def compare_binary(self, fname):
        global METRICS
        mask = self.last_hypno.copy()
        if isinstance(fname, str): self.load_txthypno(fname)
        txthypno = self.txthypno.copy()
        min_len = min(len(txthypno), len(mask))
        # print('Difference in lengths = {}'.format(abs(len(txthypno)-len(mask))))
        mask, txthypno = mask[:min_len], txthypno[:min_len]
        # print('General accuracy = {:.2f}'.format(acc(txthypno, mask)))
        LOG.info('General accuracy = {:.2f}'.format(acc(txthypno, mask)))
        # print('Balanced accuracy = {:.2f}'.format(bacc(txthypno, mask)))
        LOG.info('Balanced accuracy = {:.2f}'.format(bacc(txthypno, mask)))
        # print('General Cohen\'s kappa = {:.2f}'.format(kappa(txthypno, mask)))
        LOG.info('General Cohen\'s kappa = {:.2f}'.format(kappa(txthypno, mask)))
        # print('F1 = {:.2f}'.format(f1_score(txthypno, mask, average='weighted')))
        LOG.info('F1 = {:.2f}'.format(f1_score(txthypno, mask, average='weighted')))
        # print('Confusion matrix:')
        LOG.info('Confusion matrix:')
        conf = cm(txthypno, mask)
        # print(conf)
        LOG.info(str(conf))
        accs = [conf[i, i]/np.sum(conf[i]) for i in range(len(conf))]
        # print('Accuracies by class: {:.2f}, {:.2f}, {:.2f}'.format(*accs))
        LOG.info('Accuracies by class: {:.2f}, {:.2f}, {:.2f}'.format(*accs))
        specs, sens, f1s = [], [], []
        # print(len(conf))
        for i in range(len(conf)):
            tp = conf[i, i]
            tn, fp, fn = 0, 0, 0
            for j in range(len(conf)):
                for k in range(len(conf)):
                    if (j != i) and (k != i): tn += conf[j, k]
            for j in range(len(conf)):
                if j != i: 
                    fp += conf[j, i]
                    fn += conf[i, j]
            sens.append(tp/(tp+fn))
            specs.append(tn/(tn+fp))
            f1s.append(tp/(tp + .5*(fp + fn)))
        # print('Sensitivities by class: {:.2f} {:.2f}'.format(*sens))
        LOG.info('Sensitivities by class: {:.2f} {:.2f}'.format(*sens))
        # print('Specificities by class: {:.2f} {:.2f}'.format(*specs))
        LOG.info('Specificities by class: {:.2f} {:.2f}'.format(*specs))
        # print('Cohen\'s kappas by class: {:.2f} {:.2f}'.format(*[kappa((txthypno == code).astype(int), (mask== code).astype(int)) for code in (0, 1)]))
        LOG.info('Cohen\'s kappas by class: {:.2f} {:.2f}'.format(*[kappa((txthypno == code).astype(int), (mask== code).astype(int)) for code in (0, 1)]))
        # print('F1 by class: {:.2f} {:.2f}'.format(*f1s))
        LOG.info('F1 by class: {:.2f} {:.2f}'.format(*f1s))
        LOG.info('\n')
    
    def compare(self, fname):
        mask = self.last_hypno.copy()
        if isinstance(fname, str): self.load_txthypno(fname)
        txthypno = self.txthypno.copy()
        min_len = min(len(txthypno), len(mask))
        # print('Difference in lengths = {}'.format(abs(len(txthypno)-len(mask))))
        mask, txthypno = mask[:min_len], txthypno[:min_len]
        print('General accuracy = {:.2f}'.format(acc(txthypno, mask)))
        METRICS.setdefault('General accuracy', [])
        METRICS['General accuracy'].append(acc(txthypno, mask))
        # LOG.info('General accuracy = {:.2f}'.format(acc(txthypno, mask)))
        print('Balanced accuracy = {:.2f}'.format(bacc(txthypno, mask)))
        METRICS.setdefault('Balanced accuracy', [])
        METRICS['Balanced accuracy'].append(bacc(txthypno, mask))
        # LOG.info('Balanced accuracy = {:.2f}'.format(bacc(txthypno, mask)))
        print('General Cohen\'s kappa = {:.2f}'.format(kappa(txthypno, mask)))
        METRICS.setdefault('General Cohen', [])
        METRICS['General Cohen'].append(kappa(txthypno, mask))
        # LOG.info('General Cohen\'s kappa = {:.2f}'.format(kappa(txthypno, mask)))
        print('F1 = {:.2f}'.format(f1_score(txthypno, mask, average='weighted')))
        METRICS.setdefault('F1', [])
        METRICS['F1'].append(f1_score(txthypno, mask, average='weighted'))
        # LOG.info('F1 = {:.2f}'.format(f1_score(txthypno, mask, average='weighted')))
        print('Confusion matrix:')
        # LOG.info('Confusion matrix:')
        conf = cm(txthypno, mask)
        METRICS.setdefault('Confusion matrix', [])
        METRICS['Confusion matrix'].append(conf)
        print(conf)
        # LOG.info(str(conf))
        accs = [conf[i, i]/np.sum(conf[i]) for i in range(len(conf))]
        print('Accuracies by class: {:.2f}, {:.2f}, {:.2f}'.format(*accs))
        METRICS.setdefault('Accuracies by class', [])
        METRICS['Accuracies by class'].append(accs)
        # LOG.info(('Accuracies by class: {:.2f}, {:.2f}, {:.2f}'.format(*accs)))
        
        specs, sens, f1s = [], [], []
        # print(len(conf))
        for i in range(len(conf)):
            tp = conf[i, i]
            tn, fp, fn = 0, 0, 0
            for j in range(len(conf)):
                for k in range(len(conf)):
                    if (j != i) and (k != i): tn += conf[j, k]
            for j in range(len(conf)):
                if j != i: 
                    fp += conf[j, i]
                    fn += conf[i, j]
            sens.append(tp/(tp+fn))
            specs.append(tn/(tn+fp))
            f1s.append(tp/(tp + .5*(fp + fn)))
        print('Sensitivities by class: {:.2f} {:.2f} {:.2f}'.format(*sens))
        METRICS.setdefault('Sensitivities by class', [])
        METRICS['Sensitivities by class'].append(sens)
        # LOG.info('Sensitivities by class: {:.2f} {:.2f} {:.2f}'.format(*sens))
        print('Specificities by class: {:.2f} {:.2f} {:.2f}'.format(*specs))
        METRICS.setdefault('Specificities by class', [])
        METRICS['Specificities by class'].append(specs)
        # LOG.info('Specificities by class: {:.2f} {:.2f} {:.2f}'.format(*specs))
        kappas = [kappa((txthypno == code).astype(int), (mask== code).astype(int)) for code in (0, 1, 2)]
        print('Cohen\'s kappas by class: {:.2f} {:.2f} {:.2f}'.format(*kappas))
        METRICS.setdefault('Cohen by class', [])
        METRICS['Cohen by class'].append(kappas)
        # LOG.info('Cohen\'s kappas by class: {:.2f} {:.2f} {:.2f}'.format(*[kappa((txthypno == code).astype(int), (mask== code).astype(int)) for code in (0, 1, 2)]))
        print('F1 by class: {:.2f} {:.2f} {:.2f}'.format(*f1s))
        METRICS.setdefault('F1 by class', [])
        METRICS['F1 by class'].append(f1s)
        # LOG.info('F1 by class: {:.2f} {:.2f} {:.2f}'.format(*f1s))
        # LOG.info('\n')
    
    def compare_stage(self, expert, stage=2):
        hyp = np.array(self.last_hypno)
        ld = abs(len(expert)-len(hyp))
        min_len = min(len(expert), len(hyp))
        hyp = hyp[:min_len]
        expert = expert[:min_len].astype(float)
        hyp[hyp != stage] = 0
        expert *= (stage/np.max(expert)) # zero division if wake
        coinc_mask = ((hyp == stage) & (expert == stage)).astype(int)
        coinc = int(np.sum(coinc_mask))
        exp = int(np.sum(expert)/stage)
        fr = int(np.sum((hyp == stage) & (expert != stage)).astype(int))
        # print('Accuracy = {:.2f}% - {} REM epochs detected out of {} presented in expert hypnogram (difference in length = {}); False REM = {}'.format(100*coinc/exp, coinc, exp, ld,fr))
        fn_mask = ((hyp != stage) & (expert == stage)).astype(int)
        fn = int(np.sum(fn_mask))
        return coinc, exp, fr, fn
        
    def no_rems_in_wake(self, n_back, thr=5):
        self.last_hypno = np.array(self.last_hypno)
        # print(np.where(self.last_hypno == 2)[0].size)
        res = [*self.last_hypno[:n_back]]
        i = n_back
        while i < len(self.last_hypno):
            if (self.last_hypno[i] == 2) and (np.where(np.array(res)[i-n_back:i]==0)[0].size > thr): # and (res[i-1] == 0) (np.sum(self.last_hypno[i-n_back:i]) == 0):
                res.append(0)
                j = i + 1
                if j >= len(self.last_hypno): break
                while (self.last_hypno[j] == 2):
                    res.append(0)
                    j += 1
                    if j == len(self.last_hypno): break
                i = j
                
            else: 
                res.append(self.last_hypno[i])
                i += 1
        self.last_hypno = np.array(res)
    
    def no_rems_in_wake_old(self, n_back):
        self.last_hypno = np.array(self.last_hypno)
        # print(np.where(self.last_hypno == 2)[0].size)
        res = [*self.last_hypno[:n_back]]
        i = n_back
        while i < len(self.last_hypno):
            if (self.last_hypno[i] == 2) and (res[i-1] == 0) and (np.sum(res[i-n_back:i]) == 0): # 
                res.append(0)
                j = i + 1
                if j >= len(self.last_hypno): break
                while (self.last_hypno[j] == 2):
                    res.append(0)
                    j += 1
                    if j == len(self.last_hypno): break
                i = j
                
            else: 
                res.append(self.last_hypno[i])
                i += 1
        self.last_hypno = np.array(res)
    
    def no_single_a_between_b_and_c(self, a, b, c):
        stage_dict = {0: 'Wake', 1: 'NREM', 2: 'REM'}
        res = [self.last_hypno[0]]
        count = 0
        for i in range(1, len(self.last_hypno)-1):
            if (self.last_hypno[i] == a) and (self.last_hypno[i-1] == b) and (self.last_hypno[i+1] == c):
                res.append(b)
                count += 1
            else:
                res.append(self.last_hypno[i])
        res.append(self.last_hypno[-1])
        self.last_hypno = np.array(res) 
        print(f'{count} single {stage_dict[a]}s between {stage_dict[b]}s and {stage_dict[c]}s corrected')
        
    def no_rem_low_rms(self, ):
        smooth = self.rms_smooth(self.art_rms, 60).reshape((-1, 1))
        gms = GM(n_components=2, covariance_type='full', weights_init=(.5, .5), tol=1e-6).fit_predict(smooth).astype(float)
        plt.rcParams['figure.figsize'] = (30, 6)
        centr0, centr1 = np.mean(smooth[gms==0.]), np.mean(smooth[gms==1.])
        if centr0 > centr1:
            gms -= 0.5
            gms *= -1
            gms += .5
            gms = gms.astype(int)
        sm_art_thr = np.max(smooth[gms==0]) + 1e-3
        # plt.plot(smooth)
        # plt.axhline(sm_art_thr)
        # plt.show()
        self.last_hypno[(self.last_hypno == 2) & (smooth.flatten() < sm_art_thr)] = 0          
            
        
    def stage_fraction(self, minutes):
        pred_stages = np.array(self.last_hypno).flatten()
        stages = np.unique(pred_stages)
        result = {stage: [] for stage in stages}
        interval_points = int(minutes * 60 / self.window_sec1)
        for i in range(1, ceil(len(self.last_hypno) / interval_points) + 1):
            interval_stages = self.last_hypno[(i-1)*interval_points: i*interval_points]
            for stage in stages:
                result[stage].append(np.where(interval_stages == stage)[0].size/len(interval_stages))
        return [result]

    def stage_duration(self, hours_cycle=12):
        epoch_len_sec = self.window_sec1
        interval_points = int(hours_cycle*3600 / self.window_sec1)
        pred_stages = np.array(self.last_hypno).flatten()
        result = {}
        for i in range(ceil(len(pred_stages)/interval_points) + 1):
            stats = [(stage, epoch_len_sec*len(list(seq))) for stage, seq in groupby(pred_stages[i*interval_points:(i+1)*interval_points])]
            stages = np.unique(pred_stages[i*interval_points:(i+1)*interval_points])   
            for stage, duration in stats:
                key = str(int(stage))+'cycle {}'.format(i+1)
                result.setdefault(key, [])
                result[key].append(duration)
        return [result]

    def stage_duration_report(self, fname, res_=None, return_=False, nums=[]):
        if res_ is None:
            res_ = self.stage_duration()
         # ***
        dfs = []
        for num, res in enumerate(res_):
            if len(nums) > 0: num = nums[num]
            stats = {stage: [0]*len(self.duration_ranges) for stage in np.unique(list(res.keys()))}
            for stage in np.unique(list(res.keys())):
                for dur in res[stage]:
                    for i in range(len(self.duration_ranges)):
                        if (dur >= self.duration_ranges[i][0]) and (dur < self.duration_ranges[i][1]): break
                    stats[stage][i] += 1
            res_df = pd.DataFrame(columns=('Stages #{}'.format(num), 'Duration range (sec) #{}'.format(num), 'Count #{}'.format(num)), index=range(len(stats.keys())*len(self.duration_ranges)))
            cur_row = 0
            for stage in stats.keys():
                for i, score in enumerate(stats[stage]):
                    res_df.iloc[cur_row] = (self.stage_map[int(stage[0])]+' '+stage[1:], '{}<...<={}'.format(*self.duration_ranges[i]), score)
                    cur_row += 1
            dfs.append(res_df)
        if return_:
            return pd.concat(dfs, axis=1)
        else:
            pd.concat(dfs, axis=1).to_csv(fname, index=False)

    def stage_fraction_report(self, fname, minutes, res_=None, col_names=None):
        if res_ is None:
            res_ = self.stage_fraction(minutes)
        unique_stages = np.unique(self.last_hypno)
        dfs = []
        for num, res in enumerate(res_):
            arr = np.zeros((len(res[unique_stages[0]]), len(unique_stages)))
            for i, stage in enumerate(unique_stages):
                try:
                    arr[:, i] = res[stage]
                except:
                    arr[:, i] = np.zeros(len(arr))
            # print(self.stage_map, stage)
            res_df = pd.DataFrame(columns=[self.stage_map[int(stage)] + '#{}'.format(num if col_names is None else col_names[num]) for stage in unique_stages], data=arr*100) # ???
            dfs.append(res_df)
        pd.concat(dfs, axis=1).to_csv(fname, index=False)
        
    def save_hypno(fname):
        with open(fname, 'w') as f:
            for label in self.last_hypno:
                f.write(label)
        
from scipy import signal as ss
from scipy.stats import zscore
import neo
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pandas as pd
from itertools import groupby
from sklearn.metrics import accuracy_score as acc, balanced_accuracy_score as bacc, confusion_matrix as cm, cohen_kappa_score as kappa, davies_bouldin_score as dbs, calinski_harabasz_score as chs, silhouette_score as silhs, f1_score
from sklearn.mixture import GaussianMixture as GM
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope as EE
from scipy.stats import kurtosis
import warnings
from math import ceil
import os
from ipywidgets import IntText, interact, IntSlider, Layout
from sklearn.neighbors import KernelDensity as KD
from scipy.spatial import ConvexHull, Delaunay
from sklearn.cluster import SpectralClustering as SC
import pickle
from time import perf_counter
import logging
import hvplot.pandas
import holoviews as hv
from holoviews.plotting.links import DataLink
from collections import OrderedDict as OD
import base64
from io import BytesIO
from shapely import MultiPoint, concave_hull
from shapely.geometry import Polygon, Point
import h5py
from tqdm.notebook import tqdm
from typing import NamedTuple

hv.notebook_extension('bokeh')

logging.getLogger('matplotlib').setLevel(logging.WARNING)
font = font_manager.FontProperties(family='Cambria', size=18)
plt.rcParams['font.family'] = 'Cambria'
warnings.simplefilter('ignore')
mne.set_log_level('ERROR')


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


def combine_masks(rem_mask, mask1, pps2, art_mask):
    rem_mask = 2*np.array(rem_mask)
    res_mask_ = mask1 + rem_mask[art_mask]
#     res_mask = np.zeros(len(art_mask))
#     res_mask[art_mask] = res_mask_
    res_mask_[res_mask_ > 2] = 2
    dummy = np.zeros(min(pps2*1, len(art_mask)))
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
        print('Expert hypnogram: ', os.path.join(dir_, fname), 'stages presented: ', np.unique(df.values))
        for el in df.columns:
            a, d = list(map(int, el.split('_')))
            _ = expert_rem_hypnos[fname[:-3]+DATA_EXTENSION].setdefault(a, {})
            expert_rem_hypnos[fname[:-3]+DATA_EXTENSION][a][d] = df[el].values.astype(int)
            expert_rem_hypnos[fname[:-3]+DATA_EXTENSION][a][d][expert_rem_hypnos[fname[:-3]+DATA_EXTENSION][a][d] > 2] = 0
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
        window_sec1, n_hours_cycle, delta_cluster_bands, filt_params, pre_filt, cluster_strictness, verbose = kwargs['window_sec1'], kwargs['n_hours_cycle'], kwargs['delta_cluster_bands'], kwargs['filt_params'], kwargs['pre_filt'], kwargs['cluster_strictness'], kwargs['verbose']
        scoring = Scorer(filename=os.path.join(dir_, fname), window_sec1=window_sec1, n_hours_cycle=n_hours_cycle, delta=delta_cluster_bands, filt_params=filt_params, pre_filt=pre_filt, cluster_strictness=cluster_strictness)
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
                scoring = Scorer(filename=os.path.join(dir_, fname), window_sec1=window_sec1, n_hours_cycle=n_hours_cycle, delta=delta_cluster_bands, filt_params=filt_params, pre_filt=pre_filt, verbose=verbose, cluster_strictness=cluster_strictness)
                scrngs[fname][eeg_idx].append(scoring)
                # if not cache_flag:
                scoring.pick_eeg(eeg_idx=eeg_idx, day_n=day, skip_minutes=kwargs['skip_minutes'][eeg_idx])
                scoring.prepare_rms(smooth_fft=kwargs['smooth_fft'], theta_res='ratio', add_mode='diff') 
                full_spec, freqs = scoring.fft_feats(([]), use_welch=False, log=False, manual=True, use_MT=False)                   
                if kwargs['cluster_artifact']: # тут точно не должно быть сдвинуто левее?
                    vote_dict[fname][eeg_idx].append([[], 0, None, None, red_to_interactive, None, scoring.art_rms, (full_spec, freqs), []]) # 2-scoring.mask1, 3-scoring.art_mask, 5-scoring.nr_mask
                else:
                    vote_dict[fname][eeg_idx].append([[], 0, None, None, red_to_interactive, []]) #scoring.mask1, 3-scoring.art_mask, 5-scoring.nr_mask
    return vote_dict, scoring, scrngs


INITIAL_RIDGES = {}


def prescoring_artifacts(vote_dict, scorings, **kwargs):
    global INITIAL_RIDGES
    verbose, expert_hypnos, w_humps, ep, refined_ridges, fad = kwargs['verbose'], kwargs['expert_hypnos'], kwargs['w_humps'], kwargs['ep'], kwargs['refined_ridges'], kwargs['fad']
    fnames = list(vote_dict.keys()) if fad is None else [fad[0]]
    for fn, fname in enumerate(fnames):
        print('File \"'+fname+'\"')
        eeg_ids = list(vote_dict[fname].keys()) if fad is None else [fad[1]]
        if not (fname in INITIAL_RIDGES): INITIAL_RIDGES[fname] = {}
        for eeg_idx in eeg_ids:
            print('Animal '+str(eeg_idx))                                       #перестанет работать, если много файлов !
            n_days = range(len(vote_dict[fname][eeg_idx])) if fad is None else [fad[2]]
            if not (eeg_idx in INITIAL_RIDGES[fname]): INITIAL_RIDGES[fname][eeg_idx] = []
            for day in n_days:
                print('Day '+str(day))
                scoring = scorings[fname][eeg_idx][day]
                copy_data = np.zeros(len(scoring.art_rms))
                scoring.force_art_thr = kwargs['force_art_thr'][eeg_idx]
                # определяем пороговые артефакты
                scoring.verbose = verbose
                scoring.thr_artifacts()
                if (refined_ridges is None): scoring.first_art_mask = np.array(scoring.art_mask)
                # вырезаем горбы
                if not (expert_hypnos is None): scoring.txthypno = expert_hypnos[fname][eeg_idx][day][:min(len(expert_hypnos[fname][eeg_idx][day]), len(scoring.raw_epochs))]
                scoring.set_thr2_auto(copy_data, plot=False, zbefore=kwargs['zbefore'], oafter=kwargs['oafter'], ridge_thrs=kwargs['ridge_thrs'][eeg_idx], w_hump=w_humps, refined_ridges=refined_ridges)
                to_vote = []
                failed = 0
                clusters = []
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
                    else:
                        scoring.art_mask[np.logical_not(nr_mask)] = np.logical_and(scoring.art_mask[np.logical_not(nr_mask)], mask_ep)
                    scoring.nr_mask = np.logical_and(scoring.nr_mask, scoring.art_mask)
                    nr_mask = scoring.nr_mask
                    if len(INITIAL_RIDGES[fname][eeg_idx]) < (day + 1): INITIAL_RIDGES[fname][eeg_idx].append(nr_mask)
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
            print('Animal '+str(eeg_idx))
            n_days = range(len(vote_dict[fname][eeg_idx])) if fad is None else [fad[2]]
            for day in n_days:
                print('Day '+str(day))
                scoring = scorings[fname][eeg_idx][day]
                scoring.visualize_clusters = visualize_clusters
                if (verbose > 0) and (scoring.visualize_clusters) and (len(expert_hypnos) > 0):
                    scoring.txthypno = expert_hypnos[fname][eeg_idx][day][:min(len(expert_hypnos[fname][eeg_idx][day]), len(scoring.raw_epochs))]
                scoring.scoring()
                vote_dict[fname][eeg_idx][day][2] = scoring.mask1
    return vote_dict


TO_AVE = {}


def prescoring_theta_comb(vote_dict, scoring, **kwargs):
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
            print('Animal '+str(eeg_idx))
            n_days = range(len(vote_dict[fname][eeg_idx])) if fad is None else [fad[2]]
            for day in n_days:
                random_ids = None
                TO_AVE[fname][eeg_idx].append([])
                print('Day '+str(day))
                vote_dict[fname][eeg_idx][day][0] = []
                nr_mask = vote_dict[fname][eeg_idx][day][5] #, art_mask   :7
                full_spec, freqs = vote_dict[fname][eeg_idx][day][-2]
                full_spec = np.squeeze(full_spec[nr_mask])
                delta_ = np.sum(full_spec[:, (freqs >= 2) & (freqs <= 6)], axis=1)
                CLUSTER_NAMES = []
                pbar = tqdm(range(len(delta_theta_combs)), ncols=500)
                for cl_n, (theta_range, dband) in enumerate(delta_theta_combs):
                    # print((f'Theta - {theta_range}, delta - {dband}'))
                    pbar.set_description(f'BC: {theta_range[0]}-{theta_range[1]} Hz x {dband[0]}-{dband[1]} Hz')
                    CLUSTER_NAMES.append((str(theta_range), str(dband)))
                    multiDd = full_spec[:, ((freqs >= dband[0]) & (freqs <= dband[1]))]
                    labels_up = GM(n_components=2, covariance_type='full').fit_predict(multiDd)
                    multiDd = np.sum(multiDd, axis=1)                                     # сжатая многомерная дельта
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
                        random_ids = np.random.choice(rand_borders, 10)#np.arange(10)
                    if show_plot:
                        fig = plt.figure(figsize=(44, 10)) # большой длинный график
                        gs = fig.add_gridspec(2, 6,  width_ratios=(2, 1, 2, 2, 2, 2), height_ratios=(2, 1), left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.05, hspace=0.05)
                        ax = fig.add_subplot(gs[0, 0])
                        ax_histy = fig.add_subplot(gs[0, 1], sharey=ax)
                        ax_histy.tick_params(axis="y", labelleft=False)
                        n_, b_, p_ = ax_histy.hist(multiDd, 40, orientation='horizontal', alpha=0)
                    if type(rough_sep[eeg_idx]) in (int, float): # жесткий порог для горизонтального разделения
                        appr_thr = rough_sep[eeg_idx]
                        if show_plot: ax.axhline(appr_thr, c='k')
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

                    model = GM(n_components=2, covariance_type='full').fit(rt) # старое вертикаьное разделение с двумерным Гауссом
                    labels2d = model.predict(rt)
                    left, right = rt[:, 0][labels2d == 0], rt[:, 0][labels2d == 1]
                    if np.mean(left) > np.mean(right):
                        labels2d = np.abs(labels2d - 1)
                    if show_plot:
                        ax = fig.add_subplot(gs[0, 3])
                        ax_histx = fig.add_subplot(gs[1, 3], sharex=ax)
                    # multiD
                    rt_ = rt_[labels_up == 0]
                    model1D = GM(n_components=2, covariance_type='full').fit(rt_) # новый многомерный Гаусс по тета
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
                    theta_percent = (np.percentile(rt[:, 0], 99)-np.percentile(rt[:, 0], 1))/100
                    rot_point = (0, 0)
                    shps = 0
                    fig_xlims, fig_ylims = (np.percentile(rt[:, 0], 1), np.percentile(rt[:, 0], 99)), (np.percentile(rt[:, 1], 1), np.percentile(rt[:, 1], 99))
                    yr, xr = fig_ylims[1]-fig_ylims[0], fig_xlims[1]-fig_xlims[0], 
                    yl_rot = 0
                    rot_mask = (rt[:, 1] < yl_rot).astype(int)
                    if show_plot: 
                        ax_histx = fig.add_subplot(gs[1, 5], sharex=ax)
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
                    vote_dict[fname][eeg_idx][day][0].append(masks_to_vote) # mask
                    pbar.update(1)
                # print(f'{len(vote_dict[fname][eeg_idx][day][0])} theta-delta combinations in total\n')
    return vote_dict, scoring


def spindle_check(hypno, full_spec, freqs): # TO-DO
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
    n = 0
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
    return gmm.means_[n, :2], v[0], v[1], angle, ell


def ellipse_patch2coords(ellipse):
    center = ellipse.get_center()
    width = ellipse.get_width()
    height = ellipse.get_height()
    angle = ellipse.angle  # Rotation angle in degrees
    # Parametric equations for an ellipse
    theta = np.linspace(0, 2 * np.pi, 100)  # 100 points around the circle
    a = width / 2  # Semi-major axis
    b = height / 2  # Semi-minor axis
    # Generate points on the ellipse
    x = center[0] + a * np.cos(theta)
    y = center[1] + b * np.sin(theta)
    # If the ellipse is rotated, apply the rotation
    if angle != 0:
        angle_rad = np.deg2rad(angle)  # Convert angle to radians
        x_rot = center[0] + (x - center[0]) * np.cos(angle_rad) - (y - center[1]) * np.sin(angle_rad)
        y_rot = center[1] + (x - center[0]) * np.sin(angle_rad) + (y - center[1]) * np.cos(angle_rad)
        x, y = x_rot, y_rot
    return np.column_stack((x, y))

def check_points_in_polygon(scatter_points, polygon_vertices):
    polygon = Polygon(polygon_vertices)
    points_inside = []
    for i, (x, y) in enumerate(scatter_points):
        if polygon.contains(Point(x, y)):
            points_inside.append(i) #(x, y))
    return points_inside


def process_polygon_selection(polygon_source, scatter):
    polygon_vertices = list(zip(polygon_source.data['xs'][-1], polygon_source.data['ys'][-1]))
    scatter_points = list(zip(scatter['Theta'], scatter['~Delta']))
    points_inside = check_points_in_polygon(scatter_points, polygon_vertices)
    return np.array(points_inside)


class Contour(NamedTuple):
    data: dict


def sel_scatter(df, manual=True, ellipse_magnify=1, selection=None):
    if manual:
        stage_codes = {0: 'Wake', 1: 'NREM', 2: 'REM'}
        df['stage'] = [stage_codes.get(el, '') for el in df['state'].values]
        points = df.hvplot.scatter(x="Theta", y="~Delta", c="state", by='stage', legend='top', width=700, height=500, cmap='plasma', colorbar=False, alpha=.6, hover=False).opts(tools=[], active_tools=['poly_draw']) #.opts(tools=["lasso_select", "box_select"], active_tools=["lasso_select"]) #
        # sel = hv.streams.Selection1D(source=points)
        sel = np.array([])
        poly = hv.Polygons([[]])
        source = hv.streams.PolyDraw(source=poly, drag=True, num_objects=4, show_vertices=False, styles={'fill_color': ['red', 'green', 'blue']})
        points = (points * poly).opts(hv.opts.Polygons(fill_alpha=0.3, tools=[], active_tools=['poly_draw']))
        if not (selection is None):
            xs, ys = selection.T
            points = points * hv.Polygons([{'x': xs, 'y': ys}]).opts(alpha=.5, color='r')            
    else:
        source = None
        X = df[['Theta', '~Delta']].values
        y = df['state'].values
        model = GM(n_components=1, covariance_type='full', tol=1e-6).fit(X[y==2])
        hypno_ = y
        stage_map = {0: 'Wake', 1: 'NREM', 2: 'REM'}
        colors = {'Wake': 'darkcyan', 'NREM': 'purple', 'REM': 'gold'}
        fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.7), sharey=True)
        fig.text(0.5, -.01, 'Average theta amplitude, μV', ha='center', fontsize=18, family='Cambria')
        ax = axes.flatten()[0]
        for c in range(3):
            ax.scatter(X[hypno_ == c, 0], X[hypno_ == c, 1], alpha=.5, s=10, label=stage_map[c], c=colors[stage_map[c]])                
        ax.legend(fontsize=14, handletextpad=0, borderpad=.5)
        ax.set_ylabel('Average delta amplitude, μV', fontsize=18, family='Cambria')
        ax.tick_params(axis='both', labelsize=14)
        ax = axes.flatten()[1]
        colors2 = {1: 'purple', 0: 'grey'}
        stage_map2 = {1: 'NREM', 0: 'Wake+REM'}
        for c in range(3):
            ax.scatter(X[hypno_ == c, 0], X[hypno_ == c, 1], alpha=.5, s=10, label=stage_map[c], c=colors[stage_map[c]])  
        ax.legend(fontsize=14, handletextpad=0, borderpad=.5)
        ell_params = make_ellipses(model, ax, ellipse_magnify)
        ellipse = ell_params[-1]
        ell_params = ell_params[:-1]
        ell_coords = ellipse_patch2coords(ellipse)
        source = Contour(data={'xs': [ell_coords[:, 0]], 'ys': [ell_coords[:, 1]]})
        sel = in_ellipse(X[:, 0], X[:, 1], *ell_params)
        ax.tick_params(axis='both', labelsize=14)
        plt.tight_layout()
        plt.show()
        points = None
    return points, sel, source


def precontour_init(hypno_labels, last_hypnos, fname, animal, day):
    hypno_labels.setdefault(fname, {})
    last_hypnos.setdefault(fname, {})
    hypno_labels[fname].setdefault(animal, {})
    last_hypnos[fname].setdefault(animal, {})
    hypno_labels[fname][animal].setdefault(day, {})
    last_hypnos[fname][animal].setdefault(day, {})
    return hypno_labels, last_hypnos


def prefinish_init(fracs, fragms, hypnos, saved_specs, fname, animal):
    fracs.setdefault(fname, {})
    fragms.setdefault(fname, {})
    hypnos.setdefault(fname, {})
    saved_specs.setdefault(fname, {})
    fracs[fname].setdefault(animal, {})
    fragms[fname].setdefault(animal, {})
    hypnos[fname].setdefault(animal, {})
    saved_specs[fname].setdefault(animal, {})
    return fracs, fragms, hypnos, saved_specs


def av_bcs_df(fname, animal, day, hypno_labels):
    avbcs = np.hstack([np.vstack([el[i].reshape((1, -1)) for el in TO_AVE[fname][animal][day]]).mean(axis=0)[:, None] for i in range(2)])
    hypl = np.array(hypno_labels[fname][animal][day]).reshape((-1, 1))
    return pd.DataFrame(np.hstack((avbcs, hypl)), columns=('Theta', '~Delta', 'state'))


REM_MASK = []


def end_scoring_selection(vote_dict, fname, eeg_idx, day, scoring, rule, single_nr, single_wake, spindles=None, manual_contour=True, vote_pass=slice(None, None), ellipse_magnify=1, selection=None):
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
    # ГОЛОСОВАНИЕ
    to_vote, failed, mask1, art_mask, red_to_interactive = vote_dict[fname][eeg_idx][day][:5]
    to_vote = [to_vote[i] for i in vote_pass]
    # print(f'len(to_vote) = {len(to_vote)}')
    to_vote = [mset[1] for optn, mset in enumerate(to_vote)] #mask_option[eeg_idx][optn]
    print(f"voting using {len(to_vote)} clusters. rule = {rule[eeg_idx]-failed}")
    # LOG.info(f"voting using {len(to_vote)} clusters. rule = {rule[eeg_idx]-failed}")
    mask = rem_vote(to_vote, rule=rule[eeg_idx]) # ПРАВИЛО ГОЛОСОВАНИЯ rule МОЖЕТ БЫТЬ ЦЕЛЫМ ЧИСЛОМ, КОТОРОЕ БУДЕТ ОЗНАЧАТЬ МИНИМАЛЬНОЕ КОЛИЧЕСТВО ГОЛОСОВ ДЛЯ ПРИНЯТИЯ РЕШЕНИЯ REM
    REM_MASK = mask
    # далее используется экземпляр scoring последнего дня последнего животного в последнем файле. что-то может быть некорректным
    scoring.last_hypno = combine_masks(mask, mask1, int(24*3600/scoring.window_sec1), art_mask)
    # Контекстуальные правила
    if single_nr: 
        nr_before = np.argwhere(np.array(scoring.last_hypno) == 1).size
        scoring.last_hypno = no_singles(scoring.last_hypno, 1, single_nr) # 0-1-0 -> 0-0-0; 2-1-2 -> 2-2-2
        print(f'{nr_before-np.argwhere(np.array(scoring.last_hypno) == 1).size} {int(single_nr)}-long or shorter NREM epochs corrected')
    if single_wake:
        w_before = np.argwhere(np.array(scoring.last_hypno) == 0).size
        scoring.last_hypno = no_singles(scoring.last_hypno, 0, single_wake) # 1-0-1 -> 1-1-1; 2-0-2 -> 2-2-2
        print(f'{w_before-np.argwhere(np.array(scoring.last_hypno) == 0).size} {int(single_wake)}-long or shorter Wake epochs corrected')    
    # spindles
    if spindles:
        scoring.last_hypno = spindle_check(scoring.last_hypno, full_spec, freqs)
    stages = np.unique(scoring.last_hypno)
    colors = {0: 'g', 1: 'b', 2: 'r', 4: 'purple'}
    nr_mask = vote_dict[fname][eeg_idx][day][5]
    hypno_labels = np.array(scoring.last_hypno)[nr_mask].reshape((-1, 1))
    for stcode, stname in zip((0, 1, 2), ('Wake', 'NREM', 'REM')):
        print(f'{stname} - {np.argwhere(hypno_labels.flatten() == stcode).size}; ', end='')
    print()
    av_clust = np.hstack((np.vstack([el[0].reshape((1, -1)) for el in TO_AVE[fname][eeg_idx][day]]).mean(axis=0)[:, None], np.vstack([el[1].reshape((1, -1)) for el in TO_AVE[fname][eeg_idx][day]]).mean(axis=0)[:, None]))
    df = pd.DataFrame(np.hstack((av_clust, hypno_labels)), columns=('Theta', '~Delta', 'state'))
    return sel_scatter(df, manual_contour, ellipse_magnify, selection), hypno_labels, scoring.last_hypno


def val_ridges(scoring, hypno, exp_hypno=None):
    fig = plt.figure(figsize=(25, 5))
    to_plot = np.full(len(scoring.art_rms), np.nan)
    to_plot[scoring.first_art_mask] = scoring.sm_art_rms
    plt.plot(to_plot[scoring.art_mask], c='grey', ls='--') #art_mask
    plt.title('Validation of Wake epochs reduction')
    plt.xlabel('Epoch #')
    plt.ylabel('Amplitude, a.u.')
    chunksize = 1 / len(scoring.ridge_thrs)
    for i, rthr in enumerate(scoring.ridge_thrs):
        plt.axhline(rthr, xmin=i * chunksize, xmax=(i+1)*chunksize, c='k')
    minlen = min(len(scoring.art_rms), len(hypno))
    rem_dumm = np.array(to_plot)[:minlen]
    rem_dumm = rem_dumm[scoring.art_mask[:minlen]]
    hypno = hypno[:minlen]
    hypno = hypno[scoring.art_mask[:minlen]]
    nanmask = hypno != 2
    rem_dumm[nanmask] = np.nan
    plt.scatter(np.arange(len(rem_dumm)), rem_dumm, c='gold', alpha=1, lw=2, edgecolor='gold', s=65)
    not_rem = np.array(to_plot)[:minlen]
    not_rem = not_rem[scoring.art_mask[:minlen]]
    art_nr_mask = np.array(scoring.nr_mask[:minlen])[scoring.art_mask[:minlen]]
    not_rem[art_nr_mask] = np.nan
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


def end_scoring_final(vote_dict, fnames, scoring, n_hours_cycle_bars, minutes, hours_fragm, expert_hypnos, verbose, hypno_labels_dct, last_hypnos, n_back, wake_thr, single_nrem2_1_2, single_wake2_0_2, single_r, n_back_old, save_spec_params={}, get_particular=False, save_dir='./', clear_start_rem=False, more_reduce=True, rnr_edges=False):
    to_interactive = {}
    hypnos = {}
    fragms = {}
    fracs, exp_fracs = {}, {}
    saved_specs = {}
    for fn, fname in enumerate([get_particular[0]] if get_particular else fnames):
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
                scoring.no_rems_in_wake(n_back, wake_thr)
                print(f'{rems_total-np.argwhere(scoring.last_hypno == 2).size} REMs turned to Wake by analyzing preceding states')
                # удаляем фрагментацию REM единичными эпохами других состояний
                if single_nrem2_1_2: 
                    scoring.no_single_a_between_b_and_c(1, 2, 2)
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
                fig = plt.figure(figsize=(7, 7))
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
                tmpfile = BytesIO()
                fig.savefig(tmpfile, format='png', bbox_inches='tight')
                FIGSAVE[f'{fname[:-4]}_an{eeg_idx}_d{day}_clustering'] = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                plt.show()
                red_to_interactive = vote_dict[fname][eeg_idx][day][4]
                to_interactive[fname][eeg_idx].append((scoring.last_hypno, red_to_interactive))
                if verbose > 0: 
                    cyclefigs = scoring.bars_by_cycles(n_hours_cycle=n_hours_cycle_bars, save_dir=os.path.join(save_dir, f'{fname[:-4]}_an{eeg_idx}_d{day}_fractions.png')) # соотношение стадий в циклах заданной длины
                    for cclfg in range(len(cyclefigs)): FIGSAVE[f'{fname[:-4]}_an{eeg_idx}_d{day}_cycle#{cclfg+1}'] = cyclefigs[cclfg]
                if verbose > 0:
                    if expert_hypnos is None: max_len = len(scoring.last_hypno)
                    else: max_len = max(len(scoring.last_hypno), len(expert_hypnos[fname][eeg_idx][day]))
                    if not (expert_hypnos is None):
                        fig = plt.figure(figsize=(21, 1))
                        plt.plot(expert_hypnos[fname][eeg_idx][day], lw=1)
                        plt.xlim(0, max_len)
                        plt.yticks(sorted(list(scoring.stage_map.keys()))[:3], labels=[scoring.stage_map[k] for k in sorted(list(scoring.stage_map.keys()))[:3]])
                        plt.ylabel('State', fontsize=15)
                        plt.xlabel('Epoch #', fontsize=15)
                        tmpfile = BytesIO()
                        fig.savefig(tmpfile, format='png', bbox_inches='tight')
                        FIGSAVE[f'{fname[:-4]}_an{eeg_idx}_d{day}_expert_hypnogram'] = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                        plt.show()
                    fig = plt.figure(figsize=(21, 1))
                    ax = plt.subplot(111)
                    ax.plot(scoring.last_hypno, lw=1, c='grey')
                    stages = np.unique(scoring.last_hypno)
                    colors = {0: 'darkcyan', 1: 'purple', 2: 'goldenrod', 4: 'purple'}
                    ax.set_yticks(sorted(list(scoring.stage_map.keys()))[:len(stages)])
                    ax.set_yticklabels([scoring.stage_map[k] for k in sorted(list(scoring.stage_map.keys()))[:len(stages)]])
                    ax.set_ylabel('State', fontsize=15)
                    ax.set_xlim(0, len(scoring.last_hypno))
                    ax.set_xlabel('Epoch #', fontsize=15)
                    ax.set_ylim(-.1, 2.25)
                    tmpfile = BytesIO()
                    fig.savefig(tmpfile, format='png', bbox_inches='tight')
                    FIGSAVE[f'{fname[:-4]}_an{eeg_idx}_d{day}_hypnogram'] = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                    plt.show()
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
                    colors = {0: 'darkcyan', 1: 'purple', 2: 'r', 4: 'purple'}
                    ax = plt.subplot(111)
                    ax.set_ylabel(f'Power at {band[0]}-{band[1]} Hz [${unit}^2$/{freqs[1]-freqs[0]:.2f} Hz]')
                    ax.set_xlabel('Epoch #', fontsize=15)
                    for stage in stages:
                        stage_mask = scoring.last_hypno == stage
                        ax.bar(np.argwhere(stage_mask).flatten(), spec_save[stage_mask], width=5, color=colors[stage])
                        if (stage == 1) and (save_spec_params['spec_max'].get(eeg_idx, {}).get(b, None) is None): ax.set_ylim(ax.get_ylim()[0], np.max(spec_save[stage_mask])*1.1)
                        else: ax.set_ylim(0, save_spec_params['spec_max'].get(eeg_idx, {}).get(b, None))
                    ax.bar(np.argwhere(np.logical_not(artifacts)).flatten(), spec_save[np.logical_not(artifacts)], width=5, color='yellow')
                    ax.legend([scoring.stage_map[stage] for stage in stages]+['Artifacts'])
                    ax.set_xlim(0, len(spec_save))
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
                if more_reduce:
                    seqs012 = [(el[0], len(list(el[1]))) for el in groupby(refined_ridges_mask)]
                    curidx = seqs012[0][1]
                    for i in range(1, len(seqs012)-1):
                        code, lenlst = seqs012[i]
                        # if cut_left:
                        #     if (code == 0) and all([el in (1, 2) for el in (seqs012[i-1][0], seqs012[i+1][0])]) and (not all([el == 2 for el in (seqs012[i-1][0], seqs012[i+1][0])])):#and (seqs012[i-1][0] != seqs012[i+1][0]):
                        #         refined_ridges[curidx: curidx+lenlst] = False
                        # else:
                        if ((code == 0) and all([el in (1, 2) for el in (seqs012[i-1][0], seqs012[i+1][0])]) and (not all([el == 1 for el in (seqs012[i-1][0], seqs012[i+1][0])]))):# and (seqs012[i-1][0] == 1) and (seqs012[i+1][0] == 2):
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
                        fig, axes = plt.subplots(3, 1, figsize=(21, 7), sharex=True)
                        for i, stage in enumerate(sorted(list(dct.keys()))):
                            ax = axes.flatten()[i]
                            ax.plot(np.array(dct[stage])*100, label=f'{scoring.stage_map[int(stage)]} auto')
                            ax.plot(np.array(dct_exp[stage])*100, label=f'{scoring.stage_map[int(stage)]} expert')
                            ax.legend(fontsize=14, handletextpad=.5, borderpad=.5)
                            ax.set_ylabel('Stage presence, %', fontsize=18)
                            ax.tick_params(axis='both', labelsize=14)
                        ax.set_xlim(0, len(dct[stage])-1)
                        ax.set_xlabel('Hour', fontsize=18)
                        plt.tight_layout()
                        tmpfile = BytesIO()
                        fig.savefig(tmpfile, format='png', bbox_inches='tight')
                        FIGSAVE[f'{fname[:-4]}_an{eeg_idx}_d{day}_profiles.png'] = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                        plt.show()
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
                res_ids[fname][eeg_idx] += 1
    return resfrac, resfragm, reshypn, ressasp, res_ids


def save_results(fnames, scoring, fracs, fragms, hypnos, saved_specs, minutes, res_ids, suffixes={}, save_dir='./'):
    for fname in fnames:
        an_names = [str(el) + f'{suffixes.get(fname, {}).get(el, "")}' for el in sorted(list(res_ids[fname]))]
        an_ds = [f'_an{an}_d{d}' for an, eeg_idx in zip(an_names, sorted(list(res_ids[fname]))) for d in range(res_ids[fname][eeg_idx])]# saved_specs[fname]
        scoring.stage_fraction_report(os.path.join(save_dir, fname[:-4]+'_fracs.csv'), minutes, fracs[fname], col_names=an_ds) 
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
    for idx in selection:#.index:
        if df['state'].iloc[idx] == 2:
            dists = {c: np.linalg.norm(v-df[['Theta', '~Delta']].iloc[idx].values) for c, v in centroids.items()}
            if dists[0] < dists[1]:
                relabel.append(0)
            else:
                relabel.append(1)
        else:
            relabel.append(df['state'].iloc[idx])
    return np.array(relabel)


METRICS = {}


class Scorer:
    """
    """
    def __init__(self, filename, window_sec1, n_hours_cycle, window_sec2=None, delta=(3, 4), filt_params={'length': 1501, 'transition': 0.6, 'window': 'kaiser'}, add_bands=(), pre_filt=None, cluster_strictness=0.5, verbose=0, ch_id='eeg'):
        # add **kwargs
        self.delta_refs_copy = None
        self.pre_filt = pre_filt
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
        self.bands = {'delta': delta}
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
        # print(self.extension, self.reader_map)
        if self.extension in self.reader_map:
            reader = self.reader_map[self.extension](self.fname, try_signal_grouping=False)  # describe how to change for any other reader
            self.data = reader.read(lazy=False)[0].segments[0]

            for i in range(len(reader.raw_annotations['blocks'][0]['segments'][0]['signals'])):
                ann = reader.raw_annotations['blocks'][0]['segments'][0]['signals'][i]
                # print(ann)
                chs = ann['__array_annotations__']['channel_names']
                # print(chs)
                for j in range(len(chs)):
                    if (ch_id in chs[j].lower()) or ('ch' in chs[j].lower()):
                        self.eeg_map.append((i, j))
        elif self.extension == 'hdf':
            f = h5py.File(self.fname, 'r')
            self.data = f['eeg']
            self.sf = float(f['eeg'].attrs['sf'][0])
            chs = f['eeg'].attrs['ch_names']
            for j, ch in enumerate(chs):
                if (ch_id in ch.lower()) or ('ch' in ch.lower()):
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
    
    def prepare_rms(self, theta_res='ratio', add_mode='diff', smooth_fft=False):
        print('Preparing data...', end='')
        self.sf = int(self.sf)
        self.raw_epochs = np.vstack([self.eeg[i*self.sf*self.window_sec1:(i+1)*self.sf*self.window_sec1].reshape((1, -1)) for i in range(len(self.eeg)//(self.sf*self.window_sec1))])
        if self.pre_filt:
            print('pre-filtering... ', end='')
            self.eeg = self.fir(self.pre_filt, 'highpass')
        
            
        spec_abs, freq = self.fft_feats([], manual=True)#[zscore(np.log10(el), axis=0) for el in ]
        minld = np.min([el[0] for el in self.bands['delta']])
        maxld = np.max([el[1] for el in self.bands['delta']])
        if smooth_fft:
            print('smoothing deltas... ', end='')
            slc = (freq >= minld) & (freq <= maxld)
            t1 = perf_counter()
            spec_abs[:, slc] = np.apply_along_axis(self.rms_smooth, 0, spec_abs[:, slc], *(smooth_fft, ))
            print('Done')
            # print(perf_counter()-t1)
        self.delta_clusters = [spec_abs[:, (freq >= band[0])&(freq <= band[1])] for band in self.bands['delta']]
        
        self.thetas = []
        self.ratios = []
        self.ratio = [] #self.ratios[0]
        # self.delta_ref = self.rms_smooth(self.rms(self.fir((4, 5), 'bandpass'), self.window_sec2), 20, ws=self.window_sec2)
        
        
        self.delta34 = [] #self.rms_smooth(self.rms(self.fir((3, 4), 'bandpass'), self.window_sec1), 30, ws=self.window_sec1)
                
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
    
    def set_thr2_auto(self, ratio, ymax=None, plot=False, mask=None, w_hump=150, zbefore=10, oafter=10, ridge_thrs=None, refined_ridges=None):
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
        print(f'{np.argwhere(gms == 0).size} epochs removed in total by ridges')
        
    def cluster_vote(self, deltas, cycle_slice, art_mask):
        gms = []
        fishers = []
        to_ave = []
    
        for i, delta in enumerate(deltas):
            gm_model = GM(n_components=2, covariance_type='full', reg_covar=1e-3, weights_init=self.weights).fit(delta)
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
                                
        cluster_preds = np.hstack([predict.reshape((-1, 1)) for predict in gms])
        vote = cluster_preds.mean(axis=1)
        # print('Fisher =', np.mean(fishers))
        return (vote > self.cluster_w_nr_rule).astype(int)#.round()
    
    
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
            mask1 = self.cluster_vote([cl[cycle_slice][art_masks[i]] for cl in  self.delta_clusters], cycle_slice, art_masks[i])
            mask1 = mask1.astype(int)
            result_mask_cycles.extend(mask1)
        self.mask1 = np.array(result_mask_cycles).flatten()   
        print()
        
    
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

    
    def load_txthypno(self, fname):
        with open(fname, 'r') as f: # открываем гипнограмму эксперта
            self.txthypno = np.array([int(el) for el in f.readlines()]).flatten()
        self.txthypno[self.txthypno > 4] = 0
        try:
            self.txthypno = self.txthypno[:min(len(self.txthypno), len(self.ratio))]
        except:
            self.txthypno = self.txthypno[:min(len(self.txthypno), len(self.ratios[0]))]
    
    
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
        
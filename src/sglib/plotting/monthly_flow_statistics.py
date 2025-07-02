import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def init_plotting():
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (14, 14)
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.labelsize'] = 1.1 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.1 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color, linestyle='solid')
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color='k')

def boxplots(syn, hist, xticks=True, legend=True, loc='upper right'):
    bpl = plt.boxplot(syn, positions=np.arange(1, len(syn[0]) + 1) - 0.15, sym='', widths=0.25, patch_artist=True)
    bpr = plt.boxplot(hist, positions=np.arange(1, len(hist[0]) + 1) + 0.15, sym='', widths=0.25, patch_artist=True)
    set_box_color(bpl, 'lightskyblue')
    set_box_color(bpr, 'lightcoral')
    plt.plot([], c='lightskyblue', label='Synthetic')
    plt.plot([], c='lightcoral', label='Historical')
    plt.gca().xaxis.grid(False)
    if xticks:
        points = range(1, len(syn[0]) + 1)
        plt.gca().set_xticks(points)
        plt.gca().set_xticklabels(points)
    else:
        plt.gca().set_xticks([])
    if legend:
        plt.legend(ncol=2, loc=loc)

def plot_validation(H_df, S_df, 
                    scale='weekly', 
                    sitename='Site',
                    logspace=False,
                    fname=None):
    if not isinstance(H_df, pd.DataFrame) or not isinstance(S_df, pd.DataFrame):
        raise TypeError("H_df and S_df must be pandas DataFrames")

    init_plotting()
    assure_path_exists('./figures/')
    sitefile = re.sub(r'\W+', '_', sitename)


    ### Data prep and reorganization
    # Based on the timescale, we want to aggregate the data
    # Then, reformat each so that it is in shape: (n_realizations, n_years, n_periods)
    # For historic data, we will resample the data n_realizations times to match the number of synthetic realizations
    if scale == 'monthly':
        H = H_df.resample('MS').sum()
        S = S_df.resample('MS').sum()
        
        # Reorganize to be shape (n_years, n_periods)
        H = H.pivot_table(index=H.index.year, columns=H.index.month, values=H.columns[0]).values
        
        # S has shape (n_years * 12, n_realizations)
        # we want it in shape (n_realizations, n_years, 12)
        S = (S.T
                .groupby(S.index.year, axis=1)
                .apply(lambda x: x.values)
                .values)
        S = np.stack(list(S), axis=1) 
        
        
    # For historic data, 
    # we want to resample so that it has the same number of 
    # realizations as the synthetic data
    n_realizations_S = S.shape[0]
    n_years_H = H.shape[0]
    idx = np.random.choice(n_years_H, size=(n_realizations_S, n_years_H), 
                           replace=True)
    # Make H_resamp so that it is shape (n_realizations, n_years, n_periods)
    H_resamp = H[idx]
    
        
    H_proc = np.log(np.clip(H, a_min=1e-6, a_max=None)) if logspace else H
    S_proc = np.log(np.clip(S, a_min=1e-6, a_max=None)) if logspace else S

    time_dim = H_proc.shape[1]

    fig = plt.figure()

    ax = fig.add_subplot(5, 1, 1)
    boxplots(S_proc.reshape((np.shape(S_proc)[0]*np.shape(S_proc)[1], 12)), H_proc, xticks=False, legend=True)
    ax.set_ylabel('Log(Q)' if logspace else 'Q')

    ax = fig.add_subplot(5, 1, 2)
    boxplots(S_proc.mean(axis=0), H_resamp.mean(axis=0), xticks=False, legend=False)
    ax.set_ylabel('$\hat{\mu}_Q$')

    ax = fig.add_subplot(5, 1, 3)
    boxplots(S_proc.std(axis=0), H_resamp.std(axis=0), xticks=False, legend=False)
    ax.set_ylabel('$\hat{\sigma}_Q$')

    stat_pvals = np.zeros((2, time_dim))
    for i in range(time_dim):
        stat_pvals[0, i] = stats.ranksums(H_proc[:, i], S_proc.reshape((np.shape(S_proc)[0]*np.shape(S_proc)[1], 12))[:, i])[1]
        stat_pvals[1, i] = stats.levene(H_proc[:, i], S_proc.reshape((np.shape(S_proc)[0]*np.shape(S_proc)[1], 12))[:, i])[1]

    ax = fig.add_subplot(5, 1, 4)
    ax.bar(np.arange(1, time_dim + 1), stat_pvals[0], facecolor='0.7', edgecolor='None')
    ax.plot([0, time_dim + 1], [0.05, 0.05], color='k')
    ax.set_xlim([0, time_dim + 1])
    ax.set_ylabel('Wilcoxon $p$')

    ax = fig.add_subplot(5, 1, 5)
    ax.bar(np.arange(1, time_dim + 1), stat_pvals[1], facecolor='0.7', edgecolor='None')
    ax.plot([0, time_dim + 1], [0.05, 0.05], color='k')
    ax.set_xlim([0, time_dim + 1])
    ax.set_ylabel('Levene $p$')

    fig.suptitle(('Log space' if logspace else 'Real space') + f' ({sitename})')
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)
    plt.close(fig)
"""
Trevor Amestoy

Contains a Hidden Markov Model generator.
"""

from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as ss
import statsmodels.api as sm

from sglib.core.base import Generator
from sglib.utils import common_default_kwargs, hmm_default_kwargs
from sglib.utils.assertions import set_model_kwargs, update_model_kwargs
from sglib.plotting.hmm_plots import plotDistribution, plotTimeSeries
epsilon = 1e-6

def deseasonalize_data(x, timestep='M',
                       standardize=True):
    # if timestep in ['M', 'MS']:
    #     group_idx = x.index.month
    #     n_annual_timesteps = 12
    # elif timestep in ['W', 'WS']:
    #     group_idx = x.index.week
    #     n_annual_timesteps = 52
    # # center on zero
    # if standardize:
    #     timestep_means = x.groupby(group_idx).mean()
    #     timestep_std = x.groupby(group_idx).std()
    #     for t in range(1, n_annual_timesteps + 1):
    #         x.loc[group_idx == t] = (x.loc[group_idx == t] - timestep_means[t]) / timestep_std[t]
    # else:
    
    result = sm.tsa.seasonal_decompose(x, model='multiplicative')
    x = x / result.seasonal

    # fill na with near
    return x, result


def reseasonalize_data(x, seasonal_result, 
                       timestep='M'):
    # if timestep in ['M', 'MS']:
    #     group_idx = x.index.month
    #     n_annual_timesteps = 12
    # elif timestep in ['W', 'WS']:
    #     group_idx = x.index.week
    #     n_annual_timesteps = 52
    # for m in range(1, n_annual_timesteps + 1):
    #     x.loc[group_idx == m] = x.loc[group_idx == m] * timestep_std[m] + timestep_means[m]
    x = x * seasonal_result.seasonal    
    return x


class HMM():
    """
    Hidden Markov Model for generating synthetic timeseries data.
    
    Methods:
        fit: Fit the model to the data.
        generate: Generate synthetic data from the fitted model.
        plotDistribution: Plot the distribution of the fitted model.
        plotTimeSeries: Plot the time series of the fitted model.
    """
    def __init__(self, Q, **kwargs):
        
        self.Q_obs = Q.copy()
        self.Q_train = Q.copy()
        
        ## Kwargs
        set_model_kwargs(self, hmm_default_kwargs, **kwargs)
        
        
    def preprocessing(self, **kwargs):
        """
        Preprocesses the input time series data. 
        """

        if self.deseasonalize:
            self.Q_train, self.seasonal_result = deseasonalize_data(self.Q_train, 
                                                                    timestep=self.timestep)
        self.Q_train = self.Q_train.dropna()
        self.Q_train = self.Q_train.loc[self.Q_train > 0]
        
        if self.log_transform:
            self.Q_train = np.log(self.Q_train)
        self._is_preprocessed = True
        

    
    def fit(self, **kwargs):
        if not self._is_preprocessed:
            print('Preprocessing data.')
            self.preprocessing()
        Q = self.Q_train.copy()
        if type(Q) == pd.core.frame.DataFrame:
            Q = Q.values.flatten()
        elif type(Q) == pd.core.series.Series:
            Q = Q.values
        elif type(Q) == np.ndarray:
            Q = Q.flatten()
            
        # fit Gaussian HMM to Q
        model = GaussianHMM(n_components=self.n_hidden_states, 
                            n_iter=self.max_iter, 
                            tol=self.tolerance).fit(np.reshape(Q,[len(Q),1]))
        self._is_fit = True
        
        # classify each observation as state 0 or 1
        hidden_states = model.predict(np.reshape(Q,[len(Q),1]))
    
        # find parameters of Gaussian HMM
        mus = np.array(model.means_)
        sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[i]) for i in range(self.n_hidden_states)])))
        P = np.array(model.transmat_)
    
        # find log-likelihood of Gaussian HMM
        logProb = model.score(np.reshape(Q,[len(Q),1]))
    
        # re-organize mus, sigmas and P so that first row is lower mean (if not already)
        if mus[0] > mus[1]:
            mus = np.flipud(mus)
            sigmas = np.flipud(sigmas)
            P = np.fliplr(np.flipud(P))
            hidden_states = 1 - hidden_states
        
        self.hidden_states = hidden_states
        self.mus = mus
        self.P = P
        self.sigmas = sigmas
        self.logProb = logProb
        self.model = model
        return
    
    def generate(self, **kwargs):
        if not self._is_preprocessed:
            print('Preprocessing data.')
            self.preprocessing()
        
        if not self._is_fit:
            print('Fitting model to historic data.')
            self.fit()   
        
        # Set kwargs if provided
        # update_model_kwargs(self, hmm_default_kwargs, **kwargs)
        
        # Generate n_realizations from Gaussian HMM
        X_syn = np.zeros((self.n_timesteps, self.n_realizations))
        X_syn_states = np.zeros((self.n_timesteps, self.n_realizations))
        for i in range(self.n_realizations):
            synthetic_timeseries, synthetic_states = self.model.sample(self.n_timesteps)
            X_syn[:,i] = synthetic_timeseries.flatten()
            X_syn_states[:,i] = synthetic_states.flatten()
            
        # Arrange in DF
        start_year = self.Q_obs.index.year[0]
        start_date = f'{start_year}-01-01'
        syn_datetime_index = pd.date_range(start=start_date, 
                                       periods=self.n_timesteps, 
                                       freq=self.timestep)
        X_syn = pd.DataFrame(X_syn, index=syn_datetime_index,
                             columns=[f'realization_{i}' for i in range(self.n_realizations)])
            
        self.X_syn = X_syn.copy()
        self.X_syn_states = X_syn_states
        
        # Transform back to original scale
        if self.log_transform:
            X_syn = np.exp(X_syn)
        if self.deseasonalize:
            X_syn = reseasonalize_data(X_syn, self.seasonal_result,
                                       timestep=self.timestep)

            



        self.Q_syn = X_syn
        return self.Q_syn
    
    
    # def plot(self, kind='line', **kwargs):
    #     if not self._is_fit:
    #         print('Fitting model to historic data.')
    #         self.fit()   
        
    #     # Plot 
    #     if kind == 'line':
    #         plotTimeSeries(self, **kwargs)
    #     elif kind == 'dist':
    #         plotDistribution(self, **kwargs)
    #     return
        

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as ss

from sglib.utils.kwargs import default_plot_kwargs


def plotDistribution(self, n_bins = 100):
    # check if pandas and convert to numpy
    if type(self.Q_train) == pd.core.frame.DataFrame:
        X = self.Q_train.values.flatten()
    elif type(self.Q_train) == pd.core.series.Series:
        X = self.Q_train.values
    elif type(self.Q_train) == np.ndarray:
        X = self.Q_train.flatten()
        
    # calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(self.P))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])

    x_0 = np.linspace(self.mus[0]-4*self.sigmas[0], self.mus[0]+4*self.sigmas[0], 10000)
    fx_0 = pi[0]*ss.norm.pdf(x_0,self.mus[0],self.sigmas[0])

    x_1 = np.linspace(self.mus[1]-4*self.sigmas[1], self.mus[1]+4*self.sigmas[1], 10000)
    fx_1 = pi[1]*ss.norm.pdf(x_1,self.mus[1],self.sigmas[1])

    x = np.linspace(self.mus[0]-4*self.sigmas[0], self.mus[1]+4*self.sigmas[1], 10000)
    fx = pi[0]*ss.norm.pdf(x,self.mus[0],self.sigmas[0]) + \
        pi[1]*ss.norm.pdf(x,self.mus[1],self.sigmas[1])

    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200)
    
    ax.hist(X, 
            color='k', alpha=0.8, density=True, bins= n_bins)
    
    for i in range(self.n_hidden_states):
        if i == 0:
            c = 'peru'
        elif i == 1:
            c = 'royalblue'
        else:
            c = 'green'
        x = np.linspace(self.mus[i]-4*self.sigmas[i], self.mus[i]+4*self.sigmas[i], 10000)
        fx = pi[i]*ss.norm.pdf(x, self.mus[i], self.sigmas[i])
        print(f'Plotting PDF of state {i+1}')
        ax.plot(x, fx, 
                linewidth=4, label=f'State {i+1} Dist.', color = c)
    

    low_state_index = np.argmin(self.mus)
    high_state_index = np.argmax(self.mus)
    
    x_combined = np.linspace(self.mus[low_state_index]-4*self.sigmas[low_state_index], 
                                self.mus[high_state_index]+4*self.sigmas[high_state_index], 10000)
    
    for i in range(self.n_hidden_states):
        if i == 0:
            fx_combined = pi[0]*ss.norm.pdf(x_combined,self.mus[0],self.sigmas[0])
        else:
            fx_combined = fx_combined + pi[i]*ss.norm.pdf(x_combined,self.mus[i],self.sigmas[i])
    
    ax.plot(x_combined, fx_combined, c='k', linewidth=3, label='Combined State Distn')

    fig.subplots_adjust(bottom=0.15)
    handles, labels = plt.gca().get_legend_handles_labels()
    matplotlib.rc('legend', fontsize = 16)
    plt.legend() 
    
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    
    plt.show()
    # plt.close()
    
    return


def plotTimeSeries(self, ylabel = 'Flow', n_bins = 50, date_range=None):

    sns.set_theme(style='white')
    start_year = int(self.Q_train.index.year[0])

    # check if pandas and convert to numpy
    if type(self.Q_train) == pd.core.frame.DataFrame:
        X = self.Q_train.values.flatten()
        ts = self.Q_train.index
    elif type(self.Q_train) == pd.core.series.Series:
        X = self.Q_train.values
        ts = self.Q_train.index
    elif type(self.Q_train) == np.ndarray:
        X = self.Q_train.flatten()
        ts=np.arange(start_year, start_year+len(X))
    
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(13, 6.5), dpi = 200, 
                            sharey=True, gridspec_kw={'width_ratios':[5,1]})
    
    for i in range(self.n_hidden_states):
        if i == 0:
            c = 'peru'
        elif i == 1:
            c = 'royalblue'
        else:
            c = 'green'
            
        hidden_states = self.model.predict(np.reshape(X,[len(X),1]))
        masks = hidden_states == i
                    
        # Plot distribution in second plot
        ax[1].hist(X[masks], bins = n_bins, 
                    color=c, label = f'State {i+1}', 
                    orientation = 'horizontal', density = True, 
                    alpha = 0.95)
        
        # Plot scatter of hidden states
        ax[0].scatter(ts[masks], X[masks], color=c, label=f'State {i+1}')
    
    ax[0].plot(ts, X, c='k', linewidth = 0.75, label = 'Observed Flow')
    ax[0].set_xlabel('Year',fontsize=16)
    # xticks = ax[0].get_xticks()
    # xticks = [int(x) for x in xticks]

    # make sure xticks are within range of ts indices
    # xticks = [x for x in xticks if x < len(ts)]
    # ax[0].set_xticklabels(ts[xticks])
    
    ax[0].set_ylabel(ylabel,fontsize=16)
    ax[0].legend(loc = 'upper left', fontsize =16, framealpha = 1)
    
    ax[1].set_xlabel('Modeled Hidden States')
    ax[1].legend(loc = 'lower center', fontsize = 16)
    
    fig.subplots_adjust(bottom=0.2)

    # matplotlib.rc('legend', fontsize = 16)
    plt.legend() 

    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)

    plt.show()
    plt.close()

    return None

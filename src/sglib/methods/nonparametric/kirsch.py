import numpy as np
import pandas as pd
import warnings

from sglib.core.base import Generator

class KirschGenerator(Generator):
    def __init__(self, Q: pd.DataFrame, 
                 generate_using_log_flow=True, 
                 matrix_repair_method='spectral', 
                 debug=False):
        if not isinstance(Q, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if not isinstance(Q.index, pd.DatetimeIndex):
            raise TypeError("Input index must be a pd.DatetimeIndex.")

        self.Q = Q.copy()
        self.params = {
            'generate_using_log_flow': generate_using_log_flow,
            'matrix_repair_method': matrix_repair_method,
            'debug': debug,
        }

        self.fitted = False
        self.n_historic_years = Q.index.year.nunique()
        self.n_sites = Q.shape[1]
        self.site_names = Q.columns.tolist()
        self.n_months = 12

        self.U_site = {}
        self.U_prime_site = {}

    def _get_synthetic_index(self, n_years):
        return pd.date_range(start=f"{self.Q.index.year.max() + 1}-01-01", periods=n_years * self.n_months, freq='MS')

    def preprocessing(self, timestep='monthly'):
        if timestep != 'monthly':
            raise NotImplementedError("Currently only monthly timestep is supported.")

        monthly = self.Q.groupby([self.Q.index.year, self.Q.index.month]).sum()
        monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=['year', 'month'])
        self.Qm = monthly

        if self.params['generate_using_log_flow']:
            self.Qm = np.log(self.Qm.clip(lower=1e-6))



    def fit(self):
        self.mean_month = self.Qm.groupby(level='month').mean()
        self.std_month = self.Qm.groupby(level='month').std()

        years = self.Qm.index.get_level_values('year').unique()
        Z_h = []
        valid_years = []

        for year in years:
            try:
                year_data = []
                for m in range(1, 13):
                    row = ((self.Qm.loc[(year, m)] - self.mean_month.loc[m]) / self.std_month.loc[m]).values
                    year_data.append(row)
                Z_h.append(year_data)
                valid_years.append(year)
            except KeyError:
                continue

        self.Z_h = np.array(Z_h)  # shape: (n_years, 12, n_sites)
        self.historic_years = np.array(valid_years)
        self.n_historic_years = len(valid_years)

        self.Y = self.Z_h.copy()
        self.Y_prime = np.zeros_like(self.Y[:-1])
        self.Y_prime[:, :6, :] = self.Y[:-1, 6:, :]
        self.Y_prime[:, 6:, :] = self.Y[1:, :6, :]

        for s in range(self.n_sites):
            y_s = self.Y[:, :, s]           # shape: (n_years, 12)
            y_prime_s = self.Y_prime[:, :, s]

            corr_s = np.corrcoef(y_s.T)     # shape: (12, 12)
            corr_prime_s = np.corrcoef(y_prime_s.T)

            self.U_site[s] = self._repair_and_cholesky(corr_s)
            self.U_prime_site[s] = self._repair_and_cholesky(corr_prime_s)

        self.fitted = True

    def _repair_and_cholesky(self, corr):
        """
        Given a correlation matrix, 'repair' the matrix so it is PSD, and return its Cholesky decomposition.
        
        
        Parameters
        ----------
        corr : np.ndarray
            Correlation matrix to be repaired and decomposed.
        
        Returns
        -------
        np.ndarray
            Cholesky decomposition of the correlation matrix.
        """
        try:
            return np.linalg.cholesky(corr).T
        except np.linalg.LinAlgError:
            repaired_corr = corr.copy()
            print("WARNING: Matrix not positive definite, repairing... This may cause correlation inflation.")
            for _ in range(50):
                evals, evecs = np.linalg.eigh(repaired_corr)
                evals[evals < 0] = 1e-8
                repaired_corr = evecs @ np.diag(evals) @ evecs.T
                np.fill_diagonal(repaired_corr, 1.0)
                try:
                    return np.linalg.cholesky(repaired_corr).T
                except np.linalg.LinAlgError:
                    continue
            raise ValueError("Matrix is not positive definite after 50 iterations.")

    def _get_bootstrap_indices(self, n_years, max_idx=None):
        """
        Return 'M', a matrix of bootstrap indices for the synthetic time series.

        Parameters
        ----------
        n_years : int
            Number of years for which to generate bootstrap indices.
        max_idx : int, optional
            Maximum index for the historic years. If None, uses the number of historic years.
        
        Returns
        -------
        np.ndarray
            A matrix of shape (n_years, n_months) containing bootstrap indices.
        """
        max_idx = self.n_historic_years if max_idx is None else max_idx
        return np.random.choice(max_idx, size=(n_years, self.n_months), replace=True)

    def _create_bootstrap_tensor(self, M, use_Y_prime=False):
        """
        Create the 'Z' tensor of boostrapped standardized flows.
        
        Parameters
        ----------
        M : np.ndarray
            Bootstrap indices with shape (n_years, n_months).
        use_Y_prime : bool
            If True, uses Y_prime; otherwise uses Y.
        
        Returns
        -------
        np.ndarray
            Bootstrap tensor with shape (n_years, n_months, n_sites).
        """
        source = self.Y_prime if use_Y_prime else self.Y
        n_years, n_months = M.shape
        max_idx = source.shape[0]
        output = np.zeros((n_years, n_months, self.n_sites))
        for i in range(n_years):
            for m in range(n_months):
                h_idx = M[i, m]
                if h_idx >= max_idx:
                    h_idx = max_idx - 1
                output[i, m] = source[h_idx, m]
        return output

    def _combine_Z_and_Z_prime(self, Z, Z_prime):
        """
        Combine Z and Z_prime into a single tensor, to preserve intra-year correlations.
        
        Parameters
        ----------
        Z : np.ndarray
            Standardized flows for the first half of the year with shape (n_years, n_months, n_sites).
        Z_prime : np.ndarray
            Standardized flows for the second half of the year with shape (n_years, n_months, n_sites).
        
        Returns
        -------
        np.ndarray
            Combined standardized flows with shape (n_years, n_months, n_sites).
        """
        n_years = min(Z.shape[0], Z_prime.shape[0]) - 1
        ZC = np.zeros((n_years, self.n_months, self.n_sites))
        ZC[:, :6, :] = Z_prime[:n_years, 6:, :]
        ZC[:, 6:, :] = Z[1:n_years+1, :6, :]
        return ZC

    def _destandardize_flows(self, Z_combined):
        """
        Reapply mean and standard deviation to standardized flows.
        
        Parameters
        ----------
        Z_combined : np.ndarray
            Standardized flows with shape (n_years, n_months, n_sites).
            
        Returns
        -------
        np.ndarray
            Destandardized flows with shape (n_years, n_months, n_sites).
        """
        Q_syn = np.zeros_like(Z_combined)
        for m in range(self.n_months):
            Q_syn[:, m, :] = Z_combined[:, m, :] * self.std_month.iloc[m].values + self.mean_month.iloc[m].values
        return Q_syn

    def _reshape_output(self, Q_syn):
        return Q_syn.reshape(-1, self.n_sites)

    def generate_single_series(self, 
                               n_years, 
                               M=None, 
                               as_array=True, 
                               synthetic_index=None):
        """
        Generate a single synthetic time series.
        
        Parameters
        ----------
        n_years : int
            Number of years for the synthetic time series.
        M : np.ndarray, optional
            Bootstrap indices for the synthetic time series. If None, random indices will be generated.
        as_array : bool
            If True, returns a numpy array; if False, returns a pandas DataFrame.
        synthetic_index : pd.DatetimeIndex, optional
            Custom index for the synthetic time series. If None, a default index will be generated.
        
        Returns
        -------
        np.ndarray or pd.DataFrame
            Synthetic time series data.
        """
        n_years_buffered = n_years + 1
        if not self.fitted:
            raise RuntimeError("Call preprocessing() and fit() before generate().")

        if M is None:
            M = self._get_bootstrap_indices(n_years_buffered, max_idx=self.Y.shape[0])
        else:
            M = np.asarray(M)
            if M.shape != (n_years_buffered, self.n_months):
                raise ValueError(f"M must have shape ({n_years_buffered}, {self.n_months})")

        M_prime = M[:self.Y_prime.shape[0], :]

        X = self._create_bootstrap_tensor(M, use_Y_prime=False)
        X_prime = self._create_bootstrap_tensor(M_prime, use_Y_prime=True)

        Z = np.zeros_like(X)
        Z_prime = np.zeros_like(X_prime)

        for s in range(self.n_sites):
            Z[:, :, s] = X[:, :, s] @ self.U_site[s]
            Z_prime[:, :, s] = X_prime[:, :, s] @ self.U_prime_site[s]


        ZC = self._combine_Z_and_Z_prime(Z, Z_prime)
        Q_syn = self._destandardize_flows(ZC)

        if self.params['generate_using_log_flow']:
            Q_syn = np.exp(Q_syn)

        Q_flat = self._reshape_output(Q_syn)

        if as_array:
            return Q_flat
        else:
            if synthetic_index is None:
                synthetic_index = self._get_synthetic_index(n_years)
            return pd.DataFrame(Q_flat, columns=self.site_names, index=synthetic_index)

    def generate(self, 
                 n_realizations=1, 
                 n_years=None, 
                 as_array=True):
        """
        Generate an ensemble of synthetic time series.
        
        Parameters
        ----------
        n_realizations : int
            Number of synthetic time series to generate.
        n_years : int, optional
            Number of years for each synthetic time series. If None, uses the number of historic years.
        as_array : bool
            If True, returns a numpy array; if False, returns a dictionary with realization indices as
            keys and pandas DataFrames as values.
        
        Returns
        -------
        np.ndarray or dict
        """
        if not self.fitted:
            raise RuntimeError("Call preprocessing() and fit() before generate().")
        if n_years is None:
            n_years = self.n_historic_years
        reals = [self.generate_single_series(n_years, as_array=as_array) for _ in range(n_realizations)]
        return np.stack(reals, axis=0) if as_array else {i: r for i, r in enumerate(reals)}


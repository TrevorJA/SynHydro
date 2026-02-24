"""
Ensemble management for synthetic timeseries data.

This module provides the Ensemble class for managing, formatting, and manipulating
ensembles of synthetic hydrologic timeseries data. The Ensemble class maintains
dual representations (by-site and by-realization) for flexible data access and
includes comprehensive I/O, statistical analysis, and visualization capabilities.
"""
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import h5py

logger = logging.getLogger(__name__)


@dataclass
class EnsembleMetadata:
    """
    Store metadata about an ensemble.

    Attributes
    ----------
    generator_class : str, optional
        Name of the generator class that created this ensemble.
    generator_params : Dict, optional
        Parameters used to configure the generator.
    creation_timestamp : str
        ISO format timestamp of when ensemble was created.
    n_realizations : int
        Number of realizations in the ensemble.
    n_sites : int
        Number of sites/locations in the ensemble.
    time_resolution : str, optional
        Time resolution of data ('daily', 'monthly', etc.).
    time_period : Tuple[str, str], optional
        Start and end dates of time series (ISO format strings).
    description : str, optional
        User-provided description of the ensemble.
    custom_attrs : Dict, optional
        Additional user-defined metadata attributes.
    """
    generator_class: Optional[str] = None
    generator_params: Optional[Dict[str, Any]] = None
    creation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    n_realizations: int = 0
    n_sites: int = 0
    time_resolution: Optional[str] = None
    time_period: Optional[Tuple[str, str]] = None
    description: Optional[str] = None
    custom_attrs: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'generator_class': self.generator_class,
            'generator_params': self.generator_params,
            'creation_timestamp': self.creation_timestamp,
            'n_realizations': self.n_realizations,
            'n_sites': self.n_sites,
            'time_resolution': self.time_resolution,
            'time_period': self.time_period,
            'description': self.description,
            'custom_attrs': self.custom_attrs,
        }


class Ensemble:
    """
    Manage ensemble timeseries data with dual representations.

    The Ensemble class stores synthetic timeseries data in two complementary formats:

    1. **By Realization**: `{realization_id: DataFrame[sites × time]}`
       - Keys are realization numbers (int)
       - Values are DataFrames with sites as columns

    2. **By Site**: `{site_name: DataFrame[realizations × time]}`
       - Keys are site names (str)
       - Values are DataFrames with realizations as columns

    Both representations are maintained automatically and provide efficient access
    for different analysis workflows.

    Parameters
    ----------
    data : Dict[Union[int, str], pd.DataFrame]
        Ensemble data in either format. Structure is automatically detected.
    metadata : EnsembleMetadata, optional
        Metadata about the ensemble. If None, creates default metadata.

    Attributes
    ----------
    data_by_realization : Dict[int, pd.DataFrame]
        Data organized by realization number.
    data_by_site : Dict[str, pd.DataFrame]
        Data organized by site name.
    realization_ids : List[int]
        List of all realization IDs.
    site_names : List[str]
        List of all site names.
    metadata : EnsembleMetadata
        Ensemble metadata and provenance information.

    Examples
    --------
    Create ensemble from generator output:

    >>> from synhydro import ThomasFieringGenerator, Ensemble
    >>> gen = ThomasFieringGenerator(Q_hist)
    >>> gen.fit()
    >>> Q_syn = gen.generate(n_years=50, n_realizations=100)
    >>> ensemble = Ensemble.from_generator(gen, n_years=50, n_realizations=100)

    Save and load ensemble:

    >>> ensemble.to_hdf5('synthetic_flows.h5')
    >>> ensemble_loaded = Ensemble.from_hdf5('synthetic_flows.h5')

    Access data by site or realization:

    >>> site_data = ensemble.data_by_site['site_A']  # All realizations for site A
    >>> real_data = ensemble.data_by_realization[0]  # All sites for realization 0

    Compute statistics:

    >>> stats = ensemble.summary(by='site')
    >>> percentiles = ensemble.percentile([10, 50, 90], by='site')
    """

    def __init__(self,
                 data: Dict[Union[int, str], pd.DataFrame],
                 metadata: Optional[EnsembleMetadata] = None):
        """
        Initialize Ensemble with data and optional metadata.

        Parameters
        ----------
        data : Dict[Union[int, str], pd.DataFrame]
            Ensemble data dictionary. Structure is auto-detected.
        metadata : EnsembleMetadata, optional
            Metadata about the ensemble.

        Raises
        ------
        TypeError
            If data is not a dictionary.
        ValueError
            If data structure cannot be determined.
        """
        if not isinstance(data, dict):
            logger.error("Data must be a dictionary")
            raise TypeError("Data must be a dictionary")

        if len(data) == 0:
            logger.error("Data dictionary is empty")
            raise ValueError("Data dictionary cannot be empty")

        # Infer and transform data structures
        data_structure = self._infer_data_structure(data)
        logger.debug(f"Detected data structure: {data_structure}")

        if data_structure == 'realizations':
            self.data_by_realization = data
            self.data_by_site = self._transform_realizations_to_sites(data)
        elif data_structure == 'sites':
            self.data_by_site = data
            self.data_by_realization = self._transform_sites_to_realizations(data)
        else:
            raise ValueError("Unknown data structure type. Expected 'realizations' or 'sites'.")

        self.realization_ids = list(self.data_by_realization.keys())
        self.site_names = list(self.data_by_site.keys())

        # Initialize or update metadata
        if metadata is None:
            self.metadata = EnsembleMetadata(
                n_realizations=len(self.realization_ids),
                n_sites=len(self.site_names)
            )
        else:
            self.metadata = metadata
            # Update counts from actual data
            self.metadata.n_realizations = len(self.realization_ids)
            self.metadata.n_sites = len(self.site_names)

        # Infer time period from data
        if self.metadata.time_period is None:
            self._infer_time_period()

        logger.info(f"Ensemble initialized: {len(self.realization_ids)} realizations, "
                   f"{len(self.site_names)} sites")

    @property
    def frequency(self) -> Optional[str]:
        """
        Get the time frequency/resolution of the ensemble data.

        Returns
        -------
        Optional[str]
            Time frequency (e.g., 'D', 'MS', 'YS') from metadata.
        """
        return self.metadata.time_resolution

    @property
    def sites(self) -> List[str]:
        """
        Get list of site names (alias for site_names).

        Returns
        -------
        List[str]
            List of site names in the ensemble.
        """
        return self.site_names

    def _infer_data_structure(self, data: Dict) -> str:
        """
        Infer whether data is organized by realizations or sites.

        Parameters
        ----------
        data : Dict
            Dictionary containing ensemble data.

        Returns
        -------
        str
            Either 'realizations' or 'sites'.

        Raises
        ------
        ValueError
            If structure cannot be determined.
        """
        first_key = next(iter(data))

        # Primary heuristic: check key type
        if isinstance(first_key, int):
            return 'realizations'
        elif isinstance(first_key, str):
            return 'sites'
        else:
            raise ValueError(f"Unknown key type: {type(first_key)}. "
                           "Expected int (realization) or str (site)")

    def _transform_realizations_to_sites(self,
                                         data_dict: Dict[int, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Transform from realization-indexed to site-indexed structure.

        Parameters
        ----------
        data_dict : Dict[int, pd.DataFrame]
            Dictionary keyed by realization numbers with DataFrames containing
            timeseries for multiple sites.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary keyed by site names with DataFrames containing
            timeseries for multiple realizations.
        """
        if not data_dict:
            return {}

        # Get all unique sites
        all_sites = set()
        for df in data_dict.values():
            all_sites.update(df.columns)

        result = {}

        for site in all_sites:
            # Collect all series for this site across realizations
            site_series = []
            for realization, df in data_dict.items():
                if site in df.columns:
                    series = df[site].rename(realization)
                    site_series.append(series)

            # Concatenate all series for this site
            if site_series:
                site_df = pd.concat(site_series, axis=1, sort=True)
                result[site] = site_df

        return result

    def _transform_sites_to_realizations(self,
                                         site_dict: Dict[str, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """
        Transform from site-indexed to realization-indexed structure.

        Parameters
        ----------
        site_dict : Dict[str, pd.DataFrame]
            Dictionary keyed by site names with DataFrames containing
            timeseries for multiple realizations.

        Returns
        -------
        Dict[int, pd.DataFrame]
            Dictionary keyed by realization numbers with DataFrames containing
            timeseries for multiple sites.
        """
        if not site_dict:
            return {}

        # Get all unique realizations across all sites
        all_realizations = set()
        for df in site_dict.values():
            all_realizations.update(df.columns)

        result = {}

        for realization in all_realizations:
            # Collect all series for this realization across sites
            realization_series = []
            for site, df in site_dict.items():
                if realization in df.columns:
                    series = df[realization].rename(site)
                    realization_series.append(series)

            # Concatenate all series for this realization
            if realization_series:
                realization_df = pd.concat(realization_series, axis=1, sort=True)
                result[realization] = realization_df

        return result

    def _infer_time_period(self):
        """Infer time period from data."""
        if self.data_by_realization:
            first_real = self.data_by_realization[self.realization_ids[0]]
            if hasattr(first_real.index, 'min') and hasattr(first_real.index, 'max'):
                start = str(first_real.index.min().date())
                end = str(first_real.index.max().date())
                self.metadata.time_period = (start, end)

    @classmethod
    def from_hdf5(cls,
                  filename: str,
                  realization_subset: Optional[List[int]] = None,
                  stored_by_node: bool = True) -> 'Ensemble':
        """
        Load ensemble from HDF5 file.

        Parameters
        ----------
        filename : str
            Path to HDF5 file.
        realization_subset : List[int], optional
            Load only specified realizations. If None, loads all.
        stored_by_node : bool, default=True
            If True, data is stored with sites as top-level groups.

        Returns
        -------
        Ensemble
            Loaded ensemble object.

        Examples
        --------
        >>> ensemble = Ensemble.from_hdf5('synthetic_flows.h5')
        >>> ensemble = Ensemble.from_hdf5('flows.h5', realization_subset=[0, 1, 2])
        """
        logger.info(f"Loading ensemble from {filename}")

        with h5py.File(filename, 'r') as f:
            # Load metadata if present
            metadata_dict = {}
            if 'metadata' in f.attrs:
                import json
                metadata_dict = json.loads(f.attrs['metadata'])

            if stored_by_node:
                # Get structure info from first node
                keys = list(f.keys())
                if not keys:
                    raise ValueError(f"HDF5 file {filename} is empty")

                first_node = f[keys[0]]
                column_labels = first_node.attrs['column_labels']
                dates_raw = first_node['date'][:].tolist()

                # Convert bytes to strings if necessary
                dates = [d.decode('utf-8') if isinstance(d, bytes) else d for d in dates_raw]

                # Determine which realizations to load
                if realization_subset is not None:
                    missing = [r for r in realization_subset if r not in column_labels]
                    if missing:
                        raise ValueError(f"Realizations {missing} not found in HDF5 file")
                    realization_ids = realization_subset
                else:
                    realization_ids = list(column_labels)

                logger.info(f"Loading {len(realization_ids)} realizations from {filename}")

                # Load data for all realizations
                ensemble_dict = {}
                for i, realization in enumerate(realization_ids):
                    data = {}
                    for node in keys:
                        node_data = f[node]
                        data[node] = node_data[str(realization)][:]

                    df = pd.DataFrame(data, index=pd.to_datetime(dates))
                    df.index.name = 'datetime'
                    ensemble_dict[i] = df
            else:
                # Data stored by realization
                keys = list(f.keys())
                if realization_subset is not None:
                    keys = [k for k in keys if k in realization_subset]

                ensemble_dict = {}
                for i, realization in enumerate(keys):
                    realization_group = f[str(realization)]
                    column_labels = realization_group.attrs['column_labels']
                    dates_raw = realization_group['date'][:].tolist()

                    # Convert bytes to strings if necessary
                    dates = [d.decode('utf-8') if isinstance(d, bytes) else d for d in dates_raw]

                    data = {}
                    for label in column_labels:
                        data[str(label)] = realization_group[str(label)][:]

                    df = pd.DataFrame(data, index=pd.to_datetime(dates))
                    df.index.name = 'datetime'
                    ensemble_dict[i] = df

        # Create metadata
        metadata = EnsembleMetadata(**metadata_dict) if metadata_dict else None

        return cls(ensemble_dict, metadata=metadata)

    def to_hdf5(self,
                filename: str,
                compression: Optional[str] = 'gzip',
                stored_by_node: bool = True):
        """
        Save ensemble to HDF5 file.

        Parameters
        ----------
        filename : str
            Path to output HDF5 file.
        compression : str, optional
            Compression algorithm ('gzip', 'lzf', None). Default is 'gzip'.
        stored_by_node : bool, default=True
            If True, store data with sites as top-level groups.

        Examples
        --------
        >>> ensemble.to_hdf5('synthetic_flows.h5')
        >>> ensemble.to_hdf5('flows.h5', compression='lzf')
        """
        logger.info(f"Saving ensemble to {filename}")

        with h5py.File(filename, 'w') as f:
            # Save metadata as attributes
            import json
            f.attrs['metadata'] = json.dumps(self.metadata.to_dict())

            if stored_by_node:
                # Store by site (nodes as top-level groups)
                for site, site_df in self.data_by_site.items():
                    grp = f.create_group(str(site))

                    # Store metadata
                    grp.attrs['column_labels'] = list(site_df.columns)

                    # Store dates
                    dates = site_df.index.astype(str).tolist()
                    grp.create_dataset('date', data=dates, compression=compression)

                    # Store each realization's data for this site
                    for col in site_df.columns:
                        grp.create_dataset(str(col),
                                         data=site_df[col].values,
                                         compression=compression)
            else:
                # Store by realization
                for real_id, real_df in self.data_by_realization.items():
                    grp = f.create_group(str(real_id))

                    # Store metadata
                    grp.attrs['column_labels'] = list(real_df.columns)

                    # Store dates
                    dates = real_df.index.astype(str).tolist()
                    grp.create_dataset('date', data=dates, compression=compression)

                    # Store each site's data for this realization
                    for col in real_df.columns:
                        grp.create_dataset(str(col),
                                         data=real_df[col].values,
                                         compression=compression)

        logger.info(f"Ensemble saved successfully to {filename}")

    @classmethod
    def from_generator(cls,
                       generator,
                       n_years: int,
                       n_realizations: int,
                       **gen_kwargs) -> 'Ensemble':
        """
        Create ensemble directly from a fitted Generator.

        Parameters
        ----------
        generator : Generator
            A fitted generator instance.
        n_years : int
            Number of years to generate.
        n_realizations : int
            Number of realizations to generate.
        **gen_kwargs
            Additional keyword arguments passed to generator.generate().

        Returns
        -------
        Ensemble
            New ensemble with metadata from generator.

        Examples
        --------
        >>> from synhydro import ThomasFieringGenerator, Ensemble
        >>> gen = ThomasFieringGenerator(Q_hist)
        >>> gen.fit()
        >>> ensemble = Ensemble.from_generator(gen, n_years=50, n_realizations=100)
        """
        logger.info(f"Creating ensemble from {generator.__class__.__name__}")

        # Generate synthetic data
        Q_syn = generator.generate(n_years, n_realizations, **gen_kwargs)

        # Check if generator already returned an Ensemble (e.g., from pipelines with disaggregation)
        if isinstance(Q_syn, cls):
            logger.debug(f"Generator returned Ensemble directly")
            return Q_syn

        # Convert generator output to ensemble format
        # Assuming generator returns DataFrame with columns as realizations
        ensemble_dict = {}

        # Try to get site name from generator
        if hasattr(generator, '_sites') and len(generator._sites) > 0:
            site_name = generator._sites[0]
        else:
            site_name = 'site_0'

        for i in range(n_realizations):
            if n_realizations == 1:
                # Single realization: keep Q_syn as is, just rename column
                df = Q_syn.copy()
                df.columns = [site_name]
                ensemble_dict[i] = df
            else:
                # Multiple realizations: extract column i and rename
                ensemble_dict[i] = Q_syn[[i]].copy()
                ensemble_dict[i].columns = [site_name]

        # Create metadata from generator
        metadata = EnsembleMetadata(
            generator_class=generator.__class__.__name__,
            generator_params=generator.get_params() if hasattr(generator, 'get_params') else None,
            n_realizations=n_realizations,
            description=f"Generated with {generator.__class__.__name__}"
        )

        return cls(ensemble_dict, metadata=metadata)

    def summary(self, by: str = 'site') -> pd.DataFrame:
        """
        Compute statistical summary across realizations or sites.

        Parameters
        ----------
        by : {'site', 'realization'}, default='site'
            Compute statistics by site or by realization.

        Returns
        -------
        pd.DataFrame
            Summary statistics (mean, std, min, max) for each site or realization.

        Examples
        --------
        >>> stats = ensemble.summary(by='site')
        >>> print(stats)
        """
        if by == 'site':
            results = []
            for site_name, site_df in self.data_by_site.items():
                stats = {
                    'site': site_name,
                    'mean': site_df.mean(axis=1).mean(),
                    'std': site_df.std(axis=1).mean(),
                    'min': site_df.min(axis=1).min(),
                    'max': site_df.max(axis=1).max(),
                }
                results.append(stats)
            return pd.DataFrame(results).set_index('site')
        elif by == 'realization':
            results = []
            for real_id, real_df in self.data_by_realization.items():
                stats = {
                    'realization': real_id,
                    'mean': real_df.mean().mean(),
                    'std': real_df.std().mean(),
                    'min': real_df.min().min(),
                    'max': real_df.max().max(),
                }
                results.append(stats)
            return pd.DataFrame(results).set_index('realization')
        else:
            raise ValueError("by must be 'site' or 'realization'")

    def percentile(self,
                   q: Union[float, List[float]],
                   by: str = 'site') -> Dict[str, pd.DataFrame]:
        """
        Compute percentiles across realizations.

        Parameters
        ----------
        q : float or List[float]
            Percentile(s) to compute (0-100).
        by : {'site', 'realization'}, default='site'
            Compute percentiles by site or realization.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping site/realization to DataFrame of percentiles over time.

        Examples
        --------
        >>> p = ensemble.percentile([10, 50, 90], by='site')
        >>> site_a_percentiles = p['site_A']
        """
        if not isinstance(q, list):
            q = [q]

        results = {}

        if by == 'site':
            for site_name, site_df in self.data_by_site.items():
                percentiles = {}
                for percentile in q:
                    percentiles[f'p{percentile}'] = site_df.quantile(percentile/100, axis=1)
                results[site_name] = pd.DataFrame(percentiles)
        elif by == 'realization':
            for real_id, real_df in self.data_by_realization.items():
                percentiles = {}
                for percentile in q:
                    percentiles[f'p{percentile}'] = real_df.quantile(percentile/100, axis=1)
                results[real_id] = pd.DataFrame(percentiles)
        else:
            raise ValueError("by must be 'site' or 'realization'")

        return results

    def subset(self,
               sites: Optional[List[str]] = None,
               realizations: Optional[List[int]] = None,
               start_date: Optional[str] = None,
               end_date: Optional[str] = None) -> 'Ensemble':
        """
        Create subset of ensemble by sites, realizations, or time period.

        Parameters
        ----------
        sites : List[str], optional
            Site names to include.
        realizations : List[int], optional
            Realization IDs to include.
        start_date : str, optional
            Start date (ISO format or pandas-parseable).
        end_date : str, optional
            End date (ISO format or pandas-parseable).

        Returns
        -------
        Ensemble
            New ensemble containing only the subset.

        Examples
        --------
        >>> subset = ensemble.subset(sites=['site_A', 'site_B'],
        ...                          start_date='2000-01-01',
        ...                          end_date='2010-12-31')
        """
        # Start with current data
        data = self.data_by_realization.copy()

        # Filter by realizations
        if realizations is not None:
            data = {k: v for k, v in data.items() if k in realizations}

        # Filter by time period
        if start_date is not None or end_date is not None:
            for real_id in data:
                df = data[real_id]
                if start_date is not None:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date is not None:
                    df = df[df.index <= pd.to_datetime(end_date)]
                data[real_id] = df

        # Filter by sites
        if sites is not None:
            for real_id in data:
                df = data[real_id]
                available_sites = [s for s in sites if s in df.columns]
                data[real_id] = df[available_sites]

        # Create new ensemble with updated metadata
        new_metadata = EnsembleMetadata(
            generator_class=self.metadata.generator_class,
            generator_params=self.metadata.generator_params,
            description=f"Subset of {self.metadata.description or 'ensemble'}"
        )

        return Ensemble(data, metadata=new_metadata)

    def resample(self, freq: str) -> 'Ensemble':
        """
        Resample time series to different frequency.

        Parameters
        ----------
        freq : str
            Pandas frequency string ('D', 'W', 'MS', 'AS', etc.).

        Returns
        -------
        Ensemble
            New ensemble with resampled data.

        Examples
        --------
        >>> monthly_ensemble = daily_ensemble.resample('MS')
        """
        resampled_data = {}
        for real_id, df in self.data_by_realization.items():
            resampled_data[real_id] = df.resample(freq).sum()

        new_metadata = EnsembleMetadata(
            generator_class=self.metadata.generator_class,
            generator_params=self.metadata.generator_params,
            time_resolution=freq,
            description=f"Resampled to {freq}"
        )

        return Ensemble(resampled_data, metadata=new_metadata)

    def __repr__(self) -> str:
        """String representation of Ensemble."""
        return (f"Ensemble(n_realizations={len(self.realization_ids)}, "
                f"n_sites={len(self.site_names)}, "
                f"generator={self.metadata.generator_class or 'unknown'})")

    def __str__(self) -> str:
        """Detailed string representation."""
        lines = [
            "=" * 60,
            "Ensemble Summary",
            "=" * 60,
            f"Realizations: {len(self.realization_ids)}",
            f"Sites: {len(self.site_names)}",
        ]

        if self.metadata.time_period:
            lines.append(f"Time Period: {self.metadata.time_period[0]} to {self.metadata.time_period[1]}")

        if self.metadata.generator_class:
            lines.append(f"Generator: {self.metadata.generator_class}")

        if self.metadata.description:
            lines.append(f"Description: {self.metadata.description}")

        lines.append("=" * 60)

        return "\n".join(lines)

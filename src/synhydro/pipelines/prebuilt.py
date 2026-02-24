"""
Pre-built pipeline configurations.

This module provides convenience wrapper classes for common generator-disaggregator
combinations. These classes internally use the GeneratorDisaggregatorPipeline but
provide a simplified interface for frequently used combinations.
"""
import pandas as pd
from typing import Optional

from synhydro.core.pipeline import GeneratorDisaggregatorPipeline
from synhydro.methods.generation.nonparametric.kirsch import KirschGenerator
from synhydro.methods.generation.parametric.thomas_fiering import ThomasFieringGenerator
from synhydro.methods.disaggregation.temporal.nowak import NowakDisaggregator


class KirschNowakPipeline(GeneratorDisaggregatorPipeline):
    """
    Pre-configured pipeline combining Kirsch generator with Nowak disaggregator.

    This pipeline generates monthly synthetic flows using the Kirsch nonparametric
    bootstrap method, then disaggregates them to daily flows using the Nowak
    KNN-based temporal disaggregation method.

    Parameters
    ----------
    Q_obs : pd.DataFrame
        Daily observed streamflow data with DatetimeIndex.
        Used to train both the generator (aggregated to monthly) and
        disaggregator (used as daily).
    generate_using_log_flow : bool, default=True
        Whether to generate in log-space (Kirsch parameter).
    matrix_repair_method : str, default='spectral'
        Method for repairing correlation matrices (Kirsch parameter).
    n_neighbors : int, default=5
        Number of KNN neighbors for disaggregation (Nowak parameter).
    max_month_shift : int, default=7
        Maximum day shift for monthly profiles (Nowak parameter).
    name : str, optional
        Name for this pipeline instance.
    debug : bool, default=False
        Enable debug logging.

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.pipelines import KirschNowakPipeline
    >>>
    >>> # Load daily historic flows
    >>> Q_daily = pd.read_csv('daily_flows.csv', index_col=0, parse_dates=True)
    >>>
    >>> # Create pipeline
    >>> pipeline = KirschNowakPipeline(Q_daily)
    >>>
    >>> # Fit and generate
    >>> pipeline.preprocessing()
    >>> pipeline.fit()
    >>> daily_ensemble = pipeline.generate(n_realizations=100, n_years=50)

    Notes
    -----
    This pipeline is equivalent to creating:
    ```python
    generator = KirschGenerator(Q_obs, generate_using_log_flow=True)
    disaggregator = NowakDisaggregator(Q_obs, n_neighbors=5)
    pipeline = GeneratorDisaggregatorPipeline(generator, disaggregator)
    ```

    References
    ----------
    Kirsch generator: Nonparametric bootstrap with correlation preservation
    Nowak disaggregator: KNN-based temporal disaggregation (Nowak et al., 2010)
    """

    def __init__(self,
                 Q_obs: pd.DataFrame,
                 generate_using_log_flow: bool = True,
                 matrix_repair_method: str = 'spectral',
                 n_neighbors: int = 5,
                 max_month_shift: int = 7,
                 name: Optional[str] = None,
                 debug: bool = False):
        """
        Initialize the Kirsch-Nowak pipeline.

        Parameters
        ----------
        Q_obs : pd.DataFrame
            Daily observed streamflow data.
        generate_using_log_flow : bool, default=True
            Generate in log-space for Kirsch.
        matrix_repair_method : str, default='spectral'
            Correlation matrix repair method for Kirsch.
        n_neighbors : int, default=5
            Number of KNN neighbors for Nowak.
        max_month_shift : int, default=7
            Day shift for Nowak monthly profiles.
        name : str, optional
            Pipeline name.
        debug : bool, default=False
            Enable debug logging.
        """
        # Create Kirsch generator
        generator = KirschGenerator(
            Q_obs=Q_obs,
            generate_using_log_flow=generate_using_log_flow,
            matrix_repair_method=matrix_repair_method,
            debug=debug
        )

        # Create Nowak disaggregator
        disaggregator = NowakDisaggregator(
            Q_obs=Q_obs,
            n_neighbors=n_neighbors,
            max_month_shift=max_month_shift,
            debug=debug
        )

        # Initialize parent pipeline
        super().__init__(
            generator=generator,
            disaggregator=disaggregator,
            name=name or "KirschNowakPipeline",
            debug=debug
        )


class ThomasFieringNowakPipeline(GeneratorDisaggregatorPipeline):
    """
    Pre-configured pipeline combining Thomas-Fiering generator with Nowak disaggregator.

    This pipeline generates monthly synthetic flows using the Thomas-Fiering AR(1)
    parametric method with Stedinger-Taylor normalization, then disaggregates them
    to daily flows using the Nowak KNN-based temporal disaggregation method.

    Note: Thomas-Fiering is a univariate method, so only single-site generation
    is supported. For multisite, use KirschNowakPipeline instead.

    Parameters
    ----------
    Q_obs : pd.Series or pd.DataFrame
        Daily observed streamflow data with DatetimeIndex.
        If DataFrame, only the first column is used (Thomas-Fiering is univariate).
        Used to train both the generator (aggregated to monthly) and
        disaggregator (used as daily).
    n_neighbors : int, default=5
        Number of KNN neighbors for disaggregation (Nowak parameter).
    max_month_shift : int, default=7
        Maximum day shift for monthly profiles (Nowak parameter).
    name : str, optional
        Name for this pipeline instance.
    debug : bool, default=False
        Enable debug logging.

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.pipelines import ThomasFieringNowakPipeline
    >>>
    >>> # Load daily historic flows (single site)
    >>> Q_daily = pd.read_csv('daily_flows.csv', index_col=0, parse_dates=True)
    >>>
    >>> # Create pipeline
    >>> pipeline = ThomasFieringNowakPipeline(Q_daily['site_1'])
    >>>
    >>> # Fit and generate
    >>> pipeline.preprocessing()
    >>> pipeline.fit()
    >>> daily_ensemble = pipeline.generate(n_realizations=100, n_years=50)

    Notes
    -----
    This pipeline is equivalent to creating:
    ```python
    generator = ThomasFieringGenerator(Q_obs)
    disaggregator = NowakDisaggregator(Q_obs, n_neighbors=5)
    pipeline = GeneratorDisaggregatorPipeline(generator, disaggregator)
    ```

    References
    ----------
    Thomas-Fiering: AR(1) with Stedinger-Taylor normalization
    Nowak disaggregator: KNN-based temporal disaggregation (Nowak et al., 2010)
    """

    def __init__(self,
                 Q_obs,
                 n_neighbors: int = 5,
                 max_month_shift: int = 7,
                 name: Optional[str] = None,
                 debug: bool = False):
        """
        Initialize the Thomas-Fiering-Nowak pipeline.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Daily observed streamflow data. If DataFrame with multiple columns,
            only the first column is used (Thomas-Fiering is univariate).
        n_neighbors : int, default=5
            Number of KNN neighbors for Nowak.
        max_month_shift : int, default=7
            Day shift for Nowak monthly profiles.
        name : str, optional
            Pipeline name.
        debug : bool, default=False
            Enable debug logging.
        """
        # Create Thomas-Fiering generator
        generator = ThomasFieringGenerator(
            Q_obs=Q_obs,
            debug=debug
        )

        # Create Nowak disaggregator
        disaggregator = NowakDisaggregator(
            Q_obs=Q_obs,
            n_neighbors=n_neighbors,
            max_month_shift=max_month_shift,
            debug=debug
        )

        # Initialize parent pipeline
        super().__init__(
            generator=generator,
            disaggregator=disaggregator,
            name=name or "ThomasFieringNowakPipeline",
            debug=debug
        )

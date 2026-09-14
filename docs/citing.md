# How to cite SynHydro

If you use SynHydro in your research, please cite the version you used.
Two DOIs identify the software: a concept DOI that always resolves to the
latest version, and a version DOI for the specific release. Both are
filled in after the first release is archived on Zenodo.

```
Amestoy, T. (2026). SynHydro: Synthetic hydrologic timeseries generation
in Python (Version 0.1.0) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.YYYYYYY
```

```bibtex
@software{amestoy_synhydro_2026,
  author    = {Amestoy, Trevor},
  title     = {{SynHydro}: Synthetic hydrologic timeseries generation in Python},
  version   = {0.1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.YYYYYYY},
  url       = {https://github.com/TrevorJA/SynHydro}
}
```

The concept DOI (`10.5281/zenodo.XXXXXXX`) is the one to put in a README
or a data availability statement that should keep pointing at the newest
release. Cite the version DOI in a paper so readers can reproduce the
exact code that generated your ensembles. Ensembles saved with
`Ensemble.to_hdf5()` record the SynHydro version that produced them in
the `synhydro_version` attribute.

The repository also ships a `CITATION.cff` file, which GitHub renders as
"Cite this repository" and Zenodo reads when archiving a release.

## Citing the methods

Each generator and disaggregator implements a published method. When you
use one, cite its source paper as well. The primary reference is listed at
the top of every [algorithm page](algorithms/index.md) and in the
[reference list](references/references.md).

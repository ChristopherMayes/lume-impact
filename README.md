# lume-impact
Tools for using Impact-T and Impact-Z in LUME.

Basic usage:
```python
from impact import Impact

# Prepare Impact object. This will call I.configure() automatically. 
I = Impact('/path/to/ImpactT.in', verbose=True)

# Change some things
I.header['Np'] = 10000
I.header['Nx'] = 16
I.header['Ny'] = 16
I.header['Nz'] = 16

# Run
I.run()
...

# Archive all output

I.archive('test.h5')



```


Current release info
====================

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-lume--impact-green.svg)](https://anaconda.org/conda-forge/lume-impact) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/lume-impact.svg)](https://anaconda.org/conda-forge/lume-impact) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/lume-impact.svg)](https://anaconda.org/conda-forge/lume-impact) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/lume-impact.svg)](https://anaconda.org/conda-forge/lume-impact) |

Installing lume-impact
======================

Installing `lume-impact` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Once the `conda-forge` channel has been enabled, `lume-impact` can be installed with:

```
conda install lume-impact
```

It is possible to list all of the versions of `lume-impact` available on your platform with:

```
conda search lume-impact --channel conda-forge
```



# Impact-T Source

https://github.com/impact-lbl/IMPACT-T



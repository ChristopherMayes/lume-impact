# LUME-Impact
Tools for using Impact-T and Impact-Z in LUME.


**`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/impact-documentation-blue.svg)](https://christophermayes.github.io/lume-impact/)  |



Basic usage:
```python
from impact import Impact

# Prepare Impact object. This will call I.configure() automatically.
I = Impact('/path/to/ImpactT.in', verbose=True)

# Change some things
I.header['Np'] = 10000
I.header['Nx'] = 32
I.header['Ny'] = 32
I.header['Nz'] = 32

# Run
I.run()
...


# Plot the results
I.plot()
```


![Summary LUME-Impact plot](docs/assets/plot.png)


```python
# Archive all output
I.archive('test.h5')

# Plot particle phase space projection
I.particles['final_particles'].plot('z', 'pz')

```

![openPMD-beamphysics z-pz phase space plot](docs/assets/zpz.png)

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



# Impact-T Executables

Impact-T is available through conda-forge and can be installed via:
```bash
conda create -n impact
source activate impact # or conda activate impact
# For non-MPI
conda install -c conda-forge impact-t

# For OpenMPI
conda install -c conda-forge impact-t=*=mpi_openmpi*

# For MPICH
conda install -c conda-forge impact-t=*=mpi_mpich*
```
After these steps, the IMPACT-T executable `ImpactTexe` or `ImpactTexe-mpi`, respectively, will be in your [PATH](https://en.wikipedia.org/wiki/PATH_(variable)) environment variable and is thus ready to use like any regular command-line command.



Visit [https://github.com/impact-lbl/IMPACT-T](https://github.com/impact-lbl/IMPACT-T) for these and further instructions, including those to build from source.

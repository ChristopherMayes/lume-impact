# lume-impact
Tools for using Impact-T and Impact-Z in LUME.

Basic usage:
```python
from impact import Impact

# Prepare Impact object. This will call I.configure() automatically. 
I = Impact('../templates/lcls_injector/ImpactT.in', verbose=True)

# Change some things
I.input['header']['Np'] = 10000
I.input['header']['Nx'] = 16
I.input['header']['Ny'] = 16
I.input['header']['Nz'] = 16

# Run
I.run()
...

# Archive all output
h5 = h5py.File('test.h5', 'w')
I.archive(h5)



```


# Impact-T Source

https://github.com/impact-lbl/IMPACT-T


# Notes: 
Do not run scripts within the package directory.


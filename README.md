# lume-impact
Tools for using Impact-T and Impact-Z in LUME.

```python
from impact import Impact
I = Impact('../templates/lcls_injector/ImpactT.in', verbose=True)

# Run
I.run()
...

# Archive all output
h5 = h5py.File('test.h5', 'w')
I.archive(h5)

```


# Notes: 
Do not run scripts within the package directory.


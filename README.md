# lume-impact
Tools for using Impact-T and Impact-Z in LUME.

Basic usage:
```python
from impact import Impact

# Prepare Impact object. This will call I.configure() automatically. 
I = Impact('../templates/lcls_injector/ImpactT.in', verbose=True)

# Run
I.run()
...

# Archive all output
h5 = h5py.File('test.h5', 'w')
I.archive(h5)

# Clean up all files
I.cleanup()

```


# Notes: 
Do not run scripts within the package directory.


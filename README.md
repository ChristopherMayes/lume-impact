# lume-impact
Tools for using Impact-T and Impact-Z in LUME


# Notes: 
If running the package in a script.
Save script files outside of the lume-impact directory.
Otherwise, the relative imports might not be resolved.

Example error: 
Traceback (most recent call last):                                                                      
  File "test.py", line 4, in <module>                                                                   
    from impact.parsers import *                                                                        
  File "/Users/nneveu/Code/Github/lume-impact/impact/impact.py", line 3, in <module>                    
    from .parsers import parse_impact_input, load_many_fort, FORT_STAT_TYPES, FORT_PARTICLE_TYPES, FORT_SLICE_TYPES, header_str, header_bookkeeper
ImportError: attempted relative import with no known parent package  

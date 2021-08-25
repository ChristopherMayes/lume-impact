from .impact import Impact
from .impact_distgen import run_impact_with_distgen, evaluate_impact_with_distgen
from .control import ControlGroup


import os
# Used to access data directory
root, _ = os.path.split(__file__)
template_dir = os.path.join(root, '../templates/')

from . import _version
__version__ = _version.get_versions()['version']

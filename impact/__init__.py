from .impact import Impact

from ._version import __version__

import os
# Used to access data directory
root, _ = os.path.split(__file__)
template_dir = os.path.join(root, '../templates/')
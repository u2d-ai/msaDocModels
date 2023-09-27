import glob
from os.path import basename, dirname, isfile, join

version = "0.0.105"
__author__ = "Stefan Welcker"
__copyright__ = "Copyright 2022, U2D.ai"
__license__ = "MIT"
__version__ = version
__maintainer__ = "Stefan Welcker"
__email__ = "stefan@u2d.ai"
__status__ = "Beta"
__url__ = "https://github.com/swelcker/msaDocModels"

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]

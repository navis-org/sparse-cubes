import os
import sys

# Make sibling helper modules (e.g. `_shapes`) importable by bare name whatever
# import mode pytest is running in.
sys.path.insert(0, os.path.dirname(__file__))

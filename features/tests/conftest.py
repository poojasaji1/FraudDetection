import sys
import os

# Make features/ and project root importable from any pytest invocation path
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_features_dir = os.path.dirname(_tests_dir)
_project_dir = os.path.dirname(_features_dir)

for p in (_features_dir, _project_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

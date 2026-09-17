import sys


def test_python_version():
    assert sys.version_info[:2] == (3, 12)


def test_core_packages_import():
    import numpy
    import pandas
    import sklearn
    import scipy
    import matplotlib
    import seaborn
    import xgboost
    import betacal
    import scikit_posthocs
    import ucimlrepo
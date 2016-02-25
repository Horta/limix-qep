from __future__ import division, print_function
try:
    import limix_build
except ImportError:
    from ez_build import use_limix_build
    use_limix_build()
    import limix_build

import os
import sys
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins

builtins.__LIMIX_QEP_SETUP__ = True

PKG_NAME            = "limix_qep"
MAJOR               = 0
MINOR               = 0
MICRO               = 1
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

from limix_build import write_version_py
from limix_build import get_version_info

def get_test_suite():
    from unittest import TestLoader
    return TestLoader().discover(PKG_NAME)

def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    write_version_py(PKG_NAME, VERSION, ISRELEASED)

    install_requires = ['hcache', 'limix-math']
    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')
    
    setup_requires = []

    metadata = dict(
        name=PKG_NAME,
        maintainer="Limix Developers",
        version=get_version_info(PKG_NAME, VERSION, ISRELEASED)[0],
        maintainer_email="horta@ebi.ac.uk",
        test_suite='setup.get_test_suite',
        license="BSD",
        url='http://pmbio.github.io/limix/',
        packages=[PKG_NAME],
        install_requires=install_requires,
        setup_requires=setup_requires
    )

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)

if __name__ == '__main__':
    setup_package()

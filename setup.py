from __future__ import division, print_function
import os
import sys
import glob
from setuptools import setup
from setuptools import find_packages

PKG_NAME = 'limix_qep'
VERSION = '0.1.17.dev1'

try:
    from distutils.command.bdist_conda import CondaDistribution
except ImportError:
    conda_present = False
else:
    conda_present = True

try:
    import numpy
except ImportError:
    print("Error: numpy package couldn't be found." +
          " Please, install it so I can proceed.")
    sys.exit(1)
else:
    print("Good: numpy %s" % numpy.__version__)

try:
    import scipy
except ImportError:
    print("Error: scipy package couldn't be found." +
          " Please, install it so I can proceed.")
    sys.exit(1)
else:
    print("Good: scipy %s" % scipy.__version__)

def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    install_requires = ['hcache', 'limix_math>=0.1.12', 'limix_tool>=0.1.16',
                        'limix_util', 'lim>=0.0.5', 'pytest']
    setup_requires = ['pytest-runner']
    tests_require = ['pytest']

    metadata = dict(
        name=PKG_NAME,
        maintainer="Limix Developers",
        version=VERSION,
        maintainer_email="horta@ebi.ac.uk",
        packages=find_packages(),
        license="BSD",
        url='http://pmbio.github.io/limix/',
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
        zip_safe=False,
        include_package_data=True,
        cffi_modules=['moments_build.py:binomial']
    )

    if conda_present:
        metadata['distclass'] = CondaDistribution
        metadata['conda_buildnum'] = 1
        metadata['conda_features'] = ['mkl']

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)

if __name__ == '__main__':
    import logging.config
    logging.config.fileConfig('logging.ini')
    setup_package()

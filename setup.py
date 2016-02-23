from __future__ import division, print_function
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')
import imp
import sys

if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins

builtins.__LIMIX_QEP_SETUP__ = True

PKG_NAME            = "limix_qep"
MAJOR               = 0
MINOR               = 1
MICRO               = 1
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

try:
    imp.find_module('limix_build')
except ImportError:
    import subprocess
    r = subprocess.call("pip install limix_build", shell=True)
    if r != 0:
        print('Fatal: could not install limix_build using pip. We need this' +
              ' package before we can build.')
        sys.exit(1)

    try:
        imp.find_module('limix_build')
    except ImportError:
        print('Fatal: could not import limix_build. Please, make sure it is ' +
              'installed before attempting to build this package.')
        sys.exit(1)

from limix_build import write_version_py
from limix_build import parse_setuppy_commands
from limix_build import generate_cython
from limix_build import get_version_info
from limix_build import get_version_filename

def get_test_suite():
    from unittest import TestLoader
    return TestLoader().discover(PKG_NAME)

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage(PKG_NAME)

    config.get_version(get_version_filename(PKG_NAME)) # sets config.version

    return config

def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    write_version_py(PKG_NAME, VERSION, ISRELEASED)

    metadata = dict(
        name=PKG_NAME,
        maintainer="Limix Developers",
        maintainer_email = "horta@ebi.ac.uk",
        test_suite='setup.get_test_suite',
        packages=[PKG_NAME],
        install_requires=['limix_util', 'scipy'],
        setup_requires=[]
    )

    if "--force" in sys.argv:
        run_build = True
    else:
        run_build = parse_setuppy_commands(PKG_NAME)

    from setuptools import setup
    if run_build:
        from numpy.distutils.core import setup
        cwd = os.path.abspath(os.path.dirname(__file__))
        if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
            generate_cython(PKG_NAME)

        metadata['configuration'] = configuration
    else:
        metadata['version'] = get_version_info(PKG_NAME, VERSION, ISRELEASED)[0]

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)

if __name__ == '__main__':
    setup_package()

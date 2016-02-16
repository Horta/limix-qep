import os
# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')
from numpy.distutils.core import setup
# from Cython.Build import cythonize
# from Cython.Distutils import Extension
# from Cython.Distutils import build_ext
# from Cython.Distutils import Extension
import sys
import imp

MAJOR               = 0
MINOR               = 1
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# def create(mod_name, sources, **kwargs):
#     return Extension(mod_name, sources,
#             extra_compile_args=['-Wno-unused-function'], **kwargs)

def get_test_suite():
    from unittest import TestLoader
    return TestLoader().discover('limix_qep')

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('limix_qep')

    config.get_version('limix_qep/version.py') # sets config.version

    return config

def setup_package():
    path = os.path.realpath(__file__)
    dirname = os.path.dirname(path)
    mod = imp.load_source('__init__',
                          os.path.join(dirname, 'build_util', '__init__.py'))
    write_version_py = mod.write_version_py
    generate_cython = mod.generate_cython

    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    # Rewrite the version file everytime
    write_version_py(VERSION, ISRELEASED, filename='limix_qep/version.py')

    build_requires = ['numpy', 'scipy', 'Cython', 'numba']

    metadata = dict(
        name='limix_qep',
        test_suite='setup.get_test_suite',
        setup_requires=build_requires,
        install_requires=build_requires,
        packages=['limix_qep']
    )

    from setuptools import setup
    # from numpy.distutils.core import setup
    cwd = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
        # Generate Cython sources, unless building from source release
        generate_cython()

    metadata['configuration'] = configuration

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)

if __name__ == '__main__':
    setup_package()

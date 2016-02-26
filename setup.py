from __future__ import division, print_function
# try:
# import limix_build
# except ImportError:
#     from ez_build import use_limix_build
#     use_limix_build()
#     import limix_build

import os
import sys
import glob
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import distutils.command.bdist_conda
from distutils.command.bdist_conda import CondaDistribution
import numpy as np

if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins

builtins.__LIMIX_QEP_SETUP__ = True

PKG_NAME            = "limix_qep"
MAJOR               = 0
MINOR               = 1
MICRO               = 4
ISRELEASED          = True
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

from limix_build import write_version_py
from limix_build import get_version_info

def cephes_info():
    curdir = os.path.abspath(os.path.dirname(__file__))

    define_macros = []
    if sys.platform == 'win32':
        define_macros.append(('_USE_MATH_DEFINES', None))
    define_macros.append(('PI', 3.141592653589793238462643383279502884))


    cephes_src = glob.glob(os.path.join(curdir, 'cephes', '*/*.c'))
    cephes_src.extend(glob.glob(os.path.join(curdir, 'cephes', '*.c')))

    cephes_hdr = glob.glob(os.path.join(curdir, 'cephes', '*/*.h'))
    cephes_hdr.extend(glob.glob(os.path.join(curdir, 'cephes', '*.h')))

    return dict(src=cephes_src, include_dirs=[curdir],
                define_macros=define_macros,
                extra_compile_args=['-Wno-unused-function'],
                depends=cephes_src+cephes_hdr)

def special_extension():
    ci = cephes_info()

    curdir = os.path.abspath(os.path.dirname(__file__))

    special_folder = os.path.join(curdir, 'limix_qep/special/')

    src = ['nbinom_moms.pyx', 'nbinom_moms_base.c']
    src = [os.path.join(special_folder, s) for s in src]

    hdr = ['nbinom_moms.pxd', 'nbinom_moms_base.h']
    hdr = [os.path.join(special_folder, h) for h in hdr]

    depends = src + hdr + ci['depends']

    ext = Extension('limix_qep/special/nbinom_moms',
                    src+ci['src'],
                    include_dirs=ci['include_dirs']+[np.get_include()],
                    extra_compile_args=ci['extra_compile_args'],
                    define_macros=ci['define_macros'],
                    depends=depends)

    return ext

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

    setup_requires = ['limix_build']

    metadata = dict(
        name=PKG_NAME,
        maintainer="Limix Developers",
        version=get_version_info(PKG_NAME, VERSION, ISRELEASED)[0],
        maintainer_email="horta@ebi.ac.uk",
        test_suite='setup.get_test_suite',
        packages=find_packages(),
        license="BSD",
        url='http://pmbio.github.io/limix/',
        install_requires=install_requires,
        setup_requires=setup_requires,
        zip_safe=False,
        ext_modules=cythonize([special_extension()]),
        cmdclass=dict(build_ext=build_ext),
        distclass=CondaDistribution,
        conda_buildnum=1,
        conda_features=['mkl']
    )

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)

if __name__ == '__main__':
    setup_package()

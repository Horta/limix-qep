from __future__ import division, print_function
import os
import sys
import glob
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

PKG_NAME = 'limix_qep'
VERSION  = '0.1.9'

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

try:
    import scipy
except ImportError:
    print("Error: scipy package couldn't be found."+
          " Please, install it so I can proceed.")
    sys.exit(1)

try:
    import cython
except ImportError:
    print("Error: cython package couldn't be found."+
          " Please, install it so I can proceed.")
    sys.exit(1)


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
                    include_dirs=ci['include_dirs'],
                    extra_compile_args=ci['extra_compile_args'],
                    define_macros=ci['define_macros'],
                    depends=depends)

    return ext

def get_test_suite():
    from unittest import TestLoader
    return TestLoader().discover(PKG_NAME)

def write_version():
    cnt = """
# THIS FILE IS GENERATED FROM %(package_name)s SETUP.PY
version = '%(version)s'
"""
    filename = os.path.join(PKG_NAME, 'version.py')
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'package_name': PKG_NAME.upper()})
    finally:
        a.close()

def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    write_version()

    install_requires = ['hcache', 'limix_math=>0.1.7', 'limix_tool',
                        'limix_util']
    setup_requires = ['pytest-runner']
    tests_require = ['pytest']

    metadata = dict(
        name=PKG_NAME,
        maintainer="Limix Developers",
        version=VERSION,
        maintainer_email="horta@ebi.ac.uk",
        test_suite='setup.get_test_suite',
        packages=find_packages(),
        license="BSD",
        url='http://pmbio.github.io/limix/',
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
        zip_safe=False,
        ext_modules=cythonize([special_extension()]),
        cmdclass=dict(build_ext=build_ext),
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
    setup_package()

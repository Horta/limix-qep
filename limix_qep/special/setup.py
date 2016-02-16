import os
import sys
import glob
from os.path import join
from numpy.distutils.misc_util import Configuration

curdir = os.path.abspath(os.path.dirname(__file__))

define_macros = []
if sys.platform == 'win32':
    define_macros.append(('_USE_MATH_DEFINES', None))

cephes_src = join(curdir, 'cephes', '*.c')
cephes_hdr = join(curdir, 'cephes', '*.h')

cephes_src = glob.glob(cephes_src)
cephes_hdr = glob.glob(cephes_hdr)

def configuration(parent_package='', top_path=None):

    config = Configuration('special', parent_package, top_path)

    config.add_library('cephes', sources=cephes_src,
                   include_dirs=[curdir],
                   depends=cephes_hdr,
                   extra_compile_args=['-Wno-unused-function'])

    base_src = ['nbinom_moms_base.c']
    base_hdr = ['nbinom_moms_base.h']

    config.add_library('nbinom_moms_base',
                       sources=base_src + cephes_src,
                       include_dirs=[curdir],
                       depends=base_hdr + cephes_hdr)

    config.add_extension('nbinom_moms', ['nbinom_moms.pyx'],
                         include_dirs=[curdir],
                         libraries=['cephes', 'nbinom_moms_base'],
                         depends=base_src + cephes_src + base_hdr + cephes_hdr)

    config.add_subpackage('test')
    config.add_define_macros(define_macros)
    config.make_config_py() # installs __config__.py
    return config

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')

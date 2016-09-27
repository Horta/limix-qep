from os.path import join
from glob import glob
import logging


def _make():
    from cffi import FFI
    import ncephes
    import limix_math

    logger = logging.getLogger()

    logger.debug('CFFI make')

    ffi = FFI()

    sources = glob(join('lib', 'liknorm', 'src', '*.c'))
    hdrs = glob(join('lib', 'liknorm', 'include', '*.h'))
    hdrs += glob(join('lib', 'liknorm', 'lib', '*.h'))
    incls = [join('lib', 'liknorm', 'include'), join('lib', 'liknorm', 'lib')]
    libraries = ['m']

    logger.debug('Sources: %s', bytes(sources))
    logger.debug('Headers: %s', bytes(hdrs))
    logger.debug('Incls: %s', bytes(incls))
    logger.debug('Libraries: %s', bytes(libraries))

    ffi.set_source('limix_qep.moments.liknorm._liknorm_ffi',
                   '''#include "liknorm_python.h"''',
                   include_dirs=incls,
                   sources=sources,
                   libraries=libraries,
                   library_dirs=[],
                   depends=sources + hdrs,
                   extra_compile_args=["-std=c11"])

    with open(join('lib', 'liknorm', 'include', 'liknorm_python.h'), 'r') as f:
        ffi.cdef(f.read())

    return ffi

liknorm = _make()

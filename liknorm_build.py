from os.path import join
from glob import glob
import logging


def _make():
    from cffi import FFI

    logger = logging.getLogger()

    logger.debug('CFFI make')

    ffi = FFI()

    sources = glob(join('lib', 'liknorm', '*.c')) + [join('lib', 'liknorm.c')]
    hdrs = glob(join('lib', 'liknorm', '*.h')) + [join('lib', 'liknorm.h')]
    incls = ['lib']
    libraries = ['m']

    logger.debug('Sources: %s', bytes(sources))
    logger.debug('Headers: %s', bytes(hdrs))
    logger.debug('Incls: %s', bytes(incls))
    logger.debug('Libraries: %s', bytes(libraries))

    ffi.set_source('limix_qep.liknorm._liknorm_ffi',
                   '''#include "liknorm.h"''',
                   include_dirs=incls,
                   sources=sources,
                   libraries=libraries,
                   library_dirs=[],
                   depends=sources + hdrs,
                   extra_compile_args=["-std=c11"])

    with open(join('lib', 'liknorm.h'), 'r') as f:
        ffi.cdef(f.read())

    return ffi

liknorm = _make()

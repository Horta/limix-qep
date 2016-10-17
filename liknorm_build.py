import logging
from glob import glob
from os.path import join


def _make():
    from cffi import FFI

    logger = logging.getLogger()

    logger.debug('CFFI make')

    ffi = FFI()

    sources = glob(join('liknorm', 'liknorm', '*.c')) + \
        [join('liknorm', 'liknorm.c')]
    hdrs = glob(join('liknorm', 'liknorm', '*.h')) + \
        [join('liknorm', 'liknorm.h')]
    incls = ['liknorm']
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

    with open(join('liknorm', 'liknorm.h'), 'r') as f:
        ffi.cdef(f.read())

    return ffi

liknorm = _make()

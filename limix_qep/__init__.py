from __future__ import absolute_import
from pkg_resources import get_distribution

from . import lik
from . import tool
from . import liknorm
from . import ep

__version__ = get_distribution('limix-qep').version


def test():
    import os
    p = __import__('limix_qep').__path__[0]
    src_path = os.path.abspath(p)
    old_path = os.getcwd()
    os.chdir(src_path)

    try:
        return_code = __import__('pytest').main([])
    finally:
        os.chdir(old_path)

    return return_code

__all__ = ['test', 'scan_binomial']

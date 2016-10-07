from __future__ import absolute_import
from pkg_resources import get_distribution

from .lik import Bernoulli
from .lik import Binomial
# from .tool.heritability import estimate as h2_estimate
from .tool.scan import scan
from .tool.scan import scan_binomial
from . import liknorm

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

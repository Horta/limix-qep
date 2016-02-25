try:
    __LIMIX_QEP_SETUP__
except NameError:
    __LIMIX_QEP_SETUP__ = False

if not __LIMIX_QEP_SETUP__:
    from .version import git_revision as __git_revision__
    from .version import version as __version__

from lik import Bernoulli
from lik import Binomial
from tool.heritability import estimate as h2_estimate
from tool.scan import scan as scan

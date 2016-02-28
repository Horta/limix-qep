from __future__ import absolute_import
from .version import version as __version__
from .lik import Bernoulli
from .lik import Binomial
from .tool.heritability import estimate as h2_estimate
from .tool.scan import scan as scan

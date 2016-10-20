Poisson heritability
--------------------

A variable :math:`y \in \{0, 1, 2, \dots\}` is Poisson distributed if

.. math::

    y \sim \frac{\lambda^{y} e^{-\lambda}}{y!}

where :math:`\lambda_i` is the *rate of occurrence* of a specific event over
a constant period of time.

Suppose we have a dataset of :math:`n` individuals, each one showing an
outcome :math:`y_i`. It also happens that we have genotype information about
those individuals and we want to assess whether those outcomes might be
explained by their genotype.

Since we don't know the underlying genetic framework, we might well start
with the most simple one: additive genetic effect.
Let us define


.. math::

    z_i = \mu_i + \sum_j \mathrm B_{i,j} b_j + \epsilon_i

where :math:`b_j` are the effect-sizes of genetic contribution.
We assume

.. math::

    \mathbf b \sim \mathcal N(0, \sigma_b^2\mathrm I)

.. math::

    \boldsymbol\epsilon \sim \mathcal N(0, \sigma_{\epsilon}^2\mathrm I)

We have

.. math::

    \mathbf z \sim \mathcal N(1 \mu, \mathrm K = \sigma_b^2 \mathrm B \mathrm B^{\intercal}
            + \sigma_{\epsilon}^2\mathrm I)


p.d.f for each individual

.. math::

    p(y_i | \lambda_i) = \frac{\lambda_i^{y_i} e^{-\lambda_i}}{y_i!}

for the i-th individual.
The marginal likelihood is

.. math::

    p(\mathbf y) = \int \prod_i p(y_i | g(\lambda_i)=z_i)
        \mathcal N(\mathbf z ~|~ \mu, \mathrm K) \mathrm d\mathbf z

where

.. math::

    g(x) = \log x

is the canonical link function.

.. literalinclude:: /../examples/poisson_heritability.py

The output should be similar to::

    Heritability: 0.686429589528
..
.. .. program-output:: python ../examples/poisson_heritability.py

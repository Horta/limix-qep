# Prior

\[
\begin{align}
    \mathrm K &= v \mathrm Q \mathrm S \mathrm Q^{T} +
        v \delta \mathrm I\\
    \mathbf m &= \mathrm M \boldsymbol\beta
\end{align}
\]

# Posterior
\[
\begin{align}
    \Sigma &= (\mathrm K^{-1} + \tilde{\mathrm T})^{-1}\\
    \boldsymbol \mu &= \Sigma (\mathrm K^{-1} \mathbf m +
         \tilde{\boldsymbol\eta})
\end{align}
\]

# Aux

\[
\begin{align}
    \mathrm A_0 &= v \delta \mathrm I\\
    \mathrm A_1 &= (v \delta \mathrm I + \tilde{\mathrm T}^{-1})^{-1}
                = \tilde{\mathrm T} (v \delta \tilde{\mathrm T} +
                        \mathrm I)^{-1}\\
    \mathrm A_2 &= \tilde{\mathrm T}^{-1} \mathrm A_1
                = (v \delta \tilde{\mathrm T} + \mathrm I)^{-1}\\
    \mathrm B &= \mathrm Q^{\mathrm T} \mathrm A_1 \mathrm Q +
    (v \mathrm S)^{-1}
\end{align}
\]

\[
    (\mathrm K^{-1} + \tilde{\mathrm T})^{-1} =
        \tilde{\mathrm T}^{-1} (\mathrm K + \tilde{\mathrm T}^{-1})^{-1}
        \mathrm K
\]

\[
  (\mathrm K + \tilde{\mathrm T}^{-1})^{-1} = \mathrm A_1 -
      \mathrm A_1 \mathrm Q\mathrm B^{-1} \mathrm Q^{\mathrm T} \mathrm A_1
\]

# Tips

Pre-compute $\mathrm Q \mathrm B^{-1} \mathrm Q^{\mathrm T}$ and $\mathrm K$
when there is no rank deficiency in $\mathrm K$.

# Posterior implementation

\[
\begin{align}
  \Sigma &=
    \tilde{\mathrm T}^{-1} (\tilde{\mathrm T}^{-1} + \mathrm K)^{-1} \mathrm K
    = \tilde{\mathrm T}^{-1} (\mathrm A_1 -
      \mathrm A_1 \mathrm Q \mathrm B^{-1}\mathrm Q^{\mathrm T} \mathrm A_1)
        \mathrm K\\
    &= \mathrm A_2 \mathrm K - \mathrm A_2 \mathrm Q
            \mathrm B^{-1}\mathrm Q^{\mathrm T} \mathrm A_1 \mathrm K\\
  \boldsymbol \mu &= \tilde{\mathrm T}^{-1} (\tilde{\mathrm
                     T}^{-1} + \mathrm K)^{-1} \mathbf m
                     + \tilde{\mathrm T}^{-1} (\tilde{\mathrm T}^{-1} +
                            \mathrm K)^{-1} \mathrm K \tilde{\boldsymbol
                                \eta}\\
         &= \tilde{\mathrm T}^{-1} (\mathrm A_1 -
           \mathrm A_1 \mathrm Q \mathrm B^{-1}\mathrm Q^{\mathrm T}
           \mathrm A_1)
            \mathbf m
        + \tilde{\mathrm T}^{-1} (\mathrm A_1 -
          \mathrm A_1 \mathrm Q \mathrm B^{-1}\mathrm Q^{\mathrm T}
          \mathrm A_1)
          \mathrm K \tilde{\boldsymbol \eta}\\
          &= \mathrm A_2 \mathbf m -
            \mathrm A_2 \mathrm Q \mathrm B^{-1}\mathrm Q^{\mathrm T}
            \mathrm A_1 \mathbf m

         + \mathrm A_2 \mathrm K \tilde{\boldsymbol \eta} -
           \mathrm A_2 \mathrm Q \mathrm B^{-1}\mathrm Q^{\mathrm T}
           \mathrm A_1 \mathrm K \tilde{\boldsymbol \eta}
\end{align}
\]

# Prediction

\[
\begin{align}
    \mu_* &= m_* + \mathrm k_{*}^{\mathrm T}
            (\mathrm A_1 - \mathrm A_1 \mathrm Q
                \mathrm B^{-1} \mathrm Q^{\mathrm T} \mathrm A_1)
                (\tilde{\mathrm T}\boldsymbol{\mathrm m} +
                    \tilde{\boldsymbol\eta}) \\
    \sigma^2_* &= \mathrm k_{*,*} - \mathrm k_{*}^{\mathrm T}
            (\mathrm A_1 - \mathrm A_1 \mathrm Q
                \mathrm B^{-1} \mathrm Q^{\mathrm T} \mathrm A_1)
            \tilde{\mathrm T} \mathrm k_{*}
\end{align}
\]

## Log marginal likelihood

\[
  \text{LML} = - \frac{1}{2} \log |\mathrm K + \tilde{\mathrm T}^{-1}| -
                \frac{1}{2} (\mathbf m - \tilde{\boldsymbol\mu})^{T}
                (\mathrm K + \tilde{\mathrm T}^{-1})^{-1}
                  (\mathbf m - \tilde{\boldsymbol\mu})
              + \sum_i \log \Phi\left(
                    \frac{y_i \mu_{-i}}{\sqrt{1 + \sigma_{-i}^2}} \right)\\
      + \frac{1}{2} \sum_i \log(\tilde{\sigma}_i^2 + \sigma_{-i}^2)
      + \sum_i \frac{(\tilde{\mu_i} - \mu_{-i})^2}{2
          (\tilde{\sigma}_i^2 + \sigma_{-i}^2)}\\
\]

__Part 1__:
\[
  -\frac{1}{2}\log \big|\mathrm K + \tilde{\mathrm T}^{-1}\big| =
    -\frac{1}{2}\log\big|\mathrm B\big|
        -\frac{1}{2} \log\big|v \mathrm S\big|
        +\frac{1}{2}\log\big|\mathrm A_1\big|
\]

__Part 2__:
\[
\sum_i \frac{(\tilde{\mu}_i - \mu_{-i})^2}{2
    (\tilde{\sigma}_i^2 + \sigma_{-i}^2)} =
      \frac{\tilde{\boldsymbol \mu}^{T} (\tilde{\mathrm T}^{-1} +
          \Sigma_-)^{-1} \tilde{\boldsymbol\mu}}{2} -
          \tilde{\boldsymbol\mu}^{T} (\tilde{\mathrm T}^{-1} +
              \Sigma_-)^{-1} \boldsymbol\mu_- +
              \frac{\boldsymbol \mu_{-}^{\mathrm T} (\tilde{\mathrm T}^{-1} +
                  \Sigma_-)^{-1}\boldsymbol  \mu_-}{2}
\]

__Part 3__:
\[
\begin{align}
  - \frac{\tilde{\boldsymbol\mu}^{\mathrm T} (\mathrm K +
        \tilde{\mathrm T}^{-1})^{-1}
    \tilde{\boldsymbol\mu}}{2} + \frac{\tilde{\boldsymbol\mu}^{T}
        (\tilde{\mathrm T}^{-1} +
        \Sigma_-)^{-1} \tilde{\boldsymbol  \mu}}{2}
        &= \frac{1}{2} \tilde{\boldsymbol \eta}^{\mathrm T} \Big(
          -\tilde{\mathrm T}^{-1}\mathrm A_1\tilde{\mathrm T}^{-1} +
            \tilde{\mathrm T}^{-1} \mathrm A_1 \mathrm Q\mathrm B^{-1}
            \mathrm Q^{\mathrm T} \mathrm A_1\tilde{\mathrm T}^{-1} +
            \tilde{\mathrm T}^{-1} - (\tilde{\mathrm T} +
                  \Sigma_-^{-1})^{-1}
          \Big) \tilde{\boldsymbol\eta}\\
        &= \frac{1}{2} \tilde{\boldsymbol\eta}
            (
                \mathrm A_0 (\tilde{\mathrm T} \mathrm A_0 + \mathrm I)^{-1}
                + \mathrm A_2
                    \mathrm Q\mathrm B^{-1}\mathrm Q^{\mathrm T}
                  \mathrm A_2
                - (\tilde{\mathrm T} +
                      \Sigma_-^{-1})^{-1}
            )
            \tilde{\boldsymbol\eta}
\end{align}
\]

__Part 4__:
\[
\frac{1}{2}\boldsymbol\mu_-^{\mathrm T} (\tilde{\mathrm T}^{-1} +
    \Sigma_-)^{-1} (\boldsymbol\mu_- - 2 \tilde{\boldsymbol{\mu}}) =
    \frac{1}{2} \boldsymbol\eta_-^{\mathrm T} (\tilde{\mathrm T} +
        \Sigma_-^{-1})^{-1}
        (\tilde{\mathrm T} \boldsymbol\mu_- - 2 \tilde{\boldsymbol\eta})
\]

__Part 5__:

\[
    \mathbf m^{\mathrm T}(\mathrm K + \tilde{\mathrm T}^{-1})^{-1}
        \tilde{\boldsymbol \mu}
    = \mathbf m^{\mathrm T} \mathrm A_2 \tilde{\boldsymbol\eta} -
        \mathbf m^{\mathrm T} \mathrm A_1
                \mathrm Q \mathrm B^{-1} \mathrm Q^{\mathrm T}
              \mathrm A_2 \tilde{\boldsymbol\eta}
\]

__Part 6__:
\[
  - \frac{1}{2} \mathbf m^{\mathrm T} (\mathrm K + \tilde{\mathrm T}^{-1})^{-1}
  \mathbf m =
    - \frac{1}{2} \mathbf m^{\mathrm T} \mathrm A_1 \mathbf m
    + \frac{1}{2} \mathbf m^{\mathrm T} \mathrm A_1 \mathrm Q
    \mathrm B^{-1} \mathrm Q^{\mathrm T} \mathrm A_1 \mathbf m
\]

__Part 7__:
\[
  \frac{1}{2} \sum_i \log(\tilde{\sigma}_i^2 + \sigma_{-i}^2) =
    \frac{1}{2} \big(-\log|\mathrm{\tilde T}| + \log|\mathrm{\tilde T} +
      \Sigma_-^{-1}| - \log|\Sigma_-^{-1}|\big)
\]

> Note that from Part 1 and Part 7 have $\frac{1}{2} \log |\mathrm A_1|$ and
> $-\frac{1}{2} \log |\tilde{\mathrm T}|$, respectively.
> Their sum is given by $\frac{1}{2} \log |\mathrm A_2|$ and is defined
> as Part 9.

# Optimal beta

Let $\mathbf m = \mathrm M \boldsymbol\beta$ be the mean and $K$ be the
covariance. The optimal $\boldsymbol\beta^*$ is given by
taking the gradient of LML and setting it to zero.

\[
  \boldsymbol\beta^* = (\mathrm M^{\mathrm T} (\tilde{\mathrm T}^{-1} +
      \mathrm K)^{-1} \mathrm M)^{-1}
    \mathrm M^{\mathrm T} (\tilde{\mathrm T}^{-1} + \mathrm K)^{-1} \tilde{\boldsymbol\mu}.
\]

## Implementation

\[
\begin{align}
  \boldsymbol \beta^* &= (\mathrm M^{\mathrm T} \mathrm A_1 \mathrm M -
    \mathrm M^{\mathrm T} \mathrm A_1 \mathrm Q \mathrm B^{-1} \mathrm
    Q^{\mathrm T} \mathrm A_1 \mathrm M)^{-1}
      (\mathrm M^{\mathrm T} \mathrm A_1 - \mathrm M^{\mathrm T}\mathrm A_1
        \mathrm Q \mathrm B^{-1} \mathrm Q^{\mathrm T} \mathrm A_1)
        \tilde{\boldsymbol \mu}\\
        &= (\mathrm M^{\mathrm T} \mathrm A_1 \mathrm M - \mathrm M^{\mathrm T}
        \mathrm A_1 \mathrm Q \mathrm B_1^{-1} \mathrm Q^{\mathrm T}\mathrm A_1
        \mathrm M)^{-1} \mathrm M^{\mathrm T}
        (\mathrm A_2 - \mathrm A_1 \mathrm Q \mathrm B_1^{-1}
            \mathrm Q^{\mathrm T}\mathrm A_2) \tilde {\boldsymbol\eta}
\end{align}
\]


## Derivative over variance (it needs a review)

\[
  \frac{\partial \text{LML}}{\partial \theta} =
    \frac{1}{2} (\mathbf m - \tilde{\boldsymbol{\mu}})^{\mathrm T}
    (\mathrm K + \tilde \Sigma)^{-1} \frac{\partial \mathrm K}{\partial \theta}
    (\mathrm K + \tilde \Sigma)^{-1} (\mathbf m - \tilde{\boldsymbol{\mu}})\\
    - (\mathbf m - \tilde{\boldsymbol{\mu}})^{\mathrm T} (\mathrm K +
        \tilde{\mathrm T}^{-1})^{-1}
    \frac{\partial \mathbf m}{\partial \theta} - \frac{1}{2} \text{tr}\Big(
      (\mathrm K + \tilde{\mathrm T}^{-1})^{-1}
      \frac{\partial \mathrm K}{\partial \theta}\Big)
\]

Part 1

\[
  (\mathrm K+\tilde{\mathrm T}^{-1})^{-1} \tilde{\boldsymbol \mu} =
    \mathrm A_0(\mathrm A_0 + \tilde{\mathrm T})^{-1}
    \tilde{\boldsymbol\eta} - \mathrm A_1 \mathrm Q \mathrm B^{-1}
    \mathrm Q^{\mathrm T} \mathrm A_0 (\mathrm A_0 + \tilde{\mathrm T})^{-1}
    \tilde{\boldsymbol \eta}
\]

Part 2

\[
  (\mathrm K+\tilde{\mathrm T}^{-1})^{-1} \mathbf m =
    \mathrm A_1 \mathbf m
    - \mathrm A_1 \mathrm Q \mathrm B^{-1}
    \mathrm Q^{\mathrm T} \mathrm A_1 \mathbf m
\]

Part 3

\[
  \text{tr}[(\mathrm K + \tilde{\mathrm T}^{-1})^{-1}] = \text{tr}[\mathrm A_1
    \frac{\partial \mathrm K}{\partial \theta}] -
    \text{tr}[\mathrm A_1 \mathrm Q \mathrm B^{-1} \mathrm Q^{\mathrm T}
    \mathrm A_1
    \frac{\partial \mathrm K}{\partial \theta}]
\]

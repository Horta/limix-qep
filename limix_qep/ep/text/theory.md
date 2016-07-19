# Formulae

Covariance matrix:
\[
  \mathrm K = \sigma_g^2 \mathrm Q \mathrm S \mathrm Q^{T} +
          \sigma_g^2 \delta \mathrm I
\]

\[
\mathrm A_0 = \sigma_g^{-2} \delta^{-1} \mathrm I \quad \text{if }\delta > 0
\]

\[
  \mathrm A_1 = (\sigma_g^2 \delta \mathrm I + \tilde\Sigma)^{-1}
\]

\[
  \mathrm A_1 = \mathrm A_0 (\mathrm A_0 +
            \tilde{\mathrm T})^{-1} \tilde{\mathrm T}
            = \tilde{\mathrm T} - \tilde{\mathrm T}
            (\mathrm A_0 +
              \tilde{\mathrm T})^{-1}
              \tilde{\mathrm T} \quad \text{if }\delta > 0

\]

\[
  \mathrm A_1 = \tilde{\mathrm T} \quad \text{if }
    \delta = 0
\]

\[
  \mathrm A_2 = (\mathrm A_0 + \tilde{\mathrm T})^{-1}
    \quad \text{if } \delta > 0
\]


\[
  \mathrm B_0 = \mathrm Q^T
      \mathrm A_0 \mathrm Q + (\sigma_g^2
      \mathrm S)^{-1} \quad \text{if } \delta > 0\\
  \mathrm B_1 = \mathrm Q^T
      \mathrm A_1 \mathrm Q + (\sigma_g^2
      \mathrm S)^{-1}
\]

\[
  \mathrm K^{-1} = \mathrm A_0 - \mathrm A_0 \mathrm Q \mathrm B_0^{-1}
        \mathrm Q^T \mathrm A_0 \quad \text{if } \delta > 0
\]

\[
  (\mathrm K + \tilde{\Sigma})^{-1} = \mathrm A_1 -
      \mathrm A_1 \mathrm Q\mathrm B_1^{-1} \mathrm Q^T \mathrm A_1
\]

Mean $\mathbf m = \mathrm X \boldsymbol\beta$.

## Log marginal likelihood

\[
  \text{LML} = - \frac{1}{2} \log |\mathrm K + \tilde{\Sigma}| -
                \frac{1}{2} (\mathbf m - \tilde{\boldsymbol\mu})^{T}
                (\mathrm K + \tilde{\Sigma})^{-1}
                  (\mathbf m - \tilde{\boldsymbol\mu})
              + \sum_i \log \Phi\left(
                    \frac{y_i \mu_{-i}}{\sqrt{1 + \sigma_{-i}^2}} \right)\\
      + \frac{1}{2} \sum_i \log(\tilde{\sigma}_i^2 + \sigma_{-i}^2)
      + \sum_i \frac{(\tilde{\mu_i} - \mu_{-i})^2}{2
          (\tilde{\sigma}_i^2 + \sigma_{-i}^2)}\\
\]

__Part 1__:
\[
  -\frac{1}{2}\log \big|\mathrm K + \tilde{\Sigma}\big| =
    -\frac{1}{2}\log\big|\mathrm B_1\big|
        -\frac{1}{2} \log\big|\sigma_g^2 \mathrm S\big|
        +\frac{1}{2}\log\big|\mathrm A_1\big|
\]

__Part 2__:
\[
\sum_i \frac{(\tilde{\mu}_i - \mu_{-i})^2}{2
    (\tilde{\sigma}_i^2 + \sigma_{-i}^2)} =
      \frac{\tilde{\boldsymbol \mu}^{T} (\tilde{\Sigma} +
          \Sigma_-)^{-1} \tilde{\boldsymbol\mu}}{2} -
          \tilde{\boldsymbol\mu}^{T} (\tilde{\Sigma} +
              \Sigma_-)^{-1} \boldsymbol\mu_- +
              \frac{\boldsymbol \mu_{-}^T (\tilde{\Sigma} +
                  \Sigma_-)^{-1}\boldsymbol  \mu_-}{2}
\]

__Part 3__:
\[
  - \frac{\tilde{\boldsymbol\mu}^T (\mathrm K + \tilde{\Sigma})^{-1}
    \tilde{\boldsymbol\mu}}{2} + \frac{\tilde{\boldsymbol\mu}^{T}
        (\tilde{\Sigma} +
        \Sigma_-)^{-1} \tilde{\boldsymbol  \mu}}{2}\\
        = \frac{1}{2} \tilde{\boldsymbol \eta}^T \Big(
          -\tilde{\Sigma}\mathrm A_1\tilde{\Sigma} +
            \tilde{\Sigma} \mathrm A_1 \mathrm Q\mathrm B_1^{-1}
            \mathrm Q^T \mathrm A_1\tilde{\Sigma} +
            \tilde{\Sigma} - (\tilde{\mathrm T} +
                  \Sigma_-^{-1})^{-1}
          \Big) \tilde{\boldsymbol\eta}
\]
$~~~~$If $\delta > 0$:
\[
          = \frac{1}{2} \tilde{\boldsymbol \eta}^T \Big(
              -\tilde{\Sigma} + (\mathrm A_0 +
                    \tilde{\mathrm T})^{-1} + (\mathrm A_0 +
                        \tilde{\mathrm T})^{-1}
                        \mathrm A_0 \mathrm Q \mathrm B_1^{-1} \mathrm Q^T
                        \mathrm A_0(\mathrm A_0 +
                            \tilde{\mathrm T})^{-1}+\tilde{\Sigma}-
                            (\tilde{\mathrm T} +
                                  \Sigma_-^{-1})^{-1}
            \Big) \tilde{\boldsymbol\eta}\\
            = \frac{1}{2} \tilde{\boldsymbol \eta}^T \Big(
                (\mathrm A_0 +
                      \tilde{\mathrm T})^{-1} + (\mathrm A_0 +
                          \tilde{\mathrm T})^{-1}
                          \mathrm A_0 \mathrm Q \mathrm B_1^{-1} \mathrm Q^T
                          \mathrm A_0(\mathrm A_0 +
                              \tilde{\mathrm T})^{-1}-
                              (\tilde{\mathrm T} +
                                    \Sigma_-^{-1})^{-1}
              \Big) \tilde{\boldsymbol\eta}
\]
$~~~~$If $\delta = 0$:
\[
  = \frac{1}{2} \tilde{\boldsymbol \eta}^T \Big(
    \mathrm Q \mathrm B_1^{-1} \mathrm Q^T -
    (\tilde{\mathrm T} + \Sigma_{-}^{-1})^{-1}
    \Big) \tilde{\boldsymbol\eta}
\]

__Part 4__:
\[
\frac{1}{2}\boldsymbol\mu_-^T (\tilde{\Sigma} +
    \Sigma_-)^{-1} (\boldsymbol\mu_- - 2 \tilde{\boldsymbol{\mu}}) =
    \frac{1}{2} \boldsymbol\eta_-^T (\tilde{\mathrm T} + \Sigma_-^{-1})^{-1}
        (\tilde{\mathrm T} \boldsymbol\mu_- - 2 \tilde{\boldsymbol\eta})
\]

__Part 5__:

$~~~~$If $\delta > 0$:
\[
  \mathbf m^T(\mathrm K + \tilde{\Sigma})^{-1} \tilde{\boldsymbol \mu} =
    \mathbf m^T \big(\mathrm A_0(\mathrm A_0 + \tilde{\mathrm T})^{-1}
      \tilde{\boldsymbol\eta} - \mathrm A_1 \mathrm Q \mathrm B_1^{-1}
      \mathrm Q^T \mathrm A_0 (\mathrm A_0 + \tilde{\mathrm T})^{-1}
      \tilde{\boldsymbol \eta}\big)
\]

$~~~~$If $\delta = 0$:
\[
    =\mathbf m^T \tilde{\boldsymbol \eta} - \mathbf m^T
    \tilde{\mathrm T} \mathrm Q
    \mathrm B_1^{-1} \mathrm Q^T \tilde{\boldsymbol \eta}
\]

__Part 6__:
\[
  - \frac{1}{2} \mathbf m^T (\mathrm K + \tilde{\Sigma})^{-1} \mathbf m =
    - \frac{1}{2} \mathbf m^T \mathrm A_1 \mathbf m
    + \frac{1}{2} \mathbf m^T \mathrm A_1 \mathrm Q
    \mathrm B_1^{-1} \mathrm Q^T \mathrm A_1 \mathbf m
\]

__Part 7__:
\[
  \frac{1}{2} \sum_i \log(\tilde{\sigma}_i^2 + \sigma_{-i}^2) =
    \frac{1}{2} \big(-\log\mathrm{\tilde T} + \log(\mathrm{\tilde T} +
      \Sigma_-^{-1}) - \log(\Sigma_-^{-1})\big)
\]

When $\delta = 0$, the terms $\frac{1}{2} \log \mathrm A_1$ from Part 1
and $-\frac{1}{2} \log \tilde{\mathrm T}$ from Part 7 can be subtracted out.

## Derivative over variance

\[
  \frac{\partial \text{LML}}{\partial \theta} =
    \frac{1}{2} (\mathbf m - \tilde{\boldsymbol{\mu}})^T
    (\mathrm K + \tilde \Sigma)^{-1} \frac{\partial \mathrm K}{\partial \theta}
    (\mathrm K + \tilde \Sigma)^{-1} (\mathbf m - \tilde{\boldsymbol{\mu}})\\
    - (\mathbf m - \tilde{\boldsymbol{\mu}})^T (\mathrm K + \tilde\Sigma)^{-1}
    \frac{\partial \mathbf m}{\partial \theta} - \frac{1}{2} \text{tr}\Big(
      (\mathrm K + \tilde\Sigma)^{-1}
      \frac{\partial \mathrm K}{\partial \theta}\Big)
\]

Part 1

\[
  (\mathrm K+\tilde\Sigma)^{-1} \tilde{\boldsymbol \mu} =
    \mathrm A_0(\mathrm A_0 + \tilde{\mathrm T})^{-1}
    \tilde{\boldsymbol\eta} - \mathrm A_1 \mathrm Q \mathrm B_1^{-1}
    \mathrm Q^T \mathrm A_0 (\mathrm A_0 + \tilde{\mathrm T})^{-1}
    \tilde{\boldsymbol \eta}
\]

Part 2

\[
  (\mathrm K+\tilde\Sigma)^{-1} \mathbf m =
    \mathrm A_1 \mathbf m
    - \mathrm A_1 \mathrm Q \mathrm B_1^{-1}
    \mathrm Q^T \mathrm A_1 \mathbf m
\]

Part 3

\[
  \text{tr}[(\mathrm K + \tilde{\Sigma})^{-1}] = \text{tr}[\mathrm A_1
    \frac{\partial \mathrm K}{\partial \theta}] -
    \text{tr}[\mathrm A_1 \mathrm Q \mathrm B_1^{-1} \mathrm Q^T \mathrm A_1
    \frac{\partial \mathrm K}{\partial \theta}]
\]

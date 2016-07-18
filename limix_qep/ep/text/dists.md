# Approximated posterior (i.e., normalized Joint from EP)

$\boldsymbol \eta$ and $\tilde{\mathrm{T}}$ are the natural parameters
of the site-likelihood approximations.

Posterior:
\[
  p(\mathbf f \,|\, \mathbf y)_{\text{EP}} =
        \mathcal N(\mathbf f \,|\, \boldsymbol \mu, \Sigma)
\]

\[
  \boldsymbol \mu = \Sigma (\mathrm K^{-1} \mathbf m +
      \tilde{\boldsymbol \eta})\\
  \Sigma = (\mathrm K^{-1} + \tilde{\mathrm{T}})^{-1}
\]

__Implementation__

\[
  \Sigma = (\mathrm I + \mathrm K \tilde{\mathrm T})^{-1}  \mathrm K =
    \tilde{\mathrm T}^{-1} (\tilde{\mathrm T}^{-1} + \mathrm K)^{-1} \mathrm K\\
    = \tilde{\mathrm T}^{-1} (\mathrm A_1 -
      \mathrm A_1 \mathrm Q \mathrm B_1^{-1}\mathrm Q^T \mathrm A_1)\mathrm K
\]

\[
  \boldsymbol \mu = (\mathrm I + \mathrm K \tilde{\mathrm T})^{-1} \mathbf m
                     + (\mathrm I + \mathrm K \tilde{\mathrm T})^{-1}
                     \mathrm K \tilde{\boldsymbol \eta}\\
         = \tilde{\mathrm T}^{-1} (\mathrm A_1 -
           \mathrm A_1 \mathrm Q \mathrm B_1^{-1}\mathrm Q^T \mathrm A_1)
            \mathbf m
        + \tilde{\mathrm T}^{-1} (\mathrm A_1 -
          \mathrm A_1 \mathrm Q \mathrm B_1^{-1}\mathrm Q^T \mathrm A_1)
          \mathrm K \tilde{\boldsymbol \eta}
\]


<!-- If $\delta > 0$:
\[
  \Sigma = (\mathrm K^{-1} + \tilde{\mathrm{T}})^{-1}
            = \mathrm A_2 - \mathrm A_2 \mathrm A_0 \mathrm Q
            (\mathrm Q^T \mathrm A_0 \mathrm A_2 \mathrm A_0 \mathrm Q
              - \mathrm B_0)^{-1} \mathrm Q^T \mathrm A_0 \mathrm A_2
\] -->

If $\delta = 0$:

\[
  \Sigma = (\mathrm I -
    \mathrm Q \mathrm B_1^{-1}\mathrm Q^T \mathrm A_1)
    \sigma_g^2 \mathrm Q \mathrm S \mathrm Q^T
\]

\[
  \boldsymbol \mu = \mathbf m -
    \mathrm Q \mathrm B_1^{-1}\mathrm Q^T \mathrm A_1 \mathbf m +
    \mathrm K \tilde{\boldsymbol \eta} -
    \mathrm Q \mathrm B_1^{-1}\mathrm Q^T \mathrm A_1
    \mathrm K \tilde{\boldsymbol \eta}
\]

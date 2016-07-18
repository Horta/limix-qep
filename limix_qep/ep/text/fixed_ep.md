# Optimal beta

Let $m = M \beta$ be the mean and $K$ be the covariance.
The optimal $\beta^*$ is given by
taking the gradient of LML and setting to zero.

\[
  \beta^* = (M^\intercal (\tilde \Sigma + K)^{-1} M)^{-1}
    (\tilde \Sigma + K)^{-1} \tilde \mu.
\]

## Implementation

\[
  \boldsymbol \beta^* = (\mathrm M^T \mathrm A_1 \mathrm M -
    \mathrm M^T \mathrm A_1 \mathrm Q \mathrm B_1^{-1} \mathrm Q^T
      \mathrm A_1 \mathrm M)^{-1}
      (\mathrm M^T \mathrm A_1 - \mathrm M^T\mathrm A_1
        \mathrm Q \mathrm B_1^{-1} \mathrm Q^T \mathrm A_1)
        \tilde{\boldsymbol \mu}
\]

For $\delta > 0$
\[
  \boldsymbol \beta^* = (\mathrm M^T \mathrm A_1 \mathrm M - \mathrm M^T
        \mathrm A_1 \mathrm Q \mathrm B_1^{-1} \mathrm Q^T\mathrm A_1
        \mathrm M)^{-1} (\tilde {\boldsymbol\eta} - \tilde{\mathrm T}
          (\mathrm A_0 + \tilde{\mathrm T})^{-1} \tilde{\boldsymbol\eta} -
          \mathrm A_1 \mathrm Q \mathrm B_1^{-1} \mathrm Q^T
          (\tilde{\boldsymbol\eta} - \tilde{\mathrm T} (\mathrm A_0 +
            \tilde{\mathrm T})^{-1} \tilde{\boldsymbol\eta}))
\]

<!--
Site likelihood parameter
\[
  \tilde S = \text{diag}(\tilde \tau)
\]

\[
  LL^\intercal = Q^\intercal \tilde S Q + S^{-1}
\]

\[
  (M^\intercal \tilde S - M^\intercal \tilde S
      Q(LL^\intercal)^{-1} Q^\intercal \tilde S M)^{-1}
        (\tilde S - \tilde S Q (LL^\intercal)^{-1} Q^\intercal \tilde S)
          \tilde \mu
\] -->

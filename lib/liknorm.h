void initialize(int n);
void destroy(void);
void binomial_moments(double size, double *k, double *n, double eta, double tau,
                      double *log_zeroth, double *mean, double *variance);
void poisson_moments(double size, double *k, double eta, double tau,
                     double *log_zeroth, double *mean, double *variance);

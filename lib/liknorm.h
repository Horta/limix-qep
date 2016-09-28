void initialize(int n);
void destroy(void);
void moments_scale(int likname_id, double *y, double *aphi,
               double *normal_tau, double *normal_eta, int n,
               double *log_zeroth, double *mean, double *variance);
void moments_noscale(int likname_id, double *y,
              double *normal_tau, double *normal_eta, int n,
              double *log_zeroth, double *mean, double *variance);

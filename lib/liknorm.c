#include "liknorm/liknorm.h"
#include <math.h>

static LikNormMachine *machine;

const char *liknames[] = { "binomial",    "bernoulli",    "poisson",    "gamma",
                           "exponential", "geometric" };


void initialize(int n)
{
  machine = liknorm_create_machine(350);
}

void destroy(void)
{
  liknorm_destroy_machine(machine);
}

void moments(int likname_id, double *y, double *aphi,
             double *normal_tau, double *normal_eta, int n,
             double *log_zeroth, double *mean, double *variance)
{
  Normal normal;
  ExpFam ef;
  log_partition *lp = get_log_partition(liknames[likname_id]);

  get_interval(liknames[likname_id], &(ef.left), &(ef.right));

  for (int i = 0; i < n; ++i)
  {
    ef.y           = y[i];
    ef.aphi        = aphi[i];
    ef.log_aphi    = log(ef.aphi);
    ef.lp          = lp;
    normal.tau     = normal_tau[i];
    normal.eta     = normal_eta[i];
    normal.log_tau = -log(normal.tau);

    liknorm_integrate(machine,
                      &ef,
                      &normal,
                      log_zeroth + i,
                      mean + i,
                      variance + i);
  }
}

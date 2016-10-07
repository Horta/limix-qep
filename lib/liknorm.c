#include "liknorm/liknorm.h"

static LikNormMachine *machine;

void initialize(int n) { machine = liknorm_create_machine(n); }

void destroy(void) { liknorm_destroy_machine(machine); }

void binomial_moments(double size, double *k, double *n, double eta, double tau,
                      double *log_zeroth, double *mean, double *variance) {

  for (size_t i = 0; i < size; i++) {
    liknorm_set_binomial(machine, k[i], n[i]);
    liknorm_moments(machine, log_zeroth + i, mean + i, variance + i);
  }
}

void poisson_moments(double size, double *k, double eta, double tau,
                     double *log_zeroth, double *mean, double *variance) {

  for (size_t i = 0; i < size; i++) {
    liknorm_set_poisson(machine, k[i]);
    liknorm_moments(machine, log_zeroth + i, mean + i, variance + i);
  }
}

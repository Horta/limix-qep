#ifndef _CEPHES_H_
#define _CEPHES_H_
//
#include <math.h>
#include <float.h>
//
// #include "bessel/mconf.h"
// #include "cmath/mconf.h"
// #include "cprob/mconf.h"
// #include "ellf/mconf.h"
// #include "misc/mconf.h"
// #include "polyn/mconf.h"
//
// #include "cmath/protos.h"
// #include "ellf/protos.h"
//

extern double torch_cephes_ndtr ( double );
static double inline norm_cdf(double x)
{
  return ndtr(x);
}

double torch_cephes_incbi(double a, double b, double y);
static double inline beta_isf(double a, double b, double x)
{
  return torch_cephes_incbi(a, b, 1.0-x);
}

extern double log_ndtr(double a);
static double inline norm_logcdf(double x)
{
  return log_ndtr(x);
}

extern double torch_cephes_ndtri ( double );
static double inline norm_cdfi(double x)
{
  return torch_cephes_ndtri(x);
}

static double inline norm_pdf(x)
{
  // static const double _norm_pdf_C = sqrt(2*M_PI);
  static const double _norm_pdf_C = 2.5066282746310002416123552393401041626930236816406250;
  return exp(-(x*x)/2.0) / _norm_pdf_C;
}

static double inline norm_logpdf(double x)
{
  // static const double _norm_pdf_logC = log(2*M_PI)/2.0;
  static const double _norm_pdf_logC = 0.9189385332046726695409688545623794198036193847656250;
  return -(x*x)/ 2.0 - _norm_pdf_logC;
}

static double inline norm_logpdf2(double x, double mu, double var)
{
  return norm_logpdf((x - mu) / sqrt(var)) - log(var)/2.0;
}

static double inline norm_sf(double x)
{
  return norm_cdf(-x);
}

static double inline norm_logsf(double x)
{
  return norm_logcdf(-x);
}
//
// static double inline betaln(double N, double K)
// {
//   return lbeta(N, K);
// }
//
static double inline logaddexp(double x, double y)
{
  double tmp = x - y;

  if (x == y)
    return x + M_LN2;

  if (tmp > 0)
    return x + log1p(exp(-tmp));
  else if (tmp <= 0)
    return y + log1p(exp(tmp));

  return tmp;
}

static double inline logaddexps(double x, double y, double sx, double sy)
{
  double tmp = x - y;

  double sxx = log(fabs(sx)) + x;
  double syy = log(fabs(sy)) + y;

  if (sxx == syy)
    if (sx * sy > 0)
      return sxx + M_LN2;
    else
      return -DBL_MAX;

  if (sx > 0 && sy > 0)
  {
    if (tmp > 0)
      return sxx + log1p((sy/sx) * exp(-tmp));
    else if (tmp <= 0)
      return syy + log1p((sx/sy) * exp(tmp));
  }
  else if (sx > 0)
    return sxx + log1p((sy/sx) * exp(-tmp));
  else
    return syy + log1p((sx/sy) * exp(tmp));

  return tmp;
}

static double inline logaddexpss(double x, double y, double sx,
                                 double sy, double* sign)
{
  // printf("sx sy: %.30f %.30f\n", sx, sy); fflush(stdout);
  double sxx = log(fabs(sx)) + x;
  double syy = log(fabs(sy)) + y;

  // printf("!!\n"); fflush(stdout);

  if (sxx == syy)
  {
    if (sx * sy > 0)
    {
      if (sx > 0)
        *sign = +1.0;
      else
        *sign = -1.0;
      return sxx + M_LN2;
    }
    else
    {
      *sign = 1.0;
      return -DBL_MAX;
    }
  }

  // printf("x-y: %.30f\n", x-y); fflush(stdout);
  // printf("sxx-syy: %.30f\n", sxx-syy); fflush(stdout);

  if (sxx > syy)
  {
    if (sx >= 0.0)
      *sign = +1.0;
    else
      *sign = -1.0;
  }
  else
  {
    if (sy >= 0.0)
      *sign = +1.0;
    else
      *sign = -1.0;
  }

  sx *= *sign;
  sy *= *sign;
  return logaddexps(x, y, sx, sy);
}

// ln(sx*exp(x) + sy*exp(y)), -1
//
// ln(|sx|) + x - ln(|sx|exp(x)) + ln(sx*exp(x) + sy*exp(y))
// ln(|sx|) + x + ln(sign(sx) + sy/|sx| * exp(y-x))
// ln(|sx|) + x + ln(sign(sx) * (1 + sy/sx * exp(y-x)))

extern double torch_cephes_lbeta ( double a, double b );
static double inline binomln(double N, double K)
{
  return -torch_cephes_lbeta(1 + N - K, 1 + K) - log1p(N);
}

#endif

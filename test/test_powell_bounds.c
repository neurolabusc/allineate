#include <float.h>
#include <limits.h>
#include <stdio.h>

extern int powell_newuoa(int ndim, double *x, double rstart, double rend,
                         int maxcall, double (*ufunc)(int, double *));
extern int powell_newuoa_con(int ndim, double *x, double *xbot, double *xtop,
                             int nrand, double rstart, double rend, int maxcall,
                             double (*ufunc)(int, double *));
extern void powell_set_mfac(float mm, float aa);

static int callback_calls = 0;

static double objective(int n, double *x) {
    callback_calls++;
    return n > 0 ? x[0] * x[0] : 0.0;
}

int main(void) {
    double x = 0.5, bot = 0.0, top = 1.0;

    int rc = powell_newuoa(INT_MAX, &x, 1.0, 0.01, 10, objective);
    if (rc >= 0 || callback_calls != 0) {
        fprintf(stderr, "unconstrained oversized dimension was not rejected early\n");
        return 1;
    }

    rc = powell_newuoa_con(INT_MAX, &x, &bot, &top, 0, 1.0, 0.01, 10,
                           objective);
    if (rc >= 0 || callback_calls != 0) {
        fprintf(stderr, "constrained oversized dimension was not rejected early\n");
        return 1;
    }

    powell_set_mfac(FLT_MAX, FLT_MAX);
    rc = powell_newuoa(2, &x, 1.0, 0.01, 10, objective);
    powell_set_mfac(0.0f, 0.0f);
    if (rc >= 0 || callback_calls != 0) {
        fprintf(stderr, "overflowing sampling factors were not rejected early\n");
        return 1;
    }

    puts("NEWUOA boundary tests passed");
    return 0;
}

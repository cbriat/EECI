#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static uint64_t s[4];
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}
static inline uint64_t next_rand(void) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}
static inline double rand_double(void) {
    return (next_rand() >> 11) * 0x1.0p-53;
}
static inline double rand_exp(double rate) {
    return -log(rand_double()) / rate;
}

/* SSA: returns mean_z1, mean_z2, cov_z1z2, var_z1, var_z2 */
void ssa_aic(double k, double eta, int n_events, int burn,
             double *out_mean_z1, double *out_mean_z2,
             double *out_cov, double *out_var_z1, double *out_var_z2) {
    int m = 1, p = 1;
    int z1 = (int)fmax(1, 1.0/k);
    int z2 = (int)fmax(1, k/eta);
    double sum_z1=0, sum_z2=0, sum_z1z2=0;
    double sum_z1sq=0, sum_z2sq=0, sum_t=0;

    for (int ev = 0; ev < n_events; ev++) {
        double a1 = k * z1;
        double a2 = (double)m;
        double a3 = (double)m;
        double a4 = (double)p;
        double a5 = 1.0;
        double a6 = (double)p;
        double a7 = eta * z1 * z2;
        double a_tot = a1+a2+a3+a4+a5+a6+a7;
        if (a_tot <= 0) a_tot = 1.0;
        double dt = rand_exp(a_tot);

        if (ev >= burn) {
            sum_z1 += z1 * dt;
            sum_z2 += z2 * dt;
            sum_z1z2 += (double)z1 * z2 * dt;
            sum_z1sq += (double)z1 * z1 * dt;
            sum_z2sq += (double)z2 * z2 * dt;
            sum_t += dt;
        }

        double r = rand_double() * a_tot;
        if (r < a1) { m++; }
        else if (r < a1+a2) { p++; }
        else if (r < a1+a2+a3) { m--; }
        else if (r < a1+a2+a3+a4) { p--; }
        else if (r < a1+a2+a3+a4+a5) { z1++; }
        else if (r < a1+a2+a3+a4+a5+a6) { z2++; }
        else { z1--; z2--; }
    }

    if (sum_t > 0) {
        double mz1 = sum_z1/sum_t, mz2 = sum_z2/sum_t;
        *out_mean_z1 = mz1;
        *out_mean_z2 = mz2;
        *out_cov = sum_z1z2/sum_t - mz1*mz2;
        *out_var_z1 = sum_z1sq/sum_t - mz1*mz1;
        *out_var_z2 = sum_z2sq/sum_t - mz2*mz2;
    }
}

int main(int argc, char *argv[]) {
    double k = atof(argv[1]);
    int n_eta = atoi(argv[2]);
    int nevents = atoi(argv[3]);
    int burn = atoi(argv[4]);

    double eta_vals[500];
    for (int i = 0; i < n_eta; i++) scanf("%lf", &eta_vals[i]);

    s[0]=12345678901234ULL; s[1]=98765432109876ULL;
    s[2]=11111111111111ULL; s[3]=22222222222222ULL;

    for (int i = 0; i < n_eta; i++) {
        double mz1, mz2, cov, vz1, vz2;
        ssa_aic(k, eta_vals[i], nevents, burn, &mz1, &mz2, &cov, &vz1, &vz2);
        printf("%.8f %.8f %.8f %.8f %.8f\n", mz1, mz2, cov, vz1, vz2);
        fflush(stdout);
        fprintf(stderr, "eta=%.2f done\n", eta_vals[i]);
    }
    return 0;
}

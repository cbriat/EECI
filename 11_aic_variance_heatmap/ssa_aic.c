#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* Xoshiro256** PRNG */
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

/* SSA for AIC + gene expression. Returns variance of P. */
double ssa_aic(double k, double eta, int n_events, int burn) {
    int m = 1, p = 1;
    int z1 = (int)fmax(1, 1.0/k);
    int z2 = (int)fmax(1, k/eta);
    double sum_p = 0, sum_p2 = 0, sum_t = 0;

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
            sum_p += p * dt;
            sum_p2 += (double)p * p * dt;
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
        double mean = sum_p / sum_t;
        double mean2 = sum_p2 / sum_t;
        return mean2 - mean*mean;
    }
    return -1;
}

int main(int argc, char *argv[]) {
    int nk = atoi(argv[1]);
    int neta = atoi(argv[2]);
    int nevents = atoi(argv[3]);
    int burn = atoi(argv[4]);

    /* Read k and eta arrays from stdin */
    double *k_vals = malloc(nk * sizeof(double));
    double *eta_vals = malloc(neta * sizeof(double));
    for (int j = 0; j < nk; j++) scanf("%lf", &k_vals[j]);
    for (int i = 0; i < neta; i++) scanf("%lf", &eta_vals[i]);

    /* Seed PRNG */
    s[0] = 12345678901234ULL; s[1] = 98765432109876ULL;
    s[2] = 11111111111111ULL; s[3] = 22222222222222ULL;

    for (int i = 0; i < neta; i++) {
        for (int j = 0; j < nk; j++) {
            double var = ssa_aic(k_vals[j], eta_vals[i], nevents, burn);
            printf("%.6f ", var);
        }
        printf("\n");
        fflush(stdout);
        fprintf(stderr, "row %d/%d done\n", i+1, neta);
    }

    free(k_vals); free(eta_vals);
    return 0;
}

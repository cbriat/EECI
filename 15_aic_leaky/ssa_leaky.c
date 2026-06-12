#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static uint64_t s[4];
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}
static inline uint64_t next_rand(void) {
    const uint64_t result = rotl(s[1]*5, 7)*9;
    const uint64_t t = s[1] << 17;
    s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3];
    s[2]^=t; s[3]=rotl(s[3],45);
    return result;
}
static inline double rand_double(void) {
    return (next_rand()>>11)*0x1.0p-53;
}
static inline double rand_exp(double rate) {
    return -log(rand_double())/rate;
}

/* Leaky AIC SSA. Returns mean_p, mean_z1, mean_z2. */
void ssa_leaky(double k, double eta, double gc,
               int n_events, int burn,
               double *out_mp, double *out_mz1, double *out_mz2) {
    int m=1, p=1, z1=1, z2=1;
    double sp=0, sz1=0, sz2=0, st=0;

    for (int ev=0; ev<n_events; ev++) {
        double a1 = k*z1;          /* Z1 -> Z1+mRNA */
        double a2 = (double)m;     /* mRNA -> mRNA+P */
        double a3 = (double)m;     /* mRNA -> 0 */
        double a4 = (double)p;     /* P -> 0 */
        double a5 = 1.0;           /* 0 -> Z1 */
        double a6 = (double)p;     /* P -> P+Z2 */
        double a7 = eta*z1*z2;     /* Z1+Z2 -> 0 */
        double a8 = gc*z1;         /* Z1 -> 0 (leak) */
        double a9 = gc*z2;         /* Z2 -> 0 (leak) */
        double at = a1+a2+a3+a4+a5+a6+a7+a8+a9;
        if (at<=0) at=1.0;
        double dt = rand_exp(at);

        if (ev>=burn) {
            sp += p*dt; sz1 += z1*dt; sz2 += z2*dt; st += dt;
        }

        double r = rand_double()*at;
        double c = 0;
        if (r < (c+=a1)) { m++; }
        else if (r < (c+=a2)) { p++; }
        else if (r < (c+=a3)) { m--; }
        else if (r < (c+=a4)) { p--; }
        else if (r < (c+=a5)) { z1++; }
        else if (r < (c+=a6)) { z2++; }
        else if (r < (c+=a7)) { z1--; z2--; }
        else if (r < (c+=a8)) { z1--; }
        else                  { z2--; }
    }
    if (st>0) { *out_mp=sp/st; *out_mz1=sz1/st; *out_mz2=sz2/st; }
}

int main(int argc, char *argv[]) {
    double k = atof(argv[1]);
    double eta = atof(argv[2]);
    int ngc = atoi(argv[3]);
    int nev = atoi(argv[4]);
    int burn = atoi(argv[5]);

    double gc_vals[500];
    for (int i=0; i<ngc; i++) scanf("%lf", &gc_vals[i]);

    s[0]=12345678901234ULL; s[1]=98765432109876ULL;
    s[2]=11111111111111ULL; s[3]=22222222222222ULL;

    for (int i=0; i<ngc; i++) {
        double mp, mz1, mz2;
        ssa_leaky(k, eta, gc_vals[i], nev, burn, &mp, &mz1, &mz2);
        printf("%.8f %.8f %.8f\n", mp, mz1, mz2);
        fflush(stdout);
    }
    return 0;
}

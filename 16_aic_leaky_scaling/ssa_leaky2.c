#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static uint64_t s_rng[4];
static inline uint64_t rotl(const uint64_t x, int k){return(x<<k)|(x>>(64-k));}
static inline uint64_t next_rand(void){
    const uint64_t result=rotl(s_rng[1]*5,7)*9;
    const uint64_t t=s_rng[1]<<17;
    s_rng[2]^=s_rng[0];s_rng[3]^=s_rng[1];s_rng[1]^=s_rng[2];
    s_rng[0]^=s_rng[3];s_rng[2]^=t;s_rng[3]=rotl(s_rng[3],45);
    return result;
}
static inline double rnd(void){return(next_rand()>>11)*0x1.0p-53;}
static inline double rexp(double r){return -log(rnd())/r;}

double ssa_leaky(double k, double eta, double gc,
                 double mu, double theta, int nev, int burn){
    /* Start near steady state: z1~1, z2~mu/(eta*1)=mu/eta */
    int m=1, p=1, z1=1, z2=(int)fmax(1,mu/eta);
    double sp=0, st=0;
    for(int ev=0;ev<nev;ev++){
        double a1=k*z1, a2=(double)m, a3=(double)m, a4=(double)p;
        double a5=mu, a6=theta*p, a7=eta*z1*z2;
        double a8=gc*z1, a9=gc*z2;
        double at=a1+a2+a3+a4+a5+a6+a7+a8+a9;
        if(at<=0) at=1;
        double dt=rexp(at);
        if(ev>=burn){sp+=p*dt;st+=dt;}
        double r=rnd()*at,c=0;
        if(r<(c+=a1))m++;
        else if(r<(c+=a2))p++;
        else if(r<(c+=a3))m--;
        else if(r<(c+=a4))p--;
        else if(r<(c+=a5))z1++;
        else if(r<(c+=a6))z2++;
        else if(r<(c+=a7)){z1--;z2--;}
        else if(r<(c+=a8))z1--;
        else z2--;
    }
    return(st>0)?sp/st:-1;
}

int main(int argc, char *argv[]){
    double k=atof(argv[1]),gc=atof(argv[2]);
    int np=atoi(argv[3]),nev=atoi(argv[4]),burn=atoi(argv[5]);
    s_rng[0]=123456789ULL;s_rng[1]=987654321ULL;
    s_rng[2]=111111111ULL;s_rng[3]=222222222ULL;
    for(int i=0;i<np;i++){
        double sf,eta;
        scanf("%lf %lf",&sf,&eta);
        double mp=ssa_leaky(k,eta,gc,sf,sf,nev,burn);
        printf("%.8f\n",mp);
        fflush(stdout);
    }
    return 0;
}

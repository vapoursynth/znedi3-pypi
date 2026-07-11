// Validate the full nnedi3_weights_hvsym.bin pipeline: load generators -> fold -> predict,
// and check the folded predictor (C + AVX-512) == a dense predictor on the materialised experts.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <immintrin.h>
using std::vector;

static const int XDIM[7]={8,16,32,48,8,16,32}, YDIM[7]={6,6,6,6,4,4,4}, NNS[5]={16,32,64,128,256};
static const float HC[4][4]={{1,1,1,1},{1,-1,1,-1},{1,1,-1,-1},{1,-1,-1,1}};

struct Orbit { int pos[4]; float wscale; };
static vector<Orbit> build_orbits(int nm1,int m){ int C=(nm1-1)/2; vector<Orbit> o;
    for(int cp=0;cp<C;++cp){int c0=cp,c1=nm1-1-cp; for(int rp=0;rp<m/2;++rp){int r0=rp,r1=m-1-rp; o.push_back({{r0*nm1+c0,r0*nm1+c1,r1*nm1+c0,r1*nm1+c1},1.0f});}}
    for(int rp=0;rp<m/2;++rp){int r0=rp,r1=m-1-rp; o.push_back({{r0*nm1+C,r0*nm1+C,r1*nm1+C,r1*nm1+C},0.5f});}
    return o; }
static vector<float> flipf(const vector<float>&g,int nm1,int m,int f){ vector<float> out(g.size());
    for(int r=0;r<m;++r)for(int c=0;c<nm1;++c){int rr=(f&2)?m-1-r:r,cc=(f&1)?nm1-1-c:c; out[r*nm1+c]=g[rr*nm1+cc];} return out; }
static float elliott(float x){return x/(1.0f+std::fabs(x));}
static float smexp(float x){return std::exp(std::min(std::max(x,-80.0f),80.0f));}

// znedi3's weights.cpp subtract_mean(), applied to materialised experts. Preserves flip symmetry
// (mean_filter is symmetric; per-filter means are equal within a flip orbit).
static void subtract_mean_std(vector<vector<float>>&SM,vector<vector<float>>&EL,vector<float>&SMB,vector<float>&ELB,int nns,int D){
    vector<double> smm(nns),elm(nns),mf(D,0.0);
    for(int nn=0;nn<nns;++nn){ double s=0,e=0; for(int k=0;k<D;++k){s+=SM[nn][k];e+=EL[nn][k];} smm[nn]=s/D; elm[nn]=e/D;
        for(int k=0;k<D;++k) mf[k]+=SM[nn][k]-smm[nn]; }
    for(int k=0;k<D;++k) mf[k]/=nns;
    double mb=0; for(int nn=0;nn<nns;++nn) mb+=SMB[nn]; mb/=nns;
    for(int nn=0;nn<nns;++nn){ for(int k=0;k<D;++k){ SM[nn][k]-=(float)(smm[nn]+mf[k]); EL[nn][k]-=(float)elm[nn]; } SMB[nn]-=(float)mb; }
}

// one predictor config's generators, straight from the file
struct Gen { int nm1,m,D,ng; vector<float> sg,eg,sb,eb; };   // sg/eg: [ng*D] (q-specific)

// ---- AVX-512 folded mat-vec + combine (from sym_kernel.cpp, mask-free) ----
static void folded_apply_avx512(const float* const wf[4], const float* const xh[4], int R, int ng,
                                const float* sbias, const float* ebias, const float scale[4],
                                float* as, float* ae){
    int twong=2*ng, nns=4*ng; const __m512 q=_mm512_set1_ps(0.25f);
    const float *X0=xh[0],*X1=xh[1],*X2=xh[2],*X3=xh[3];
    for(int b=0;b<twong;b+=16){
        __m512 a00=_mm512_setzero_ps(),a01=_mm512_setzero_ps(),a02=_mm512_setzero_ps(),a03=_mm512_setzero_ps();
        __m512 a10=_mm512_setzero_ps(),a11=_mm512_setzero_ps(),a12=_mm512_setzero_ps(),a13=_mm512_setzero_ps();
        __m512 a20=_mm512_setzero_ps(),a21=_mm512_setzero_ps(),a22=_mm512_setzero_ps(),a23=_mm512_setzero_ps();
        __m512 a30=_mm512_setzero_ps(),a31=_mm512_setzero_ps(),a32=_mm512_setzero_ps(),a33=_mm512_setzero_ps();
        const float *W0=wf[0]+b,*W1=wf[1]+b,*W2=wf[2]+b,*W3=wf[3]+b;
        for(int o=0;o<R;++o){ __m512 w;
            w=_mm512_loadu_ps(W0+(size_t)o*twong); a00=_mm512_fmadd_ps(w,_mm512_set1_ps(X0[0*R+o]),a00);a01=_mm512_fmadd_ps(w,_mm512_set1_ps(X0[1*R+o]),a01);a02=_mm512_fmadd_ps(w,_mm512_set1_ps(X0[2*R+o]),a02);a03=_mm512_fmadd_ps(w,_mm512_set1_ps(X0[3*R+o]),a03);
            w=_mm512_loadu_ps(W1+(size_t)o*twong); a10=_mm512_fmadd_ps(w,_mm512_set1_ps(X1[0*R+o]),a10);a11=_mm512_fmadd_ps(w,_mm512_set1_ps(X1[1*R+o]),a11);a12=_mm512_fmadd_ps(w,_mm512_set1_ps(X1[2*R+o]),a12);a13=_mm512_fmadd_ps(w,_mm512_set1_ps(X1[3*R+o]),a13);
            w=_mm512_loadu_ps(W2+(size_t)o*twong); a20=_mm512_fmadd_ps(w,_mm512_set1_ps(X2[0*R+o]),a20);a21=_mm512_fmadd_ps(w,_mm512_set1_ps(X2[1*R+o]),a21);a22=_mm512_fmadd_ps(w,_mm512_set1_ps(X2[2*R+o]),a22);a23=_mm512_fmadd_ps(w,_mm512_set1_ps(X2[3*R+o]),a23);
            w=_mm512_loadu_ps(W3+(size_t)o*twong); a30=_mm512_fmadd_ps(w,_mm512_set1_ps(X3[0*R+o]),a30);a31=_mm512_fmadd_ps(w,_mm512_set1_ps(X3[1*R+o]),a31);a32=_mm512_fmadd_ps(w,_mm512_set1_ps(X3[2*R+o]),a32);a33=_mm512_fmadd_ps(w,_mm512_set1_ps(X3[3*R+o]),a33);
        }
        int gg=b<ng?b:b-ng; const float* bias=b<ng?sbias:ebias; float* act=b<ng?as:ae; __m512 bs=_mm512_loadu_ps(bias+gg);
        #define CMB(p,P0,P1,P2,P3){__m512 s01=_mm512_add_ps(P0,P1),d01=_mm512_sub_ps(P0,P1),s23=_mm512_add_ps(P2,P3),d23=_mm512_sub_ps(P2,P3);__m512 sc=_mm512_set1_ps(scale[p]);\
            _mm512_storeu_ps(act+(size_t)(p)*nns+0*ng+gg,_mm512_fmadd_ps(sc,_mm512_mul_ps(q,_mm512_add_ps(s01,s23)),bs));_mm512_storeu_ps(act+(size_t)(p)*nns+1*ng+gg,_mm512_fmadd_ps(sc,_mm512_mul_ps(q,_mm512_add_ps(d01,d23)),bs));\
            _mm512_storeu_ps(act+(size_t)(p)*nns+2*ng+gg,_mm512_fmadd_ps(sc,_mm512_mul_ps(q,_mm512_sub_ps(s01,s23)),bs));_mm512_storeu_ps(act+(size_t)(p)*nns+3*ng+gg,_mm512_fmadd_ps(sc,_mm512_mul_ps(q,_mm512_sub_ps(d01,d23)),bs));}
        CMB(0,a00,a10,a20,a30)CMB(1,a01,a11,a21,a31)CMB(2,a02,a12,a22,a32)CMB(3,a03,a13,a23,a33)
        #undef CMB
    }
}

int main(int argc, char** argv){
    const char* path = argc > 1 ? argv[1] : "E:\\train2017\\hvsym_test.bin";
    FILE* fp=fopen(path,"rb"); if(!fp){printf("cannot open %s\n",path);return 1;}
    char magic[8]; fread(magic,1,8,fp);
    if(memcmp(magic,"HVSYMB1\0",8)!=0){printf("bad magic\n");return 1;}
    auto rd=[&](float* dst,size_t n){ if(fread(dst,4,n,fp)!=n){printf("short read\n");exit(1);} };

    std::mt19937 rng(99); std::uniform_real_distribution<float> ud(0,1);
    double worstC=0, worstV=0;
    for(int et=0; et<2; ++et) for(int ni=0; ni<5; ++ni) for(int si=0; si<7; ++si){
        int nns=NNS[ni], ng=nns/4, nm1=XDIM[si]-1, m=YDIM[si], D=nm1*m;
        auto orbits=build_orbits(nm1,m); int R=(int)orbits.size(), Rp=((R+15)/16)*16;
        for(int qi=0; qi<2; ++qi){
            Gen g; g.nm1=nm1; g.m=m; g.D=D; g.ng=ng; g.sg.resize(ng*D); g.eg.resize(ng*D); g.sb.resize(ng); g.eb.resize(ng);
            rd(g.sg.data(),ng*D); rd(g.eg.data(),ng*D); rd(g.sb.data(),ng); rd(g.eb.data(),ng);
            // materialise experts (dense reference)
            vector<vector<float>> SM(nns),EL(nns); vector<float> SMB(nns),ELB(nns);
            for(int f=0;f<4;++f)for(int gi=0;gi<ng;++gi){int i=f*ng+gi;
                vector<float> gs(g.sg.begin()+gi*D,g.sg.begin()+(gi+1)*D), ge(g.eg.begin()+gi*D,g.eg.begin()+(gi+1)*D);
                SM[i]=flipf(gs,nm1,m,f); EL[i]=flipf(ge,nm1,m,f); SMB[i]=g.sb[gi]; ELB[i]=g.eb[gi]; }
            subtract_mean_std(SM,EL,SMB,ELB,nns,D);        // mean-fold; experts[0..ng-1] are now the mean-folded generators
            // fold the mean-folded generators -> wf[c] interleaved [Rp][2*ng]
            vector<vector<float>> wf(4, vector<float>(Rp*2*ng,0.0f));
            for(int c=0;c<4;++c) for(int o=0;o<R;++o) for(int gi=0;gi<ng;++gi){
                float s=0,e=0; for(int k=0;k<4;++k){s+=HC[c][k]*SM[gi][orbits[o].pos[k]]; e+=HC[c][k]*EL[gi][orbits[o].pos[k]];}
                wf[c][o*2*ng+gi]=orbits[o].wscale*s; wf[c][o*2*ng+ng+gi]=orbits[o].wscale*e; }
            const float* wfp[4]={wf[0].data(),wf[1].data(),wf[2].data(),wf[3].data()};

            for(int t=0;t<8;++t){
                vector<vector<float>> win(4,vector<float>(D)); float scale[4],mean4[4],std4[4];
                for(int p=0;p<4;++p){ double sum=0,sq=0; for(int k=0;k<D;++k){win[p][k]=ud(rng);sum+=win[p][k];sq+=(double)win[p][k]*win[p][k];}
                    mean4[p]=(float)(sum/D); float var=(float)(sq/D-(double)mean4[p]*mean4[p]); std4[p]=var<1e-6f?0:std::sqrt(var); scale[p]=std4[p]==0?0:1.0f/std4[p]; }
                // fold windows -> xh[c] [4][Rp]
                vector<vector<float>> xh(4,vector<float>(4*Rp,0.0f));
                for(int p=0;p<4;++p)for(int o=0;o<R;++o)for(int c=0;c<4;++c){float s=0;for(int k=0;k<4;++k)s+=HC[c][k]*win[p][orbits[o].pos[k]];xh[c][p*Rp+o]=s;}
                const float* xhp[4]={xh[0].data(),xh[1].data(),xh[2].data(),xh[3].data()};
                // final value via shared scalar softmax+wae5
                auto finish=[&](const float* a,const float* e,int p){ float vs=0,ws=0; for(int i=0;i<nns;++i){float s=smexp(a[i]);vs+=s*elliott(e[i]);ws+=s;} return ws>1e-10f?mean4[p]+5.0f*vs/ws*std4[p]:mean4[p]; };
                // dense
                for(int p=0;p<4;++p){ vector<float> a(nns),e(nns);
                    for(int i=0;i<nns;++i){float da=0,de=0;for(int k=0;k<D;++k){da+=SM[i][k]*win[p][k];de+=EL[i][k]*win[p][k];}a[i]=da*scale[p]+SMB[i];e[i]=de*scale[p]+ELB[i];}
                    float dense=finish(a.data(),e.data(),p);
                    // folded C
                    vector<float> ac(nns),ec(nns);
                    for(int gi=0;gi<ng;++gi){ float Ps[4],Pe[4];
                        for(int c=0;c<4;++c){float ps=0,pe=0;for(int o=0;o<R;++o){ps+=xh[c][p*Rp+o]*wf[c][o*2*ng+gi];pe+=xh[c][p*Rp+o]*wf[c][o*2*ng+ng+gi];}Ps[c]=ps;Pe[c]=pe;}
                        for(int f=0;f<4;++f){float ds=0,de=0;for(int c=0;c<4;++c){ds+=HC[c][f]*Ps[c];de+=HC[c][f]*Pe[c];}ac[f*ng+gi]=0.25f*ds*scale[p]+SMB[gi];ec[f*ng+gi]=0.25f*de*scale[p]+ELB[gi];} }
                    worstC=std::max(worstC,(double)std::fabs(finish(ac.data(),ec.data(),p)-dense));
                }
                // folded AVX-512 (nns>=64, ng%16==0)
                if(ng%16==0){ vector<float> as(4*nns),ae(4*nns);
                    folded_apply_avx512(wfp,xhp,Rp,ng,SMB.data(),ELB.data(),scale,as.data(),ae.data());
                    for(int p=0;p<4;++p){ vector<float> a(nns),e(nns);
                        for(int i=0;i<nns;++i){float da=0,de=0;for(int k=0;k<D;++k){da+=SM[i][k]*win[p][k];de+=EL[i][k]*win[p][k];}a[i]=da*scale[p]+SMB[i];e[i]=de*scale[p]+ELB[i];}
                        worstV=std::max(worstV,(double)std::fabs(finish(as.data()+p*nns,ae.data()+p*nns,p)-finish(a.data(),e.data(),p))); }
                }
            }
        }
    }
    long pos=ftell(fp); fseek(fp,0,SEEK_END); long end=ftell(fp); fclose(fp);
    printf("loaded all 70 configs, consumed %ld/%ld bytes (%s)\n", pos, end, pos==end?"exact":"MISMATCH");
    printf("worst |folded-C  - dense| prediction = %.3e   (all sizes/nns)\n", worstC);
    printf("worst |folded-512 - dense| prediction = %.3e   (nns>=64)\n", worstV);
    printf("%s\n", (worstC<1e-4 && worstV<1e-4 && pos==end) ? "PASS" : "FAIL");
    return 0;
}

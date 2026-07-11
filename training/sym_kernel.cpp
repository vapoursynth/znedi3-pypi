// Standalone AVX-512 h+v-symmetric folded predictor core: validates the folded mat-vec +
// Klein-4 combine against the dense reference, and benchmarks it vs the stock dense sgemv.
// Window: (N-1)xM, target-centred. Folded weights: 4 channels x R orbits x (nns/4) generators.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <immintrin.h>

using std::vector;
static const float HC[4][4] = { {1,1,1,1},{1,-1,1,-1},{1,1,-1,-1},{1,-1,-1,1} };

struct Orbit { int pos[4]; float wscale; };
static vector<Orbit> build_orbits(int nm1, int m) {
    int C = (nm1 - 1) / 2; vector<Orbit> o;
    for (int cp = 0; cp < C; ++cp) { int c0=cp,c1=nm1-1-cp;
        for (int rp=0; rp<m/2; ++rp){ int r0=rp,r1=m-1-rp; o.push_back({{r0*nm1+c0,r0*nm1+c1,r1*nm1+c0,r1*nm1+c1},1.0f}); } }
    for (int rp=0; rp<m/2; ++rp){ int r0=rp,r1=m-1-rp; o.push_back({{r0*nm1+C,r0*nm1+C,r1*nm1+C,r1*nm1+C},0.5f}); }
    return o;
}
static vector<float> flipf(const vector<float>& g, int nm1, int m, int f) {
    vector<float> out(g.size());
    for (int r=0;r<m;++r) for (int c=0;c<nm1;++c){ int rr=(f&2)?m-1-r:r, cc=(f&1)?nm1-1-c:c; out[r*nm1+c]=g[rr*nm1+cc]; }
    return out;
}

// ---- AVX-512 folded mat-vec + combine -> activations for 4 pixels ----
// wfold[c] : interleaved [R][2*ng] (softmax gens 0..ng-1, elliott gens ng..2ng-1)
// xh[c]    : folded window [4 pixels][R]
// ng % 16 == 0 required.  Outputs act_softmax[4][nns], act_elliott[4][nns] (nns = 4*ng).
static void folded_apply_avx512(const float* const wfold[4], const float* const xh[4],
                                int R, int ng, const float* softmax_bias, const float* elliott_bias,
                                const float scale[4], float* act_softmax, float* act_elliott)
{
    int twong = 2 * ng, nns = 4 * ng;
    const __m512 q = _mm512_set1_ps(0.25f);
    const float *X0 = xh[0], *X1 = xh[1], *X2 = xh[2], *X3 = xh[3];

    for (int b = 0; b < twong; b += 16) {
        // acc[channel][pixel]: 16 accumulators kept in registers, combined in place.
        __m512 a00=_mm512_setzero_ps(),a01=_mm512_setzero_ps(),a02=_mm512_setzero_ps(),a03=_mm512_setzero_ps();
        __m512 a10=_mm512_setzero_ps(),a11=_mm512_setzero_ps(),a12=_mm512_setzero_ps(),a13=_mm512_setzero_ps();
        __m512 a20=_mm512_setzero_ps(),a21=_mm512_setzero_ps(),a22=_mm512_setzero_ps(),a23=_mm512_setzero_ps();
        __m512 a30=_mm512_setzero_ps(),a31=_mm512_setzero_ps(),a32=_mm512_setzero_ps(),a33=_mm512_setzero_ps();
        const float *W0 = wfold[0]+b, *W1 = wfold[1]+b, *W2 = wfold[2]+b, *W3 = wfold[3]+b;
        for (int o = 0; o < R; ++o) {
            __m512 w;
            w = _mm512_loadu_ps(W0 + (size_t)o*twong);
            a00=_mm512_fmadd_ps(w,_mm512_set1_ps(X0[0*R+o]),a00); a01=_mm512_fmadd_ps(w,_mm512_set1_ps(X0[1*R+o]),a01);
            a02=_mm512_fmadd_ps(w,_mm512_set1_ps(X0[2*R+o]),a02); a03=_mm512_fmadd_ps(w,_mm512_set1_ps(X0[3*R+o]),a03);
            w = _mm512_loadu_ps(W1 + (size_t)o*twong);
            a10=_mm512_fmadd_ps(w,_mm512_set1_ps(X1[0*R+o]),a10); a11=_mm512_fmadd_ps(w,_mm512_set1_ps(X1[1*R+o]),a11);
            a12=_mm512_fmadd_ps(w,_mm512_set1_ps(X1[2*R+o]),a12); a13=_mm512_fmadd_ps(w,_mm512_set1_ps(X1[3*R+o]),a13);
            w = _mm512_loadu_ps(W2 + (size_t)o*twong);
            a20=_mm512_fmadd_ps(w,_mm512_set1_ps(X2[0*R+o]),a20); a21=_mm512_fmadd_ps(w,_mm512_set1_ps(X2[1*R+o]),a21);
            a22=_mm512_fmadd_ps(w,_mm512_set1_ps(X2[2*R+o]),a22); a23=_mm512_fmadd_ps(w,_mm512_set1_ps(X2[3*R+o]),a23);
            w = _mm512_loadu_ps(W3 + (size_t)o*twong);
            a30=_mm512_fmadd_ps(w,_mm512_set1_ps(X3[0*R+o]),a30); a31=_mm512_fmadd_ps(w,_mm512_set1_ps(X3[1*R+o]),a31);
            a32=_mm512_fmadd_ps(w,_mm512_set1_ps(X3[2*R+o]),a32); a33=_mm512_fmadd_ps(w,_mm512_set1_ps(X3[3*R+o]),a33);
        }
        int gg = b < ng ? b : b - ng;
        const float* bias = b < ng ? softmax_bias : elliott_bias;
        float* act = b < ng ? act_softmax : act_elliott;
        __m512 bs = _mm512_loadu_ps(bias + gg);
        #define COMBINE(p, P0,P1,P2,P3) { \
            __m512 s01=_mm512_add_ps(P0,P1), d01=_mm512_sub_ps(P0,P1), s23=_mm512_add_ps(P2,P3), d23=_mm512_sub_ps(P2,P3); \
            __m512 sc=_mm512_set1_ps(scale[p]); \
            _mm512_storeu_ps(act+(size_t)(p)*nns+0*ng+gg, _mm512_fmadd_ps(sc,_mm512_mul_ps(q,_mm512_add_ps(s01,s23)),bs)); \
            _mm512_storeu_ps(act+(size_t)(p)*nns+1*ng+gg, _mm512_fmadd_ps(sc,_mm512_mul_ps(q,_mm512_add_ps(d01,d23)),bs)); \
            _mm512_storeu_ps(act+(size_t)(p)*nns+2*ng+gg, _mm512_fmadd_ps(sc,_mm512_mul_ps(q,_mm512_sub_ps(s01,s23)),bs)); \
            _mm512_storeu_ps(act+(size_t)(p)*nns+3*ng+gg, _mm512_fmadd_ps(sc,_mm512_mul_ps(q,_mm512_sub_ps(d01,d23)),bs)); }
        COMBINE(0, a00,a10,a20,a30) COMBINE(1, a01,a11,a21,a31)
        COMBINE(2, a02,a12,a22,a32) COMBINE(3, a03,a13,a23,a33)
        #undef COMBINE
    }
}

// ---- AVX2 folded mat-vec + combine (non-fused, mirrors PredictorSymmetricAVX::apply4) ----
static void folded_apply_avx2(const float* const wf[4], const float* const xh[4], int R, int ng,
                              const float* sbias, const float* ebias, const float scale[4],
                              float* P, float* as, float* ae){
    int twong=2*ng, nns=4*ng; const __m256 q=_mm256_set1_ps(0.25f);
    for(int c=0;c<4;++c){ const float* Wc=wf[c];
        const float* X0=xh[c]+0*R,*X1=xh[c]+1*R,*X2=xh[c]+2*R,*X3=xh[c]+3*R;
        for(int b=0;b<twong;b+=8){ __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(),a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
            for(int o=0;o<R;++o){ __m256 w=_mm256_loadu_ps(Wc+(size_t)o*twong+b);
                a0=_mm256_fmadd_ps(w,_mm256_broadcast_ss(X0+o),a0); a1=_mm256_fmadd_ps(w,_mm256_broadcast_ss(X1+o),a1);
                a2=_mm256_fmadd_ps(w,_mm256_broadcast_ss(X2+o),a2); a3=_mm256_fmadd_ps(w,_mm256_broadcast_ss(X3+o),a3); }
            _mm256_storeu_ps(P+((size_t)c*4+0)*twong+b,a0); _mm256_storeu_ps(P+((size_t)c*4+1)*twong+b,a1);
            _mm256_storeu_ps(P+((size_t)c*4+2)*twong+b,a2); _mm256_storeu_ps(P+((size_t)c*4+3)*twong+b,a3); } }
    for(int p=0;p<4;++p){ __m256 sc=_mm256_broadcast_ss(&scale[p]);
        for(int b=0;b<twong;b+=8){ __m256 P0=_mm256_loadu_ps(P+((size_t)0*4+p)*twong+b),P1=_mm256_loadu_ps(P+((size_t)1*4+p)*twong+b),P2=_mm256_loadu_ps(P+((size_t)2*4+p)*twong+b),P3=_mm256_loadu_ps(P+((size_t)3*4+p)*twong+b);
            __m256 s01=_mm256_add_ps(P0,P1),d01=_mm256_sub_ps(P0,P1),s23=_mm256_add_ps(P2,P3),d23=_mm256_sub_ps(P2,P3);
            int gg=b<ng?b:b-ng; const float* bp=b<ng?sbias:ebias; float* act=b<ng?as:ae; __m256 bs=_mm256_loadu_ps(bp+gg);
            _mm256_storeu_ps(act+(size_t)p*nns+0*ng+gg,_mm256_fmadd_ps(sc,_mm256_mul_ps(q,_mm256_add_ps(s01,s23)),bs));
            _mm256_storeu_ps(act+(size_t)p*nns+1*ng+gg,_mm256_fmadd_ps(sc,_mm256_mul_ps(q,_mm256_add_ps(d01,d23)),bs));
            _mm256_storeu_ps(act+(size_t)p*nns+2*ng+gg,_mm256_fmadd_ps(sc,_mm256_mul_ps(q,_mm256_sub_ps(s01,s23)),bs));
            _mm256_storeu_ps(act+(size_t)p*nns+3*ng+gg,_mm256_fmadd_ps(sc,_mm256_mul_ps(q,_mm256_sub_ps(d01,d23)),bs)); } }
}

// ---- stock dense AVX-512 sgemv (copy of znedi3 structure) for the benchmark baseline ----
template <int Step>
static void dense_sgemv_x4(const float* matrix, const float* vector, const float* bias,
                           int matrix_rows, int matrix_cols, float* result, int nns, const float scale[4])
{
    float* as = result; float* ae = result + 4 * nns;
    for (int i = 0; i < matrix_rows; i += Step) {
        __m512 ac[Step/16][4];
        for (int s = 0; s < Step/16; ++s) for (int p = 0; p < 4; ++p) ac[s][p] = _mm512_setzero_ps();
        for (int j = 0; j < matrix_cols; ++j) {
            __m512 x[4]; for (int p = 0; p < 4; ++p) x[p] = _mm512_set1_ps(vector[p*matrix_cols + j]);
            for (int s = 0; s < Step/16; ++s) {
                __m512 co = _mm512_loadu_ps(matrix + (size_t)j*matrix_rows + i + s*16);
                for (int p = 0; p < 4; ++p) ac[s][p] = _mm512_fmadd_ps(co, x[p], ac[s][p]);
            }
        }
        for (int s = 0; s < Step/16; ++s) {
            __m512 bs = _mm512_loadu_ps(bias + i + s*16);
            for (int p = 0; p < 4; ++p) {
                __m512 v = _mm512_fmadd_ps(_mm512_set1_ps(scale[p]), ac[s][p], bs);
                int idx = i + s*16;
                float* dst = idx >= nns ? ae + (idx - nns) : as + idx;
                _mm512_storeu_ps(dst + p*nns, v);
            }
        }
    }
}

int main() {
    struct Size { int xdim, ydim; }; Size sizes[] = { {8,6},{16,6},{32,6},{48,6},{8,4},{16,4},{32,4} };
    int nns_list[] = { 64, 128, 256 };
    std::mt19937 rng(7); std::normal_distribution<float> nd(0,0.3f); std::uniform_real_distribution<float> ud(0,1);

    double worst = 0, worst2 = 0;
    for (auto sz : sizes) {
        int nm1 = sz.xdim-1, m = sz.ydim, D = nm1*m; auto orbits = build_orbits(nm1, m); int R = (int)orbits.size();
        for (int nns : nns_list) {
            int ng = nns/4;
            vector<vector<float>> gs(ng, vector<float>(D)), ge(ng, vector<float>(D));
            vector<float> gsb(ng), geb(ng);
            for (int g=0;g<ng;++g){ for(int k=0;k<D;++k){gs[g][k]=nd(rng);ge[g][k]=nd(rng);} gsb[g]=nd(rng); geb[g]=nd(rng);}
            vector<vector<float>> SM(nns), EL(nns); vector<float> SMB(nns), ELB(nns);
            for (int f=0;f<4;++f) for (int g=0;g<ng;++g){ int i=f*ng+g; SM[i]=flipf(gs[g],nm1,m,f); EL[i]=flipf(ge[g],nm1,m,f); SMB[i]=gsb[g]; ELB[i]=geb[g]; }
            // fold weights -> wfold[c] interleaved [R][2*ng]
            vector<vector<float>> wf(4, vector<float>(R*2*ng, 0.0f));
            for (int c=0;c<4;++c) for (int o=0;o<R;++o) {
                for (int g=0;g<ng;++g){ float s=0,e=0; for(int k=0;k<4;++k){s+=HC[c][k]*gs[g][orbits[o].pos[k]]; e+=HC[c][k]*ge[g][orbits[o].pos[k]];}
                    wf[c][o*2*ng + g]      = orbits[o].wscale*s;
                    wf[c][o*2*ng + ng + g] = orbits[o].wscale*e; }
            }
            const float* wfp[4] = { wf[0].data(), wf[1].data(), wf[2].data(), wf[3].data() };
            // one random window x 4 pixels
            vector<vector<float>> win(4, vector<float>(D)); float scale[4];
            for (int p=0;p<4;++p){ double sum=0,sq=0; for(int k=0;k<D;++k){win[p][k]=ud(rng);sum+=win[p][k];sq+=(double)win[p][k]*win[p][k];}
                float mean=(float)(sum/D), var=(float)(sq/D-(double)mean*mean); scale[p]= var<1e-6f?0.0f:1.0f/std::sqrt(var); }
            // fold windows (scalar) -> xh[c] [4 pixels][R]
            vector<vector<float>> xh(4, vector<float>(4*R));
            for (int p=0;p<4;++p) for (int o=0;o<R;++o) for (int c=0;c<4;++c){ float s=0; for(int k=0;k<4;++k) s+=HC[c][k]*win[p][orbits[o].pos[k]]; xh[c][p*R+o]=s; }
            const float* xhp[4] = { xh[0].data(), xh[1].data(), xh[2].data(), xh[3].data() };
            // folded activations (AVX-512 and AVX2)
            vector<float> as(4*nns), ae(4*nns);
            folded_apply_avx512(wfp, xhp, R, ng, gsb.data(), geb.data(), scale, as.data(), ae.data());
            vector<float> as2(4*nns), ae2(4*nns), Pb(4*4*2*ng);
            folded_apply_avx2(wfp, xhp, R, ng, gsb.data(), geb.data(), scale, Pb.data(), as2.data(), ae2.data());
            // dense reference activations
            for (int p=0;p<4;++p) for (int i=0;i<nns;++i){ float da=0,de=0; for(int k=0;k<D;++k){da+=SM[i][k]*win[p][k]; de+=EL[i][k]*win[p][k];}
                worst=std::max(worst,(double)std::fabs(as[p*nns+i]-(da*scale[p]+SMB[i])));
                worst=std::max(worst,(double)std::fabs(ae[p*nns+i]-(de*scale[p]+ELB[i])));
                worst2=std::max(worst2,(double)std::fabs(as2[p*nns+i]-(da*scale[p]+SMB[i])));
                worst2=std::max(worst2,(double)std::fabs(ae2[p*nns+i]-(de*scale[p]+ELB[i]))); }
        }
    }
    printf("worst |folded-avx512 - dense| activation = %.3e   %s\n", worst, worst<1e-2?"PASS":"FAIL");
    printf("worst |folded-avx2   - dense| activation = %.3e   %s\n\n", worst2, worst2<1e-2?"PASS":"FAIL");

    // ---- benchmark: flagship 47x6 / 256, folded vs dense (4 pixels) ----
    {
        int nm1=47,m=6,D=nm1*m; auto orbits=build_orbits(nm1,m); int R=(int)orbits.size();
        int Rp=((R+15)/16)*16; int nns=256, ng=nns/4;                       // pad R to 16 for the fold/matvec
        vector<vector<float>> wf(4, vector<float>(Rp*2*ng, 0.0f)); for(auto&v:wf) for(auto&x:v) x=nd(rng);
        const float* wfp[4]={wf[0].data(),wf[1].data(),wf[2].data(),wf[3].data()};
        vector<float> gsb(ng,0.1f),geb(ng,0.1f); float scale[4]={1,1,1,1}; vector<float> as(4*nns),ae(4*nns);
        // precompute gather indices idx[k][o] (pad orbits -> index 0)
        vector<vector<int32_t>> idx(4, vector<int32_t>(Rp, 0));
        for (int o=0;o<R;++o) for (int k=0;k<4;++k) idx[k][o]=orbits[o].pos[k];
        vector<vector<float>> win(4, vector<float>(D)); for(auto&v:win) for(auto&x:v) x=ud(rng);
        vector<vector<float>> xh(4, vector<float>(4*Rp,0.0f));
        const float* xhp[4]={xh[0].data(),xh[1].data(),xh[2].data(),xh[3].data()};
        // SIMD (gather) window fold: 16 orbits at a time, 4 gathers + Klein-4 butterfly
        auto simd_fold = [&]{
            for (int p=0;p<4;++p) { const float* w=win[p].data();
                for (int o=0;o<Rp;o+=16) {
                    __m512 g0=_mm512_i32gather_ps(_mm512_loadu_si512(idx[0].data()+o), w, 4);
                    __m512 g1=_mm512_i32gather_ps(_mm512_loadu_si512(idx[1].data()+o), w, 4);
                    __m512 g2=_mm512_i32gather_ps(_mm512_loadu_si512(idx[2].data()+o), w, 4);
                    __m512 g3=_mm512_i32gather_ps(_mm512_loadu_si512(idx[3].data()+o), w, 4);
                    __m512 s01=_mm512_add_ps(g0,g1),d01=_mm512_sub_ps(g0,g1),s23=_mm512_add_ps(g2,g3),d23=_mm512_sub_ps(g2,g3);
                    _mm512_storeu_ps(xh[0].data()+p*Rp+o,_mm512_add_ps(s01,s23));
                    _mm512_storeu_ps(xh[1].data()+p*Rp+o,_mm512_add_ps(d01,d23));
                    _mm512_storeu_ps(xh[2].data()+p*Rp+o,_mm512_sub_ps(s01,s23));
                    _mm512_storeu_ps(xh[3].data()+p*Rp+o,_mm512_sub_ps(d01,d23));
                }
            }
        };
        vector<float> dm(D*2*nns); for(auto&x:dm) x=nd(rng); vector<float> dv(4*D); for(auto&x:dv) x=nd(rng);
        vector<float> dbias(2*nns,0.1f), dres(8*nns);
        auto bench=[&](const char* name, auto fn){ fn(); auto t0=std::chrono::high_resolution_clock::now();
            const int N=200000; for(int it=0;it<N;++it) fn();
            auto t1=std::chrono::high_resolution_clock::now(); double ns=std::chrono::duration<double,std::nano>(t1-t0).count()/N;
            printf("  %-24s %7.1f ns / 4px\n", name, ns); return ns; };
        double d  = bench("dense sgemv", [&]{ dense_sgemv_x4<64>(dm.data(), dv.data(), dbias.data(), nns*2, D, dres.data(), nns, scale); });
        double f  = bench("folded matvec+combine", [&]{ folded_apply_avx512(wfp, xhp, Rp, ng, gsb.data(), geb.data(), scale, as.data(), ae.data()); });
        double fl = bench("simd window fold", [&]{ simd_fold(); });
        double ff = bench("folded fold+matvec+comb", [&]{ simd_fold(); folded_apply_avx512(xhp, xhp, Rp, ng, gsb.data(), geb.data(), scale, as.data(), ae.data()); });
        printf("  -> matvec+combine %.2fx dense;  full (SIMD fold+matvec+combine) %.2fx dense\n", d/f, d/ff);
        printf("  weights: dense %.0f KB vs folded %.0f KB (%.1fx smaller)\n",
               D*2.0*nns*4/1024, Rp*2.0*ng*4*4/1024, (D*2.0*nns)/(Rp*2.0*ng*4));
    }
    return 0;
}

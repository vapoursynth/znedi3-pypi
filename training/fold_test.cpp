// Standalone validation of the Klein-4 (h+v) fold for (N-1)xM target-centred windows.
// Proves: folded dot products == dense materialised flip experts, and the full folded
// predictor (softmax + wae5) == the dense predictor, for every nnedi3 window size.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

using std::vector;

// Klein-4 character table over orbit order [e, h, v, hv]. Symmetric (Hadamard).
static const float HC[4][4] = {
    { 1,  1,  1,  1},   // (h+, v+)
    { 1, -1,  1, -1},   // (h-, v+)
    { 1,  1, -1, -1},   // (h+, v-)
    { 1, -1, -1,  1},   // (h-, v-)
};

struct Orbit { int pos[4]; float wscale; };  // wscale = 0.5 for the h-fixed centre column

// Orbits of an odd-width nm1 x m window (nm1 odd, m even). Flat index = r*nm1 + c.
// Reflection axis is the centre column C = (nm1-1)/2 (the target column) and the row gap.
static vector<Orbit> build_orbits(int nm1, int m) {
    int C = (nm1 - 1) / 2;
    vector<Orbit> orbits;
    for (int cp = 0; cp < C; ++cp) {                 // paired columns (c, 2C-c)
        int c0 = cp, c1 = nm1 - 1 - cp;
        for (int rp = 0; rp < m / 2; ++rp) {         // paired rows (r, m-1-r)
            int r0 = rp, r1 = m - 1 - rp;
            Orbit o{ { r0*nm1 + c0, r0*nm1 + c1, r1*nm1 + c0, r1*nm1 + c1 }, 1.0f };
            orbits.push_back(o);
        }
    }
    for (int rp = 0; rp < m / 2; ++rp) {             // centre column: h fixes it -> pos[0]==pos[1]
        int r0 = rp, r1 = m - 1 - rp;
        Orbit o{ { r0*nm1 + C, r0*nm1 + C, r1*nm1 + C, r1*nm1 + C }, 0.5f };
        orbits.push_back(o);
    }
    return orbits;
}

// Flip a generator filter g[m*nm1] (row-major) by group element f in {0=e,1=h,2=v,3=hv}.
static vector<float> flip(const vector<float>& g, int nm1, int m, int f) {
    vector<float> out(g.size());
    for (int r = 0; r < m; ++r)
        for (int c = 0; c < nm1; ++c) {
            int rr = (f & 2) ? (m - 1 - r) : r;
            int cc = (f & 1) ? (nm1 - 1 - c) : c;
            out[r*nm1 + c] = g[rr*nm1 + cc];
        }
    return out;
}

static float elliottf(float x) { return x / (1.0f + std::fabs(x)); }
static float softmax_expf(float x) { return std::exp(std::min(std::max(x, -80.0f), 80.0f)); }

// Full nnedi3 predictor output for one window, given materialised experts (dense reference).
static float predict_dense(const vector<vector<float>>& sm, const vector<vector<float>>& el,
                           const vector<float>& smb, const vector<float>& elb,
                           const vector<float>& win, float mean, float istd) {
    int nns = (int)sm.size(), D = (int)win.size();
    vector<float> a(nns), e(nns);
    for (int i = 0; i < nns; ++i) {
        float da = 0, de = 0;
        for (int k = 0; k < D; ++k) { da += sm[i][k]*win[k]; de += el[i][k]*win[k]; }
        a[i] = da*istd + smb[i];
        e[i] = de*istd + elb[i];
    }
    float vsum = 0, wsum = 0;
    for (int i = 0; i < nns; ++i) { float s = softmax_expf(a[i]); vsum += s*elliottf(e[i]); wsum += s; }
    return wsum > 1e-10f ? mean + 5.0f*vsum/wsum*istd : mean;   // note: istd carries std here (see caller)
}

int main() {
    struct Size { int xdim, ydim; };
    Size sizes[] = { {8,6},{16,6},{32,6},{48,6},{8,4},{16,4},{32,4} };
    int nns_list[] = { 16, 64, 256 };
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    double worst_dot = 0, worst_pred = 0;
    for (auto sz : sizes) {
        int nm1 = sz.xdim - 1, m = sz.ydim, D = nm1 * m;
        auto orbits = build_orbits(nm1, m);
        int R = (int)orbits.size();
        printf("size %dx%d -> (N-1)xM = %dx%d  D=%d  orbits R=%d  (dense cols %d, folded 4xR=%d, ratio %.2fx)\n",
               sz.xdim, sz.ydim, nm1, m, D, R, D, 4*R, (double)(2*D) / (double)(2*R));  // MAC ratio ~ D/R... see below

        for (int nns : nns_list) {
            int ng = nns / 4;                          // generators
            // random generators (softmax + elliott) and per-generator biases
            vector<vector<float>> gs(ng, vector<float>(D)), ge(ng, vector<float>(D));
            vector<float> gsb(ng), geb(ng);
            for (int g = 0; g < ng; ++g) {
                for (int k = 0; k < D; ++k) { gs[g][k] = nd(rng); ge[g][k] = nd(rng); }
                gsb[g] = nd(rng); geb[g] = nd(rng);
            }
            // materialise nns experts in [e,h,v,hv] blocks (shared bias within orbit)
            vector<vector<float>> SM(nns), EL(nns); vector<float> SMB(nns), ELB(nns);
            for (int f = 0; f < 4; ++f)
                for (int g = 0; g < ng; ++g) {
                    int i = f*ng + g;
                    SM[i] = flip(gs[g], nm1, m, f); EL[i] = flip(ge[g], nm1, m, f);
                    SMB[i] = gsb[g]; ELB[i] = geb[g];
                }
            // pre-fold the generator weights: ghat[bank][g][c][o]
            auto fold_w = [&](const vector<vector<float>>& gen) {
                vector<vector<vector<float>>> gh(ng, vector<vector<float>>(4, vector<float>(R)));
                for (int g = 0; g < ng; ++g)
                    for (int o = 0; o < R; ++o)
                        for (int c = 0; c < 4; ++c) {
                            float s = 0;
                            for (int k = 0; k < 4; ++k) s += HC[c][k]*gen[g][orbits[o].pos[k]];
                            gh[g][c][o] = orbits[o].wscale * s;
                        }
                return gh;
            };
            auto ghs = fold_w(gs), ghe = fold_w(ge);

            // test several random windows
            for (int t = 0; t < 64; ++t) {
                vector<float> win(D); double sum = 0, sq = 0;
                for (int k = 0; k < D; ++k) { win[k] = ud(rng); sum += win[k]; sq += (double)win[k]*win[k]; }
                float mean = (float)(sum / D);
                float var = (float)(sq / D - (double)mean*mean);
                float std = var < 1e-6f ? 0.0f : std::sqrt(var);
                float istd = std == 0 ? 0.0f : 1.0f/std;
                vector<float> xn(D); for (int k = 0; k < D; ++k) xn[k] = win[k];  // dot on raw window; scale applied after

                // --- dense activations ---
                vector<float> a_dense(nns), e_dense(nns);
                for (int i = 0; i < nns; ++i) {
                    float da = 0, de = 0;
                    for (int k = 0; k < D; ++k) { da += SM[i][k]*xn[k]; de += EL[i][k]*xn[k]; }
                    a_dense[i] = da; e_dense[i] = de;
                }
                // --- folded activations ---
                // fold window
                vector<vector<float>> xh(4, vector<float>(R));
                for (int o = 0; o < R; ++o)
                    for (int c = 0; c < 4; ++c) {
                        float s = 0; for (int k = 0; k < 4; ++k) s += HC[c][k]*xn[orbits[o].pos[k]];
                        xh[c][o] = s;
                    }
                auto fold_dot = [&](const vector<vector<vector<float>>>& gh, vector<float>& out) {
                    out.assign(nns, 0.0f);
                    for (int g = 0; g < ng; ++g) {
                        float P[4];
                        for (int c = 0; c < 4; ++c) { float p = 0; for (int o = 0; o < R; ++o) p += xh[c][o]*gh[g][c][o]; P[c] = p; }
                        for (int f = 0; f < 4; ++f) {
                            float d = 0; for (int c = 0; c < 4; ++c) d += HC[c][f]*P[c];
                            out[f*ng + g] = 0.25f*d;
                        }
                    }
                };
                vector<float> a_fold, e_fold; fold_dot(ghs, a_fold); fold_dot(ghe, e_fold);

                for (int i = 0; i < nns; ++i) {
                    worst_dot = std::max(worst_dot, (double)std::fabs(a_dense[i]-a_fold[i]));
                    worst_dot = std::max(worst_dot, (double)std::fabs(e_dense[i]-e_fold[i]));
                }
                // --- full predictor both ways ---
                auto finish = [&](const vector<float>& a, const vector<float>& e) {
                    float vsum = 0, wsum = 0;
                    for (int i = 0; i < nns; ++i) { float s = softmax_expf(a[i]*istd + SMB[i]); vsum += s*elliottf(e[i]*istd + ELB[i]); wsum += s; }
                    return wsum > 1e-10f ? mean + 5.0f*vsum/wsum*std : mean;
                };
                worst_pred = std::max(worst_pred, (double)std::fabs(finish(a_dense,e_dense) - finish(a_fold,e_fold)));
            }
        }
    }
    printf("\nworst |dense-folded| activation = %.3e\nworst |dense-folded| prediction = %.3e\n", worst_dot, worst_pred);
    printf("%s\n", (worst_dot < 1e-2 && worst_pred < 1e-4) ? "PASS" : "FAIL");
    return 0;
}

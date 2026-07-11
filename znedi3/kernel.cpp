#include <algorithm>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include "kernel.h"
#include "weights.h"
#include "znedi3_impl.h"

#ifdef ZNEDI3_X86
  #include "x86/kernel_x86.h"
#endif

namespace znedi3 {
namespace {

template <class T>
void integer_to_float(const void *src, void *dst, size_t n)
{
	const T *src_p = static_cast<const T *>(src);
	float *dst_p = static_cast<float *>(dst);

	std::transform(src_p, src_p + n, dst_p, [](T x) { return static_cast<float>(x); });
}

template <class T>
void float_to_integer(const void *src, void *dst, size_t n)
{
	const float *src_p = static_cast<const float *>(src);
	T *dst_p = static_cast<T *>(dst);

	std::transform(src_p, src_p + n, dst_p, [](float x)
	{
		x = std::min(std::max(x, 0.0f), static_cast<float>(std::numeric_limits<T>::max()));
		return static_cast<T>(std::lrint(x));
	});
}

void copy_float(const void *src, void *dst, size_t n)
{
	std::copy_n(static_cast<const unsigned char *>(src), n * sizeof(float), static_cast<unsigned char *>(dst));
}

float dot_product(const float *kernel, const float *input, unsigned n, float scale, float bias)
{
	float accum = 0.0f;

	for (unsigned i = 0; i < n; ++i) {
		accum += kernel[i] * input[i];
	}
	return accum * scale + bias;
}

float elliott(float x)
{
	return x / (1.0f + std::fabs(x));
}

float softmax_exp(float x)
{
	return std::exp(std::min(std::max(x, -80.0f), 80.0f));
}

void cubic_interpolation_c(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n)
{
	for (unsigned i = 0; i < n; ++i) {
		if (!prescreen[i])
			continue;

		float accum = 0.0f;
		accum += (-3.0f / 32.0f) * src[0][i];
		accum += (19.0f / 32.0f) * src[1][i];
		accum += (19.0f / 32.0f) * src[2][i];
		accum += (-3.0f / 32.0f) * src[3][i];

		dst[i] = accum;
	}
}


class PrescreenerOldC final : public Prescreener {
	PrescreenerOldCoefficients m_data;
public:
	PrescreenerOldC(const PrescreenerOldCoefficients &data, double half) :
		m_data(data)
	{
		subtract_mean(m_data, half);
	}

	size_t get_tmp_size() const noexcept override { return 0; }

	void process(const float * const src[4], unsigned char *prescreen, void *, unsigned n) const noexcept override
	{
		ptrdiff_t window_offset = 5;

		for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(n); ++j) {
			float input[48];
			float state[12];

			for (unsigned i = 0; i < 4; ++i) {
				std::copy_n(src[i] - window_offset + j, 12, input + i * 12);
			}

			// Layer 0.
			for (unsigned n = 0; n < 4; ++n) {
				state[n] = dot_product(m_data.kernel_l0[n], input, 48, 1.0f, m_data.bias_l0[n]);
			}
			std::transform(state + 1, state + 4, state + 1, elliott);

			// Layer 1.
			for (unsigned n = 0; n < 4; ++n) {
				state[n + 4] = dot_product(m_data.kernel_l1[n], state, 4, 1.0f, m_data.bias_l1[n]);
			}
			std::transform(state + 4, state + 8, state + 4, elliott);

			// Layer 2.
			for (unsigned n = 0; n < 4; ++n) {
				state[n + 8] = dot_product(m_data.kernel_l2[n], state, 8, 1.0f, m_data.bias_l2[n]);
			}

			prescreen[j] = std::max(state[10], state[11]) <= std::max(state[8], state[9]) ? UCHAR_MAX : 0;
		}
	}
};


class PrescreenerNewC final : public Prescreener {
	PrescreenerNewCoefficients m_data;

	void gather_input(const float *src, ptrdiff_t src_stride, float buf[48]) const
	{
		ptrdiff_t src_stride_f = src_stride / 4;

		for (unsigned i = 0; i < 4; ++i) {
			std::copy_n(src + i * src_stride_f, 16, buf + i * 16);
		}
	}
public:
	PrescreenerNewC(const PrescreenerNewCoefficients &data, double half) :
		m_data(data)
	{
		subtract_mean(m_data, half);
	}

	size_t get_tmp_size() const noexcept override { return 0; }

	void process(const float * const src[4], unsigned char *prescreen, void *, unsigned n) const noexcept override
	{
		ptrdiff_t window_offset = 6;

		for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(n); j += 4) {
			float input[64];
			float state[8];

			for (unsigned i = 0; i < 4; ++i) {
				std::copy_n(src[i] - window_offset + j, 16, input + i * 16);
			}

			for (unsigned n = 0; n < 4; ++n) {
				state[n] = dot_product(m_data.kernel_l0[n], input, 64, 1.0f, m_data.bias_l0[n]);
			}
			std::transform(state, state + 4, state, elliott);

			for (unsigned n = 0; n < 4; ++n) {
				state[n + 4] = dot_product(m_data.kernel_l1[n], state, 4, 1.0f, m_data.bias_l1[n]);
			}

			for (unsigned n = 0; n < 4; ++n) {
				prescreen[j + n] = state[n + 4] > 0.0f;
			}
		}
	}
};


class PredictorC final : public Predictor {
	PredictorModel m_model;
	bool m_use_q2;
	ptrdiff_t m_offset_x;

	size_t filter_offset(unsigned nn) const { return static_cast<size_t>(nn) * m_model.first.xdim * m_model.first.ydim; }

	const float *softmax_q1_filter(unsigned nn) const { return m_model.second.softmax_q1 + filter_offset(nn); }
	const float *elliott_q1_filter(unsigned nn) const { return m_model.second.elliott_q1 + filter_offset(nn); }
	const float *softmax_q2_filter(unsigned nn) const { return m_model.second.softmax_q2 + filter_offset(nn); }
	const float *elliott_q2_filter(unsigned nn) const { return m_model.second.elliott_q2 + filter_offset(nn); }

	void gather_input(const float * const *src, ptrdiff_t window_offset, float *buf, float mstd[4]) const
	{
		double sum = 0;
		double sum_sq = 0;

		for (ptrdiff_t i = 0; i < m_model.first.ydim; ++i) {
			for (ptrdiff_t j = 0; j < m_model.first.xdim; ++j) {
				float val = src[i][window_offset + j];

				buf[i * m_model.first.xdim + j] = val;
				sum += val;
				sum_sq += static_cast<double>(val) * val;
			}
		}

		mstd[0] = static_cast<float>(sum / (m_model.first.xdim * m_model.first.ydim));
		mstd[3] = 0.0f;

		double tmp = sum_sq / (m_model.first.xdim * m_model.first.ydim) - static_cast<double>(mstd[0]) * mstd[0];
		if (tmp < FLT_EPSILON) {
			mstd[1] = 0.0f;
			mstd[2] = 0.0f;
		} else {
			mstd[1] = static_cast<float>(std::sqrt(tmp));
			mstd[2] = 1.0f / mstd[1];
		}
	}

	void wae5(const float *softmax, const float *elliott, unsigned n, float mstd[4]) const
	{
		float vsum = 0.0f;
		float wsum = 0.0f;

		for (unsigned i = 0; i < n; ++i) {
			vsum += softmax[i] * znedi3::elliott(elliott[i]);
			wsum += softmax[i];
		}

		if (wsum > 1e-10f)
			mstd[3] += (5.0f * vsum) / wsum * mstd[1] + mstd[0];
		else
			mstd[3] += mstd[0];
	}
public:
	PredictorC(const PredictorModel &model, bool use_q2, ptrdiff_t offset_x = -1) :
		m_model{ copy_model(model) },
		m_use_q2{ use_q2 },
		m_offset_x{ offset_x >= 0 ? offset_x : static_cast<ptrdiff_t>(model.first.xdim) / 2 - 1 }
	{
		assert(model.first.xdim * model.first.ydim <= 48 * 6);
		subtract_mean(m_model);
	}

	size_t get_tmp_size() const noexcept override { return 0; }

	void process(const float * const *src, float *dst, const unsigned char *prescreen, void *, unsigned n) const noexcept override
	{
		ptrdiff_t window_offset_y = 3 - static_cast<ptrdiff_t>(m_model.first.ydim / 2);
		ptrdiff_t window_offset_x = m_offset_x;

		unsigned filter_size = m_model.first.xdim * m_model.first.ydim;
		unsigned nns = m_model.first.nns;

		for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
			if (prescreen[i])
				continue;

			float input[48 * 6];
			float activation[256 * 2];
			float mstd[4];

			gather_input(src + window_offset_y, i - window_offset_x, input, mstd);
			float scale = mstd[2];

			for (unsigned nn = 0; nn < nns; ++nn) {
				activation[nn] = dot_product(softmax_q1_filter(nn), input, filter_size, scale, m_model.second.softmax_bias_q1[nn]);
			}
			for (unsigned nn = 0; nn < nns; ++nn) {
				activation[m_model.first.nns + nn] = dot_product(elliott_q1_filter(nn), input, filter_size, scale, m_model.second.elliott_bias_q1[nn]);
			}

			std::transform(activation, activation + nns, activation, softmax_exp);
			wae5(activation, activation + nns, nns, mstd);

			if (m_use_q2) {
				for (unsigned nn = 0; nn < nns; ++nn) {
					activation[nn] = dot_product(softmax_q2_filter(nn), input, filter_size, scale, m_model.second.softmax_bias_q2[nn]);
				}
				for (unsigned nn = 0; nn < nns; ++nn) {
					activation[nns + nn] = dot_product(elliott_q2_filter(nn), input, filter_size, scale, m_model.second.elliott_bias_q2[nn]);
				}

				std::transform(activation, activation + nns, activation, softmax_exp);
				wae5(activation, activation + nns, nns, mstd);
			}

			dst[i] = mstd[3] / (m_use_q2 ? 2 : 1);
		}
	}
};

// Materialise the nns full experts (raw, width xdim-1) from the flip-orbit generators. Used by the
// portable fallback, which then runs the dense PredictorC (it applies subtract_mean itself).
PredictorModel materialise_symmetric(const SymmetricPredictorModel &model)
{
	unsigned xdim = model.first.xdim, ydim = model.first.ydim, nns = model.first.nns;
	unsigned nm1 = xdim - 1, ng = nns / 4, Dm1 = nm1 * ydim;

	PredictorModel out = allocate_model({ nm1, ydim, nns });
	auto flip = [&](const float *gen, float *dst, int f) {
		for (unsigned r = 0; r < ydim; ++r)
			for (unsigned c = 0; c < nm1; ++c) {
				unsigned rr = (f & 2) ? ydim - 1 - r : r, cc = (f & 1) ? nm1 - 1 - c : c;
				dst[r*nm1 + c] = gen[rr*nm1 + cc];
			}
	};
	auto mat = [&](const float *sg, const float *eg, const float *sb, const float *eb, float *SM, float *EL, float *SMB, float *ELB) {
		for (int f = 0; f < 4; ++f)
			for (unsigned gi = 0; gi < ng; ++gi) {
				unsigned i = f*ng + gi;
				flip(sg + gi*Dm1, SM + i*Dm1, f);
				flip(eg + gi*Dm1, EL + i*Dm1, f);
				SMB[i] = sb[gi]; ELB[i] = eb[gi];
			}
	};
	mat(model.second.softmax_gen_q1, model.second.elliott_gen_q1, model.second.softmax_bias_q1, model.second.elliott_bias_q1,
	    out.second.softmax_q1, out.second.elliott_q1, out.second.softmax_bias_q1, out.second.elliott_bias_q1);
	mat(model.second.softmax_gen_q2, model.second.elliott_gen_q2, model.second.softmax_bias_q2, model.second.elliott_bias_q2,
	    out.second.softmax_q2, out.second.elliott_q2, out.second.softmax_bias_q2, out.second.elliott_bias_q2);
	return out;
}

constexpr float SYM_HC[4][4] = { {1,1,1,1},{1,-1,1,-1},{1,1,-1,-1},{1,-1,-1,1} };

// Portable folded h+v-symmetric predictor: gathers the (N-1)-wide window, Klein-4-folds it, runs
// the folded mat-vec (~4x fewer MACs than dense) + combine, then softmax + wae5. Works for any
// nns and on any architecture. The weights are the mean-folded generators, Klein-4-transformed.
class PredictorSymmetricC final : public Predictor {
	unsigned m_nm1, m_ydim, m_nns, m_ng, m_R;
	ptrdiff_t m_off_x;
	bool m_use_q2;
	std::vector<int> m_pos;                       // [R][4] orbit positions
	std::vector<float> m_wscale;                  // [R] (0.5 for the centre column)
	std::vector<float> m_wf_s[2], m_wf_e[2];      // [4][ng][R] folded softmax / elliott, per quality
	std::vector<float> m_bias_s[2], m_bias_e[2];  // [ng]
public:
	PredictorSymmetricC(const SymmetricPredictorModel &model, bool use_q2) : m_use_q2{ use_q2 }
	{
		unsigned xdim = model.first.xdim;
		m_ydim = model.first.ydim; m_nns = model.first.nns;
		m_nm1 = xdim - 1; m_ng = m_nns / 4; m_off_x = static_cast<ptrdiff_t>(xdim) / 2 - 1;
		unsigned Dm1 = m_nm1 * m_ydim;

		int C = static_cast<int>(m_nm1 - 1) / 2, N = static_cast<int>(m_nm1);
		for (int cp = 0; cp < C; ++cp) {
			int c0 = cp, c1 = N - 1 - cp;
			for (int rp = 0; rp < static_cast<int>(m_ydim) / 2; ++rp) {
				int r0 = rp, r1 = static_cast<int>(m_ydim) - 1 - rp;
				m_pos.insert(m_pos.end(), { r0*N+c0, r0*N+c1, r1*N+c0, r1*N+c1 }); m_wscale.push_back(1.0f);
			}
		}
		for (int rp = 0; rp < static_cast<int>(m_ydim) / 2; ++rp) {
			int r0 = rp, r1 = static_cast<int>(m_ydim) - 1 - rp;
			m_pos.insert(m_pos.end(), { r0*N+C, r0*N+C, r1*N+C, r1*N+C }); m_wscale.push_back(0.5f);
		}
		m_R = static_cast<unsigned>(m_wscale.size());

		PredictorModel tmp = materialise_symmetric(model);
		subtract_mean(tmp);
		for (int q = 0; q < 2; ++q) {
			const float *SM = q ? tmp.second.softmax_q2 : tmp.second.softmax_q1;
			const float *EL = q ? tmp.second.elliott_q2 : tmp.second.elliott_q1;
			const float *SB = q ? tmp.second.softmax_bias_q2 : tmp.second.softmax_bias_q1;
			const float *EB = q ? tmp.second.elliott_bias_q2 : tmp.second.elliott_bias_q1;
			m_wf_s[q].assign(4 * m_ng * m_R, 0.0f); m_wf_e[q].assign(4 * m_ng * m_R, 0.0f);
			m_bias_s[q].assign(m_ng, 0.0f); m_bias_e[q].assign(m_ng, 0.0f);
			for (unsigned c = 0; c < 4; ++c)
				for (unsigned gi = 0; gi < m_ng; ++gi)
					for (unsigned o = 0; o < m_R; ++o) {
						float s = 0, e = 0;
						for (int k = 0; k < 4; ++k) { int p = m_pos[o*4 + k]; s += SYM_HC[c][k]*SM[gi*Dm1 + p]; e += SYM_HC[c][k]*EL[gi*Dm1 + p]; }
						m_wf_s[q][(c*m_ng + gi)*m_R + o] = m_wscale[o] * s;
						m_wf_e[q][(c*m_ng + gi)*m_R + o] = m_wscale[o] * e;
					}
			for (unsigned gi = 0; gi < m_ng; ++gi) { m_bias_s[q][gi] = SB[gi]; m_bias_e[q][gi] = EB[gi]; }
		}
	}

	size_t get_tmp_size() const noexcept override { return 0; }

	void process(const float * const src[6], float *dst, const unsigned char *prescreen, void *, unsigned n) const noexcept override
	{
		float buf[48 * 6], xh[4 * 72], a[256], e[256];
		unsigned wy = 3 - m_ydim / 2;
		double invD = 1.0 / (static_cast<double>(m_nm1) * m_ydim);

		for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
			if (prescreen[i])
				continue;

			double sum = 0, sq = 0;
			for (unsigned r = 0; r < m_ydim; ++r) {
				const float *row = src[wy + r] + (i - m_off_x);
				for (unsigned c = 0; c < m_nm1; ++c) { float v = row[c]; buf[r*m_nm1 + c] = v; sum += v; sq += static_cast<double>(v)*v; }
			}
			float mean = static_cast<float>(sum * invD);
			double var = sq * invD - static_cast<double>(mean) * mean;
			float sd = var < FLT_EPSILON ? 0.0f : static_cast<float>(std::sqrt(var));
			float scale = sd == 0.0f ? 0.0f : 1.0f / sd;

			for (unsigned o = 0; o < m_R; ++o)
				for (unsigned c = 0; c < 4; ++c) { float s = 0; for (int k = 0; k < 4; ++k) s += SYM_HC[c][k]*buf[m_pos[o*4 + k]]; xh[c*m_R + o] = s; }

			float out = 0.0f;
			for (int q = 0; q < (m_use_q2 ? 2 : 1); ++q) {
				const float *wfs = m_wf_s[q].data(), *wfe = m_wf_e[q].data(), *bs = m_bias_s[q].data(), *be = m_bias_e[q].data();
				for (unsigned gi = 0; gi < m_ng; ++gi) {
					float Ps[4], Pe[4];
					for (unsigned c = 0; c < 4; ++c) {
						const float *ws = wfs + (c*m_ng + gi)*m_R, *we = wfe + (c*m_ng + gi)*m_R, *x = xh + c*m_R;
						float ps = 0, pe = 0;
						for (unsigned o = 0; o < m_R; ++o) { ps += x[o]*ws[o]; pe += x[o]*we[o]; }
						Ps[c] = ps; Pe[c] = pe;
					}
					for (int f = 0; f < 4; ++f) {
						float ds = 0, de = 0;
						for (int c = 0; c < 4; ++c) { ds += SYM_HC[c][f]*Ps[c]; de += SYM_HC[c][f]*Pe[c]; }
						a[f*m_ng + gi] = 0.25f*ds*scale + bs[gi];
						e[f*m_ng + gi] = 0.25f*de*scale + be[gi];
					}
				}
				float vsum = 0, wsum = 0;
				for (unsigned k = 0; k < m_nns; ++k) { float sx = softmax_exp(a[k]); vsum += sx * elliott(e[k]); wsum += sx; }
				out += wsum > 1e-10f ? 5.0f*vsum/wsum*sd + mean : mean;
			}
			dst[i] = out / (m_use_q2 ? 2 : 1);
		}
	}
};

} // namespace


pixel_io_func select_pixel_io_func(PixelType in, PixelType out, CPUClass cpu)
{
	pixel_io_func ret = nullptr;

#ifdef ZNEDI3_X86
	ret = select_pixel_io_func_x86(in, out, cpu);
#endif

	if (!ret && in == PixelType::BYTE && out == PixelType::FLOAT)
		ret = integer_to_float<uint8_t>;
	if (!ret && in == PixelType::WORD && out == PixelType::FLOAT)
		ret = integer_to_float<uint16_t>;
	if (!ret && in == PixelType::FLOAT && out == PixelType::BYTE)
		ret = float_to_integer<uint8_t>;
	if (!ret && in == PixelType::FLOAT && out == PixelType::WORD)
		ret = float_to_integer<uint16_t>;
	if (!ret && in == PixelType::FLOAT && out == PixelType::FLOAT)
		ret = copy_float;

	return ret;
}

interpolate_func select_interpolate_func(CPUClass cpu)
{
	interpolate_func ret = nullptr;

#ifdef ZNEDI3_X86
	ret = select_interpolate_func_x86(cpu);
#endif

	if (!ret)
		ret = cubic_interpolation_c;

	assert(ret);
	return ret;
}

std::unique_ptr<Prescreener> create_prescreener_old(const PrescreenerOldCoefficients &coeffs, double pixel_half, CPUClass cpu)
{
	std::unique_ptr<Prescreener> ret;

#ifdef ZNEDI3_X86
	ret = create_prescreener_old_x86(coeffs, pixel_half, cpu);
#endif

	if (!ret)
		ret = std::make_unique<PrescreenerOldC>(coeffs, pixel_half);

	assert(ret);
	return ret;
}

std::unique_ptr<Prescreener> create_prescreener_new(const PrescreenerNewCoefficients &coeffs, double pixel_half, CPUClass cpu)
{
	std::unique_ptr<Prescreener> ret;

#ifdef ZNEDI3_X86
	ret = create_prescreener_new_x86(coeffs, pixel_half, cpu);
#endif

	if (!ret)
		ret = std::make_unique<PrescreenerNewC>(coeffs, pixel_half);

	assert(ret);
	return ret;
}

std::unique_ptr<Predictor> create_predictor(const std::pair<const PredictorTraits, PredictorCoefficients> &model, bool use_q2, CPUClass cpu)
{
	std::unique_ptr<Predictor> ret;

#ifdef ZNEDI3_X86
	ret = create_predictor_x86(model, use_q2, cpu);
#endif

	if (!ret)
		ret = std::make_unique<PredictorC>(model, use_q2);

	assert(ret);
	return ret;
}

std::unique_ptr<Predictor> create_symmetric_predictor(const SymmetricPredictorModel &model, bool use_q2, CPUClass cpu)
{
	// Performance gate: the folded predictor only beats the dense one at nns >= 128. Below that its
	// fixed per-pixel window-fold + combine overhead isn't amortised by the smaller matvec, and
	// nns < 64 additionally has no mask-free AVX-512 path (it falls to AVX2/scalar while the dense
	// predictor stays AVX-512), so it regresses -- measured single-thread ~0.16x (nns16), ~0.4x
	// (nns32), ~0.7x (nns64), only marginally rescued multi-threaded. Returning nullptr makes the
	// caller fall back to the dense predictor, which is byte-identical to stock at full speed. At
	// nns >= 128 the fold is a clean win (~1.1-2.3x). This gate is independent of the weight file.
	if (model.first.nns < 128)
		return nullptr;

	std::unique_ptr<Predictor> ret;

#ifdef ZNEDI3_X86
	ret = create_symmetric_predictor_x86(model, use_q2, cpu);   // folded AVX-512 for nns >= 64
#endif

	// Portable folded fallback (small nns, or no AVX2/AVX-512 SIMD path).
	if (!ret)
		ret = std::make_unique<PredictorSymmetricC>(model, use_q2);

	assert(ret);
	return ret;
}

} // namespace znedi3

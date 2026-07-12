#ifdef ZNEDI3_X86

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#include "alloc.h"
#include "cpuinfo.h"
#include "cpuinfo_x86.h"
#include "kernel.h"
#include "kernel_x86.h"
#include "znedi3_impl.h"

namespace znedi3 {
namespace {

pixel_io_func select_pixel_io_func_sse2(PixelType in, PixelType out)
{
	if (in == PixelType::BYTE && out == PixelType::FLOAT)
		return byte_to_float_sse2;
	else if (in == PixelType::WORD && out == PixelType::FLOAT)
		return word_to_float_sse2;
	else if (in == PixelType::FLOAT && out == PixelType::BYTE)
		return float_to_byte_sse2;
	else if (in == PixelType::FLOAT && out == PixelType::WORD)
		return float_to_word_sse2;
	else
		return nullptr;
}

pixel_io_func select_pixel_io_func_f16c(PixelType in, PixelType out)
{
	if (in == PixelType::HALF && out == PixelType::FLOAT)
		return half_to_float_f16c;
	else if (in == PixelType::FLOAT && out == PixelType::HALF)
		return float_to_half_f16c;
	else
		return nullptr;
}

pixel_io_func select_pixel_io_func_avx2(PixelType in, PixelType out)
{
	if (in == PixelType::BYTE && out == PixelType::FLOAT)
		return byte_to_float_avx2;
	else if (in == PixelType::WORD && out == PixelType::FLOAT)
		return word_to_float_avx2;
	else if (in == PixelType::FLOAT && out == PixelType::BYTE)
		return float_to_byte_avx2;
	else if (in == PixelType::FLOAT && out == PixelType::WORD)
		return float_to_word_avx2;
	else
		return nullptr;
}

#ifdef ZNEDI3_X86_AVX512
pixel_io_func select_pixel_io_func_avx512f(PixelType in, PixelType out)
{
	if (in == PixelType::BYTE && out == PixelType::FLOAT)
		return byte_to_float_avx512f;
	else if (in == PixelType::WORD && out == PixelType::FLOAT)
		return word_to_float_avx512f;
	else if (in == PixelType::HALF && out == PixelType::FLOAT)
		return half_to_float_avx512f;
	else if (in == PixelType::FLOAT && out == PixelType::BYTE)
		return float_to_byte_avx512f;
	else if (in == PixelType::FLOAT && out == PixelType::WORD)
		return float_to_word_avx512f;
	else if (in == PixelType::FLOAT && out == PixelType::HALF)
		return float_to_half_avx512f;
	else
		return nullptr;
}
#endif // ZNEDI3_X86_AVX512

} // namespace

InterleavedPredictorModel create_interleaved_predictor_model(const PredictorModel &model)
{
	assert(model.first.nns % 16 == 0);

	unsigned filter_size = model.first.xdim * model.first.ydim;
	unsigned nns = model.first.nns;

	PredictorModel m = copy_model(model);
	subtract_mean(m);

	InterleavedPredictorModel interleaved{};
	interleaved.data.resize(nns * filter_size * 4 + nns * 4);

	interleaved.xdim = m.first.xdim;
	interleaved.ydim = m.first.ydim;
	interleaved.nns = m.first.nns;

	LinearAllocator alloc{ interleaved.data.data() };
	interleaved.neurons_q1 = alloc.allocate_n<float>(nns * filter_size * 2);
	interleaved.bias_q1 = alloc.allocate_n<float>(nns * 2);
	interleaved.neurons_q2 = alloc.allocate_n<float>(nns * filter_size * 2);
	interleaved.bias_q2 = alloc.allocate_n<float>(nns * 2);
	assert(alloc.count() / sizeof(float) == interleaved.data.size());

	for (unsigned k = 0; k < filter_size; ++k) {
		for (unsigned nn = 0; nn < nns; ++nn) {
			interleaved.neurons_q1[k * nns * 2 + nn] = m.second.softmax_q1[nn * filter_size + k];
			interleaved.neurons_q1[k * nns * 2 + nn + nns] = m.second.elliott_q1[nn * filter_size + k];
		}
		for (unsigned nn = 0; nn < nns; ++nn) {
			interleaved.neurons_q2[k * nns * 2 + nn] = m.second.softmax_q2[nn * filter_size + k];
			interleaved.neurons_q2[k * nns * 2 + nn + nns] = m.second.elliott_q2[nn * filter_size + k];
		}
	}
	std::copy_n(m.second.softmax_bias_q1, nns, interleaved.bias_q1);
	std::copy_n(m.second.elliott_bias_q1, nns, interleaved.bias_q1 + nns);
	std::copy_n(m.second.softmax_bias_q2, nns, interleaved.bias_q2);
	std::copy_n(m.second.elliott_bias_q2, nns, interleaved.bias_q2 + nns);

	return interleaved;
}

SymmetricInterleavedModel create_symmetric_interleaved_model(const SymmetricPredictorModel &model)
{
	assert(model.first.nns % 4 == 0);
	unsigned xdim = model.first.xdim, ydim = model.first.ydim, nns = model.first.nns;
	unsigned nm1 = xdim - 1, ng = nns / 4, Dm1 = nm1 * ydim;

	// Klein-4 orbits about the target column C = (nm1-1)/2 and the row gap. The h-fixed centre
	// column forms orbits-of-2 (pos[0]==pos[1], pos[2]==pos[3]) with a 0.5 weight correction.
	struct Orbit { int pos[4]; float wscale; };
	std::vector<Orbit> orbits;
	int C = static_cast<int>(nm1 - 1) / 2, N = static_cast<int>(nm1);
	for (int cp = 0; cp < C; ++cp) {
		int c0 = cp, c1 = N - 1 - cp;
		for (int rp = 0; rp < static_cast<int>(ydim) / 2; ++rp) {
			int r0 = rp, r1 = static_cast<int>(ydim) - 1 - rp;
			orbits.push_back({ { r0*N+c0, r0*N+c1, r1*N+c0, r1*N+c1 }, 1.0f });
		}
	}
	for (int rp = 0; rp < static_cast<int>(ydim) / 2; ++rp) {
		int r0 = rp, r1 = static_cast<int>(ydim) - 1 - rp;
		orbits.push_back({ { r0*N+C, r0*N+C, r1*N+C, r1*N+C }, 0.5f });
	}
	unsigned R = static_cast<unsigned>(orbits.size());
	unsigned Rp = (R + 15u) & ~15u;

	// Materialise the nns experts of both qualities, mean-fold, then experts[0..ng-1] of each bank
	// are the mean-folded generators (subtract_mean preserves the flip symmetry).
	PredictorModel tmp = allocate_model({ nm1, ydim, nns });
	auto flip_into = [&](const float *gen, float *dst, int f) {
		for (unsigned r = 0; r < ydim; ++r)
			for (unsigned c = 0; c < nm1; ++c) {
				unsigned rr = (f & 2) ? ydim - 1 - r : r, cc = (f & 1) ? nm1 - 1 - c : c;
				dst[r*nm1 + c] = gen[rr*nm1 + cc];
			}
	};
	auto materialise = [&](const float *sg, const float *eg, const float *sb, const float *eb,
	                       float *SM, float *EL, float *SMB, float *ELB) {
		for (int f = 0; f < 4; ++f)
			for (unsigned gi = 0; gi < ng; ++gi) {
				unsigned i = f * ng + gi;
				flip_into(sg + gi*Dm1, SM + i*Dm1, f);
				flip_into(eg + gi*Dm1, EL + i*Dm1, f);
				SMB[i] = sb[gi]; ELB[i] = eb[gi];
			}
	};
	materialise(model.second.softmax_gen_q1, model.second.elliott_gen_q1, model.second.softmax_bias_q1, model.second.elliott_bias_q1,
	            tmp.second.softmax_q1, tmp.second.elliott_q1, tmp.second.softmax_bias_q1, tmp.second.elliott_bias_q1);
	materialise(model.second.softmax_gen_q2, model.second.elliott_gen_q2, model.second.softmax_bias_q2, model.second.elliott_bias_q2,
	            tmp.second.softmax_q2, tmp.second.elliott_q2, tmp.second.softmax_bias_q2, tmp.second.elliott_bias_q2);
	subtract_mean(tmp);

	SymmetricInterleavedModel out{};
	out.xdim = xdim; out.ydim = ydim; out.nns = nns; out.ng = ng; out.R = R; out.Rp = Rp;
	out.window_offset_x = xdim / 2 - 1;

	size_t wsz = static_cast<size_t>(4) * Rp * 2 * ng;
	out.data.resize(2 * wsz + 2 * (2 * ng));
	out.gather.resize(static_cast<size_t>(4) * Rp, 0);

	LinearAllocator alloc{ out.data.data() };
	out.wfold_q1 = alloc.allocate_n<float>(wsz);
	out.wfold_q2 = alloc.allocate_n<float>(wsz);
	out.bias_q1 = alloc.allocate_n<float>(2 * ng);
	out.bias_q2 = alloc.allocate_n<float>(2 * ng);

	static const float HC[4][4] = { {1,1,1,1},{1,-1,1,-1},{1,1,-1,-1},{1,-1,-1,1} };
	auto fold_bank = [&](const float *SM, const float *EL, const float *SMB, const float *ELB, float *wfold, float *bias) {
		std::fill_n(wfold, wsz, 0.0f);
		for (int c = 0; c < 4; ++c) {
			float *Wc = wfold + static_cast<size_t>(c) * Rp * 2 * ng;
			for (unsigned o = 0; o < R; ++o)
				for (unsigned gi = 0; gi < ng; ++gi) {
					float s = 0, e = 0;
					for (int k = 0; k < 4; ++k) {
						s += HC[c][k] * SM[gi*Dm1 + orbits[o].pos[k]];
						e += HC[c][k] * EL[gi*Dm1 + orbits[o].pos[k]];
					}
					Wc[static_cast<size_t>(o)*2*ng + gi]      = orbits[o].wscale * s;
					Wc[static_cast<size_t>(o)*2*ng + ng + gi] = orbits[o].wscale * e;
				}
		}
		std::copy_n(SMB, ng, bias);          // softmax generator biases (mean-folded)
		std::copy_n(ELB, ng, bias + ng);     // elliott generator biases (unchanged by mean-fold)
	};
	fold_bank(tmp.second.softmax_q1, tmp.second.elliott_q1, tmp.second.softmax_bias_q1, tmp.second.elliott_bias_q1, out.wfold_q1, out.bias_q1);
	fold_bank(tmp.second.softmax_q2, tmp.second.elliott_q2, tmp.second.softmax_bias_q2, tmp.second.elliott_bias_q2, out.wfold_q2, out.bias_q2);

	for (int k = 0; k < 4; ++k)
		for (unsigned o = 0; o < R; ++o)
			out.gather[static_cast<size_t>(k)*Rp + o] = orbits[o].pos[k];

	return out;
}

std::unique_ptr<Predictor> create_symmetric_predictor_x86(const SymmetricPredictorModel &model, bool use_q2, CPUClass cpu, bool use_int16)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Predictor> ret;

	// The folded AVX-512 path handles nns >= 64 (ng a multiple of 16, so no mask registers);
	// smaller nns fall through to the portable C folded predictor.
	if (cpu_is_autodetect(cpu)) {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && use_int16 && cpu == CPUClass::AUTO_64B && caps.avx512f && caps.avx512vnni && model.first.nns >= 64)
			ret = create_symmetric_predictor_int16_avx512vnni(model, use_q2);
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f && model.first.nns >= 64)
			ret = create_symmetric_predictor_avx512f(model, use_q2);
#endif
		if (!ret && caps.avx2 && caps.fma && model.first.nns >= 32)
			ret = create_symmetric_predictor_avx2(model, use_q2);
	} else {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512 && model.first.nns >= 64)
			ret = create_symmetric_predictor_avx512f(model, use_q2);
#endif
		if (!ret && cpu >= CPUClass::X86_AVX2 && model.first.nns >= 32)
			ret = create_symmetric_predictor_avx2(model, use_q2);
	}
	return ret;
}

pixel_io_func select_pixel_io_func_x86(PixelType in, PixelType out, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	pixel_io_func ret = nullptr;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f)
			ret = select_pixel_io_func_avx512f(in, out);
#endif
		if (!ret && caps.avx2)
			ret = select_pixel_io_func_avx2(in, out);
		if (!ret && caps.avx && caps.f16c)
			ret = select_pixel_io_func_f16c(in, out);
		if (!ret && caps.sse2)
			ret = select_pixel_io_func_sse2(in, out);
	} else {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = select_pixel_io_func_avx512f(in, out);
#endif
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = select_pixel_io_func_avx2(in, out);
		if (!ret && cpu >= CPUClass::X86_F16C)
			ret = select_pixel_io_func_f16c(in, out);
		if (!ret && cpu >= CPUClass::X86_SSE2)
			ret = select_pixel_io_func_sse2(in, out);
	}

	return ret;
}

interpolate_func select_interpolate_func_x86(CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	interpolate_func ret = nullptr;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f)
			ret = cubic_interpolation_avx512f;
#endif
		if (!ret && caps.avx2 && caps.fma)
			ret = cubic_interpolation_avx2;
		if (!ret && caps.avx)
			ret = cubic_interpolation_avx;
		if (!ret && caps.sse2)
			ret = cubic_interpolation_sse2;
	} else {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = cubic_interpolation_avx512f;
#endif
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = cubic_interpolation_avx2;
		if (!ret && cpu >= CPUClass::X86_AVX)
			ret = cubic_interpolation_avx;
		if (!ret && cpu >= CPUClass::X86_SSE2)
			ret = cubic_interpolation_sse2;
	}

	return ret;
}

std::unique_ptr<Prescreener> create_prescreener_old_x86(const PrescreenerOldCoefficients &coeffs, double pixel_half, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Prescreener> ret;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f)
			ret = create_prescreener_old_avx512f(coeffs, pixel_half);
#endif
		if (!ret && caps.avx2 && caps.fma)
			ret = create_prescreener_old_avx2(coeffs, pixel_half);
		if (!ret && caps.avx)
			ret = create_prescreener_old_avx(coeffs, pixel_half);
		if (!ret && caps.sse)
			ret = create_prescreener_old_sse(coeffs, pixel_half);
	} else {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = create_prescreener_old_avx512f(coeffs, pixel_half);
#endif
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_prescreener_old_avx2(coeffs, pixel_half);
		if (!ret && cpu >= CPUClass::X86_AVX)
			ret = create_prescreener_old_avx(coeffs, pixel_half);
		if (!ret && cpu >= CPUClass::X86_SSE)
			ret = create_prescreener_old_sse(coeffs, pixel_half);
	}

	return ret;
}

std::unique_ptr<Prescreener> create_prescreener_new_x86(const PrescreenerNewCoefficients &coeffs, double pixel_half, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Prescreener> ret;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f)
			ret = create_prescreener_new_avx512f(coeffs, pixel_half);
#endif
		if (!ret && caps.avx2 && caps.fma)
			ret = create_prescreener_new_avx2(coeffs, pixel_half);
		if (!ret && caps.avx)
			ret = create_prescreener_new_avx(coeffs, pixel_half);
		if (!ret && caps.sse)
			ret = create_prescreener_new_sse(coeffs, pixel_half);
	} else {
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = create_prescreener_new_avx512f(coeffs, pixel_half);
#endif
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_prescreener_new_avx2(coeffs, pixel_half);
		if (!ret && cpu >= CPUClass::X86_AVX)
			ret = create_prescreener_new_avx(coeffs, pixel_half);
		if (!ret && cpu >= CPUClass::X86_SSE)
			ret = create_prescreener_new_sse(coeffs, pixel_half);
	}

	return ret;
}

std::unique_ptr<Predictor> create_predictor_x86(const PredictorModel &model, bool use_q2, CPUClass cpu, bool use_int16)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Predictor> ret;

	if (cpu_is_autodetect(cpu)) {
#ifdef ZNEDI3_X86_AVX512
		// int16-VNNI mat-vec (~2x fp32, precision-safe for high bit depth) when requested + available.
		if (!ret && use_int16 && cpu == CPUClass::AUTO_64B && caps.avx512f && caps.avx512vnni)
			ret = create_predictor_int16_avx512vnni(model, use_q2);
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f)
			ret = create_predictor_avx512f(model, use_q2);
#endif
		if (!ret && caps.avx && caps.fma)
			ret = create_predictor_avx2(model, use_q2);
		if (!ret && caps.avx)
			ret = create_predictor_avx(model, use_q2);
		if (!ret && caps.sse2)
			ret = create_predictor_sse2(model, use_q2);
		if (!ret && caps.sse)
			ret = create_predictor_sse(model, use_q2);
	} else {
		// Explicit cpu override always uses the pure fp32 kernel for that ISA (no int16-VNNI).
#ifdef ZNEDI3_X86_AVX512
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = create_predictor_avx512f(model, use_q2);
#endif
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_predictor_avx2(model, use_q2);
		if (!ret && cpu >= CPUClass::X86_AVX)
			ret = create_predictor_avx(model, use_q2);
		if (!ret && cpu >= CPUClass::X86_SSE2)
			ret = create_predictor_sse2(model, use_q2);
		if (!ret && cpu >= CPUClass::X86_SSE)
			ret = create_predictor_sse(model, use_q2);
	}

	return ret;
}

} // namespace znedi3

#endif // ZNEDI3_X86

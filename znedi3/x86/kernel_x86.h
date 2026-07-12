#pragma once

#ifdef ZNEDI3_X86

#ifndef X86_KERNEL_X86_H_
#define X86_KERNEL_X86_H_

#include <memory>
#include <utility>
#include "alloc.h"
#include "kernel.h"
#include "weights.h"

namespace znedi3 {

enum class CPUClass;
enum class PixelType;

// Polynomial coefficients for exp2f(x - 1) on the domain [1.0, 2.0].
// Coefficients are stored low-order to high-order.
constexpr float EXP2F_X_PLUS1_REMEZ[] = {
	0.509871020343597804469416f,
	0.312146713032169896138863f,
	0.166617139319965966118107f,
	-2.19061993049215080032874e-3f,
	1.3555747234758484073940937e-2f
};

// Coefficients such that converting (EXPF_LN2_INV_SCALED * x + EXPF_ONE_SCALED)
// to integer and reinterpreting the result as a float produces the integer
// component of (x / ln(2)) in the exponent and the fractional component in the
// mantissa.
constexpr float EXPF_LN2_INV_SCALED = 12102203.1615614f; // (1.0 / (127 * ln(2))) * EXPF_ONE_SCALED.
constexpr float EXPF_ONE_SCALED = 1065353216.f; // Integer representation of 1.0f.


struct InterleavedPredictorModel {
	AlignedVector<float> data;
	unsigned xdim;
	unsigned ydim;
	unsigned nns;

	// Filter coefficients are stored interleaved, such that all the
	// coefficients for the n-th softmax neuron are stored contiguously,
	// followed by all the coefficients for the n-th elliott neuron.
	//
	// f[nn=0][k=0] f[nn=1][k=0] f[nn=2][k=0] ... f[nn=nns*2][k=0]
	// f[nn=0][k=1] f[nn=1][k=1] f[nn=2][k=1] ... f[nn=nns*2][k=1]
	// ...
	//
	// Likewise, the softmax and elliott biases are stored contiguously.
	float *neurons_q1;
	float *bias_q1;
	float *neurons_q2;
	float *bias_q2;
};

InterleavedPredictorModel create_interleaved_predictor_model(const PredictorModel &model);


// h+v-symmetric folded predictor model. The nns/4 flip-orbit generators (at width xdim-1) are
// mean-folded and Klein-4-transformed into 4 symmetry channels, each interleaved [Rp][2*ng] like
// the dense model. The window is gathered at width xdim-1 and folded with the same Klein-4
// transform (gather indices below); the folded mat-vec produces 4 channels per generator which a
// cheap butterfly recombines into the nns experts. R = (xdim/2)*(ydim/2) orbits, padded to Rp
// (multiple of 16). Mask-register-free by construction (everything padded to 16).
struct SymmetricInterleavedModel {
	AlignedVector<float> data;        // wfold_q1 | wfold_q2 | bias_q1 | bias_q2
	AlignedVector<int32_t> gather;    // [4 * Rp] window-fold gather indices into the [ydim*(xdim-1)] window buffer

	float *wfold_q1;   // [4 * Rp * 2*ng]; channel c at wfold_q1 + c*Rp*2*ng, laid out [Rp][2*ng]
	float *wfold_q2;
	float *bias_q1;    // [2*ng]: softmax generator biases (0..ng-1) then elliott generator biases (ng..2ng-1)
	float *bias_q2;

	unsigned xdim, ydim, nns;   // full xdim; effective window width is xdim-1
	unsigned ng, R, Rp;         // generators (nns/4), orbits, padded orbits
	unsigned window_offset_x;   // xdim/2 - 1
};

SymmetricInterleavedModel create_symmetric_interleaved_model(const SymmetricPredictorModel &model);

std::unique_ptr<Predictor> create_symmetric_predictor_avx2(const SymmetricPredictorModel &model, bool use_q2);
std::unique_ptr<Predictor> create_symmetric_predictor_avx512f(const SymmetricPredictorModel &model, bool use_q2);
std::unique_ptr<Predictor> create_symmetric_predictor_int16_avx512vnni(const SymmetricPredictorModel &model, bool use_q2);
std::unique_ptr<Predictor> create_symmetric_predictor_x86(const SymmetricPredictorModel &model, bool use_q2, CPUClass cpu, bool use_int16);


// SSE
std::unique_ptr<Prescreener> create_prescreener_old_sse(const PrescreenerOldCoefficients &coeffs, double pixel_half);
std::unique_ptr<Prescreener> create_prescreener_new_sse(const PrescreenerNewCoefficients &coeffs, double pixel_half);
std::unique_ptr<Predictor> create_predictor_sse(const PredictorModel &model, bool use_q2);

// SSE2
void byte_to_float_sse2(const void *src, void *dst, size_t n);
void word_to_float_sse2(const void *src, void *dst, size_t n);
void float_to_byte_sse2(const void *src, void *dst, size_t n);
void float_to_word_sse2(const void *src, void *dst, size_t n);

void cubic_interpolation_sse2(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n);

std::unique_ptr<Predictor> create_predictor_sse2(const PredictorModel &model, bool use_q2);

// AVX
void cubic_interpolation_avx(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n);

std::unique_ptr<Prescreener> create_prescreener_old_avx(const PrescreenerOldCoefficients &coeffs, double pixel_half);
std::unique_ptr<Prescreener> create_prescreener_new_avx(const PrescreenerNewCoefficients &coeffs, double pixel_half);
std::unique_ptr<Predictor> create_predictor_avx(const PredictorModel &model, bool use_q2);

// F16C
void half_to_float_f16c(const void *src, void *dst, size_t n);
void float_to_half_f16c(const void *src, void *dst, size_t n);

// AVX2
void byte_to_float_avx2(const void *src, void *dst, size_t n);
void word_to_float_avx2(const void *src, void *dst, size_t n);
void float_to_byte_avx2(const void *src, void *dst, size_t n);
void float_to_word_avx2(const void *src, void *dst, size_t n);

void cubic_interpolation_avx2(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n);

std::unique_ptr<Prescreener> create_prescreener_old_avx2(const PrescreenerOldCoefficients &coeffs, double pixel_half);
std::unique_ptr<Prescreener> create_prescreener_new_avx2(const PrescreenerNewCoefficients &coeffs, double pixel_half);
std::unique_ptr<Predictor> create_predictor_avx2(const PredictorModel &model, bool use_q2);

// AVX-512F
void byte_to_float_avx512f(const void *src, void *dst, size_t n);
void word_to_float_avx512f(const void *src, void *dst, size_t n);
void half_to_float_avx512f(const void *src, void *dst, size_t n);
void float_to_byte_avx512f(const void *src, void *dst, size_t n);
void float_to_word_avx512f(const void *src, void *dst, size_t n);
void float_to_half_avx512f(const void *src, void *dst, size_t n);

void cubic_interpolation_avx512f(const float * const src[4], float *dst, const unsigned char *prescreen, unsigned n);

std::unique_ptr<Prescreener> create_prescreener_old_avx512f(const PrescreenerOldCoefficients &coeffs, double pixel_half);
std::unique_ptr<Prescreener> create_prescreener_new_avx512f(const PrescreenerNewCoefficients &coeffs, double pixel_half);
std::unique_ptr<Predictor> create_predictor_avx512f(const PredictorModel &model, bool use_q2);
std::unique_ptr<Predictor> create_predictor_int16_avx512vnni(const PredictorModel &model, bool use_q2);


pixel_io_func select_pixel_io_func_x86(PixelType in, PixelType out, CPUClass cpu);
interpolate_func select_interpolate_func_x86(CPUClass cpu);

std::unique_ptr<Prescreener> create_prescreener_old_x86(const PrescreenerOldCoefficients &coeffs, double pixel_half, CPUClass cpu);
std::unique_ptr<Prescreener> create_prescreener_new_x86(const PrescreenerNewCoefficients &coeffs, double pixel_half, CPUClass cpu);
std::unique_ptr<Predictor> create_predictor_x86(const PredictorModel &model, bool use_q2, CPUClass cpu, bool use_int16);

} // namespace znedi3

#endif // X86_KERNEL_X86_H_

#endif // ZNEDI3_X86

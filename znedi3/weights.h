#pragma once

#ifndef ZNEDI3_WEIGHTS_H_
#define ZNEDI3_WEIGHTS_H_

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>

struct znedi3_weights {
protected:
	~znedi3_weights() = default;
};

namespace znedi3 {

constexpr size_t NNEDI3_WEIGHTS_SIZE = 13574928;
constexpr unsigned NNEDI3_XDIM[] = { 8, 16, 32, 48, 8, 16, 32 };
constexpr unsigned NNEDI3_YDIM[] = { 6, 6, 6, 6, 4, 4, 4 };
constexpr unsigned NNEDI3_NNS[] = { 16, 32, 64, 128, 256 };

constexpr unsigned NNEDI3_DIMS0 = 49 * 4 + 5 * 4 + 9 * 4;
constexpr unsigned NNEDI3_DIMS0_NEW = 4 * 65 + 4 * 5;


struct PrescreenerOldCoefficients {
	float kernel_l0[4][12 * 4];
	float bias_l0[4];

	float kernel_l1[4][4];
	float bias_l1[4];

	float kernel_l2[4][8];
	float bias_l2[4];
};

struct PrescreenerNewCoefficients {
	float kernel_l0[4][16 * 4];
	float bias_l0[4];

	float kernel_l1[4][4];
	float bias_l1[4];
};

struct PredictorTraits {
	unsigned xdim;
	unsigned ydim;
	unsigned nns;
};

struct PredictorCoefficients {
	std::unique_ptr<float[]> data;
	float *softmax_q1;
	float *elliott_q1;
	float *softmax_bias_q1;
	float *elliott_bias_q1;
	float *softmax_q2;
	float *elliott_q2;
	float *softmax_bias_q2;
	float *elliott_bias_q2;
};

// h+v-symmetric predictor: only the nns/4 flip-orbit generators, at the target-centred odd
// width N-1 (filter_size = (xdim-1)*ydim). The full nns experts are the Klein-4 flips of each
// generator; biases are shared within an orbit. Loaded from nnedi3_weights_hvsym.bin.
struct SymmetricPredictorCoefficients {
	std::unique_ptr<float[]> data;
	float *softmax_gen_q1;    // [nns/4 * (xdim-1)*ydim]
	float *elliott_gen_q1;
	float *softmax_bias_q1;   // [nns/4]
	float *elliott_bias_q1;
	float *softmax_gen_q2;
	float *elliott_gen_q2;
	float *softmax_bias_q2;
	float *elliott_bias_q2;
};


// Helper functions for std::unordered_map.
struct PredictorTraitsHash {
	size_t operator()(const PredictorTraits &traits) const noexcept;
};

constexpr bool operator==(const PredictorTraits &lhs, const PredictorTraits &rhs) noexcept;
constexpr bool operator!=(const PredictorTraits &lhs, const PredictorTraits &rhs) noexcept;


typedef std::pair<const PredictorTraits, PredictorCoefficients> PredictorModel;
typedef std::unordered_map<PredictorTraits, PredictorCoefficients, PredictorTraitsHash> PredictorModelSet;

// Symmetric models are keyed by the SAME (full) traits as the normal models, so the filter can
// look one up with the user's nsize/nns and fall back to the normal model if absent.
typedef std::pair<const PredictorTraits, SymmetricPredictorCoefficients> SymmetricPredictorModel;
typedef std::unordered_map<PredictorTraits, SymmetricPredictorCoefficients, PredictorTraitsHash> SymmetricPredictorModelSet;

PredictorModel allocate_model(const PredictorTraits &traits);
PredictorModel copy_model(const PredictorModel &model);

void subtract_mean(PrescreenerOldCoefficients &coeffs, double pixel_half);
void subtract_mean(PrescreenerNewCoefficients &coeffs, double pixel_half);
void subtract_mean(PredictorModel &model);


class NNEDI3Weights : public ::znedi3_weights {
	PrescreenerOldCoefficients m_prescreener_old;
	PrescreenerNewCoefficients m_prescreener_new[3];
	PredictorModelSet m_predictors_abs;
	PredictorModelSet m_predictors_mse;
public:
	PrescreenerOldCoefficients &prescreener_old() { return m_prescreener_old; }
	const PrescreenerOldCoefficients &prescreener_old() const { return m_prescreener_old; }

	PrescreenerNewCoefficients &prescreener_new(size_t i) { return m_prescreener_new[i]; }
	const PrescreenerNewCoefficients &prescreener_new(size_t i) const { return m_prescreener_new[i]; }

	PredictorModelSet &abs_models() { return m_predictors_abs; }
	const PredictorModelSet &abs_models() const { return m_predictors_abs; }

	PredictorModelSet &mse_models() { return m_predictors_mse; }
	const PredictorModelSet &mse_models() const { return m_predictors_mse; }

	SymmetricPredictorModelSet &sym_abs_models() { return m_predictors_sym_abs; }
	const SymmetricPredictorModelSet &sym_abs_models() const { return m_predictors_sym_abs; }

	SymmetricPredictorModelSet &sym_mse_models() { return m_predictors_sym_mse; }
	const SymmetricPredictorModelSet &sym_mse_models() const { return m_predictors_sym_mse; }

	bool has_symmetric() const { return !m_predictors_sym_abs.empty(); }

private:
	SymmetricPredictorModelSet m_predictors_sym_abs;
	SymmetricPredictorModelSet m_predictors_sym_mse;
};

std::unique_ptr<NNEDI3Weights> read_nnedi3_weights(const float *data);

// Load the h+v-symmetric generator weights from an in-memory nnedi3_weights_hvsym.bin (magic
// "HVSYMB1\0" + float generators) into an existing NNEDI3Weights (which keeps its prescreener and
// normal predictors from the original file). Returns false on bad magic / short data.
bool read_nnedi3_weights_hvsym(NNEDI3Weights &into, const void *data, size_t size);


inline size_t PredictorTraitsHash::operator()(const PredictorTraits &traits) const noexcept
{
#ifdef _MSC_VER
	return std::_Hash_representation(traits);
#else
	std::hash<unsigned> h;
	return h(traits.xdim) * h(traits.ydim) * h(traits.nns);
#endif
}

constexpr bool operator==(const PredictorTraits &lhs, const PredictorTraits &rhs) noexcept
{
	return lhs.xdim == rhs.xdim && lhs.ydim == rhs.ydim && lhs.nns == rhs.nns;
}

constexpr bool operator!=(const PredictorTraits &lhs, const PredictorTraits &rhs) noexcept
{
	return !(lhs == rhs);
}

} // namespace znedi3

#endif // ZNEDI3_WEIGHTS_H_

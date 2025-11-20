package teleprompt

import (
	"math"
	"math/rand"
)

// SIMBA utility functions for bandit algorithms and importance sampling.
// These utilities are used by the SIMBA optimizer and other optimizers.
// Based on dspy/teleprompt/simba_utils.py

// UCBSelection selects an arm using Upper Confidence Bound algorithm.
// armCounts: number of times each arm has been pulled
// armMeans: current mean reward for each arm
// totalCount: total number of pulls across all arms
// explorationParam: exploration parameter (typically 2.0)
// Returns: index of the selected arm
func UCBSelection(armCounts []int, armMeans []float64, totalCount int, explorationParam float64) int {
	if len(armCounts) != len(armMeans) {
		panic("armCounts and armMeans must have the same length")
	}

	if len(armCounts) == 0 {
		panic("armCounts cannot be empty")
	}

	bestArm := 0
	bestValue := -math.MaxFloat64

	for i := range armCounts {
		// Always try unexplored arms first
		if armCounts[i] == 0 {
			return i
		}

		// UCB1 formula: mean + exploration * sqrt(ln(total) / count)
		exploration := explorationParam * math.Sqrt(math.Log(float64(totalCount))/float64(armCounts[i]))
		value := armMeans[i] + exploration

		if value > bestValue {
			bestValue = value
			bestArm = i
		}
	}

	return bestArm
}

// ThompsonSampling selects an arm using Thompson Sampling.
// armSuccesses: number of successes for each arm (Beta distribution alpha parameter)
// armFailures: number of failures for each arm (Beta distribution beta parameter)
// Returns: index of the selected arm
func ThompsonSampling(armSuccesses, armFailures []int) int {
	if len(armSuccesses) != len(armFailures) {
		panic("armSuccesses and armFailures must have the same length")
	}

	if len(armSuccesses) == 0 {
		panic("armSuccesses cannot be empty")
	}

	bestArm := 0
	bestSample := -math.MaxFloat64

	for i := range armSuccesses {
		// Sample from Beta distribution Beta(successes+1, failures+1)
		alpha := float64(armSuccesses[i] + 1)
		beta := float64(armFailures[i] + 1)

		// Use Beta distribution sampling
		// For simplicity, use a normal approximation or placeholder
		// In production, would use proper Beta distribution sampling
		sample := sampleBeta(alpha, beta)

		if sample > bestSample {
			bestSample = sample
			bestArm = i
		}
	}

	return bestArm
}

// sampleBeta samples from a Beta distribution using mean approximation.
// This is a simplified version. Production code should use proper Beta sampling.
func sampleBeta(alpha, beta float64) float64 {
	// Mean of Beta(alpha, beta) is alpha / (alpha + beta)
	mean := alpha / (alpha + beta)

	// Add small random variation
	variance := (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1))
	std := math.Sqrt(variance)

	// Sample from normal distribution around mean
	sample := mean + rand.NormFloat64()*std

	// Clip to [0, 1]
	if sample < 0 {
		sample = 0
	}
	if sample > 1 {
		sample = 1
	}

	return sample
}

// EpsilonGreedy selects an arm using epsilon-greedy strategy.
// armMeans: current mean reward for each arm
// epsilon: probability of exploration (0 = always exploit, 1 = always explore)
// Returns: index of the selected arm
func EpsilonGreedy(armMeans []float64, epsilon float64) int {
	if len(armMeans) == 0 {
		panic("armMeans cannot be empty")
	}

	// Explore with probability epsilon
	if rand.Float64() < epsilon {
		// Random exploration
		return rand.Intn(len(armMeans))
	}

	// Exploit: choose best arm
	bestArm := 0
	bestMean := armMeans[0]

	for i := 1; i < len(armMeans); i++ {
		if armMeans[i] > bestMean {
			bestMean = armMeans[i]
			bestArm = i
		}
	}

	return bestArm
}

// ImportanceWeight computes the importance sampling weight.
// probability: true probability of the sample
// samplingProbability: probability under the sampling distribution
// Returns: importance weight
func ImportanceWeight(probability, samplingProbability float64) float64 {
	if samplingProbability <= 0 {
		panic("samplingProbability must be positive")
	}

	return probability / samplingProbability
}

// NormalizeRewards normalizes rewards to have mean 0 and std 1.
// rewards: array of reward values
// Returns: normalized rewards
func NormalizeRewards(rewards []float64) []float64 {
	if len(rewards) == 0 {
		return []float64{}
	}

	if len(rewards) == 1 {
		return []float64{0.0}
	}

	// Compute mean
	sum := 0.0
	for _, r := range rewards {
		sum += r
	}
	mean := sum / float64(len(rewards))

	// Compute standard deviation
	sumSq := 0.0
	for _, r := range rewards {
		diff := r - mean
		sumSq += diff * diff
	}
	variance := sumSq / float64(len(rewards))
	std := math.Sqrt(variance)

	// Handle case where all rewards are the same
	if std < 1e-8 {
		normalized := make([]float64, len(rewards))
		for i := range normalized {
			normalized[i] = 0.0
		}
		return normalized
	}

	// Normalize
	normalized := make([]float64, len(rewards))
	for i, r := range rewards {
		normalized[i] = (r - mean) / std
	}

	return normalized
}

// SoftmaxSelection selects an arm using softmax (Boltzmann) exploration.
// armMeans: current mean reward for each arm
// temperature: temperature parameter (higher = more exploration)
// Returns: index of the selected arm
func SoftmaxSelection(armMeans []float64, temperature float64) int {
	if len(armMeans) == 0 {
		panic("armMeans cannot be empty")
	}

	if temperature <= 0 {
		panic("temperature must be positive")
	}

	// Compute softmax probabilities
	expValues := make([]float64, len(armMeans))
	maxMean := armMeans[0]

	// Find max for numerical stability
	for _, mean := range armMeans {
		if mean > maxMean {
			maxMean = mean
		}
	}

	// Compute exp values and sum
	sumExp := 0.0
	for i, mean := range armMeans {
		expValues[i] = math.Exp((mean - maxMean) / temperature)
		sumExp += expValues[i]
	}

	// Normalize to get probabilities
	probabilities := make([]float64, len(armMeans))
	for i := range expValues {
		probabilities[i] = expValues[i] / sumExp
	}

	// Sample from categorical distribution
	return sampleCategorical(probabilities)
}

// sampleCategorical samples from a categorical distribution.
func sampleCategorical(probabilities []float64) int {
	r := rand.Float64()
	cumulative := 0.0

	for i, p := range probabilities {
		cumulative += p
		if r <= cumulative {
			return i
		}
	}

	// Fallback to last index due to floating point errors
	return len(probabilities) - 1
}

// ComputeGAE computes Generalized Advantage Estimation.
// rewards: rewards at each timestep
// values: value function estimates at each timestep
// gamma: discount factor
// lambda: GAE lambda parameter
// Returns: advantages at each timestep
func ComputeGAE(rewards, values []float64, gamma, lambda float64) []float64 {
	if len(rewards) != len(values) {
		panic("rewards and values must have the same length")
	}

	if len(rewards) == 0 {
		return []float64{}
	}

	advantages := make([]float64, len(rewards))
	lastGAE := 0.0

	// Compute GAE backwards
	for t := len(rewards) - 1; t >= 0; t-- {
		var nextValue float64
		if t == len(rewards)-1 {
			nextValue = 0.0 // Terminal state
		} else {
			nextValue = values[t+1]
		}

		// TD error: delta = r + gamma * V(s') - V(s)
		delta := rewards[t] + gamma*nextValue - values[t]

		// GAE: A_t = delta + gamma * lambda * A_{t+1}
		lastGAE = delta + gamma*lambda*lastGAE
		advantages[t] = lastGAE
	}

	return advantages
}

// ClipGradients clips gradients to a maximum norm.
// gradients: gradient values
// maxNorm: maximum allowed norm
// Returns: clipped gradients
func ClipGradients(gradients []float64, maxNorm float64) []float64 {
	if len(gradients) == 0 {
		return []float64{}
	}

	// Compute norm
	sumSq := 0.0
	for _, g := range gradients {
		sumSq += g * g
	}
	norm := math.Sqrt(sumSq)

	// If norm is within limit, return as is
	if norm <= maxNorm {
		return gradients
	}

	// Scale down to maxNorm
	scale := maxNorm / norm
	clipped := make([]float64, len(gradients))
	for i, g := range gradients {
		clipped[i] = g * scale
	}

	return clipped
}

// ComputeEntropyBonus computes entropy bonus for exploration.
// probabilities: action probabilities
// Returns: entropy value
func ComputeEntropyBonus(probabilities []float64) float64 {
	if len(probabilities) == 0 {
		return 0.0
	}

	entropy := 0.0
	for _, p := range probabilities {
		if p > 0 {
			entropy -= p * math.Log(p)
		}
	}

	return entropy
}

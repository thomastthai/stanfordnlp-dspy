// Package predict provides prediction modules that interact with language models.
package predict

import (
	"context"
	"fmt"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// BestOfN generates N predictions and selects the best based on a reward function.
type BestOfN struct {
	*primitives.BaseModule

	// Module is the wrapped module to execute
	Module primitives.Module

	// N is the number of predictions to generate
	N int

	// RewardFunc evaluates predictions and returns a score
	RewardFunc func(map[string]interface{}, *primitives.Prediction) float64

	// Threshold is the minimum reward to accept
	Threshold float64

	// FailCount is the number of failures to tolerate
	FailCount int

	// Config contains additional configuration
	Config map[string]interface{}
}

// NewBestOfN creates a new BestOfN module.
func NewBestOfN(module primitives.Module, n int, rewardFunc func(map[string]interface{}, *primitives.Prediction) float64, threshold float64) *BestOfN {
	if n <= 0 {
		n = 3 // Default to 3 attempts
	}

	return &BestOfN{
		BaseModule: primitives.NewBaseModule(),
		Module:     module,
		N:          n,
		RewardFunc: rewardFunc,
		Threshold:  threshold,
		FailCount:  n, // Default to N failures
		Config:     make(map[string]interface{}),
	}
}

// Forward generates N predictions and returns the best one.
func (b *BestOfN) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	var bestPred *primitives.Prediction
	var bestReward float64 = -1e9 // Start with very low reward
	failCount := 0

	for i := 0; i < b.N; i++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Execute the module
		pred, err := b.Module.Forward(ctx, inputs)
		if err != nil {
			failCount++
			if failCount > b.FailCount {
				return nil, fmt.Errorf("too many failures: %w", err)
			}
			continue
		}

		// Calculate reward
		reward := b.RewardFunc(inputs, pred)

		// Track best prediction
		if reward > bestReward {
			bestReward = reward
			bestPred = pred
		}

		// Check if we met the threshold
		if reward >= b.Threshold {
			bestPred.SetMetadata("attempts", i+1)
			bestPred.SetMetadata("reward", reward)
			bestPred.SetMetadata("threshold_met", true)
			return bestPred, nil
		}
	}

	// Return best prediction even if threshold not met
	if bestPred != nil {
		bestPred.SetMetadata("attempts", b.N)
		bestPred.SetMetadata("reward", bestReward)
		bestPred.SetMetadata("threshold_met", false)
		return bestPred, nil
	}

	return nil, fmt.Errorf("failed to generate any valid prediction")
}

// Copy creates a deep copy of the BestOfN module.
func (b *BestOfN) Copy() primitives.Module {
	return &BestOfN{
		BaseModule: primitives.NewBaseModule(),
		Module:     b.Module.Copy(),
		N:          b.N,
		RewardFunc: b.RewardFunc,
		Threshold:  b.Threshold,
		FailCount:  b.FailCount,
		Config:     b.copyConfig(),
	}
}

func (b *BestOfN) copyConfig() map[string]interface{} {
	config := make(map[string]interface{})
	for k, v := range b.Config {
		config[k] = v
	}
	return config
}

// NamedParameters returns all parameters in this module.
func (b *BestOfN) NamedParameters() []primitives.NamedParameter {
	return b.Module.NamedParameters()
}

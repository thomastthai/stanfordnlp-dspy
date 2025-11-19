// Package predict provides prediction modules that interact with language models.
package predict

import (
	"context"
	"fmt"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// Refine iteratively refines predictions using feedback.
type Refine struct {
	*primitives.BaseModule

	// Module is the wrapped module to refine
	Module primitives.Module

	// N is the number of refinement iterations
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

// NewRefine creates a new Refine module.
func NewRefine(module primitives.Module, n int, rewardFunc func(map[string]interface{}, *primitives.Prediction) float64, threshold float64) *Refine {
	if n <= 0 {
		n = 3 // Default to 3 iterations
	}

	return &Refine{
		BaseModule: primitives.NewBaseModule(),
		Module:     module,
		N:          n,
		RewardFunc: rewardFunc,
		Threshold:  threshold,
		FailCount:  n, // Default to N failures
		Config:     make(map[string]interface{}),
	}
}

// Forward refines predictions iteratively with feedback.
func (r *Refine) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	var bestPred *primitives.Prediction
	var bestReward float64 = -1e9 // Start with very low reward
	failCount := 0
	refinedInputs := r.copyInputs(inputs)

	for i := 0; i < r.N; i++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Execute the module
		pred, err := r.Module.Forward(ctx, refinedInputs)
		if err != nil {
			failCount++
			if failCount > r.FailCount {
				return nil, fmt.Errorf("too many failures: %w", err)
			}
			continue
		}

		// Calculate reward
		reward := r.RewardFunc(inputs, pred)

		// Track best prediction
		if reward > bestReward {
			bestReward = reward
			bestPred = pred
		}

		// Check if we met the threshold
		if reward >= r.Threshold {
			bestPred.SetMetadata("refinement_iterations", i+1)
			bestPred.SetMetadata("reward", reward)
			bestPred.SetMetadata("converged", true)
			return bestPred, nil
		}

		// Add feedback for next iteration
		if i < r.N-1 {
			// Generate feedback for improvement
			feedback := r.generateFeedback(pred, reward, r.Threshold)
			refinedInputs["feedback"] = feedback
			refinedInputs["previous_outputs"] = pred.Fields()
			refinedInputs["previous_reward"] = reward
		}
	}

	// Return best prediction even if threshold not met
	if bestPred != nil {
		bestPred.SetMetadata("refinement_iterations", r.N)
		bestPred.SetMetadata("reward", bestReward)
		bestPred.SetMetadata("converged", false)
		return bestPred, nil
	}

	return nil, fmt.Errorf("failed to generate any valid prediction")
}

// generateFeedback creates feedback for the next iteration.
func (r *Refine) generateFeedback(pred *primitives.Prediction, reward float64, threshold float64) string {
	if reward < threshold {
		return fmt.Sprintf("The previous prediction did not meet the quality threshold (reward: %.2f, threshold: %.2f). Please improve the answer based on the feedback.", reward, threshold)
	}
	return "The prediction is close to the target. Please refine it further."
}

// copyInputs creates a copy of the inputs map.
func (r *Refine) copyInputs(inputs map[string]interface{}) map[string]interface{} {
	copied := make(map[string]interface{})
	for k, v := range inputs {
		copied[k] = v
	}
	return copied
}

// Copy creates a deep copy of the Refine module.
func (r *Refine) Copy() primitives.Module {
	return &Refine{
		BaseModule: primitives.NewBaseModule(),
		Module:     r.Module.Copy(),
		N:          r.N,
		RewardFunc: r.RewardFunc,
		Threshold:  r.Threshold,
		FailCount:  r.FailCount,
		Config:     r.copyConfig(),
	}
}

func (r *Refine) copyConfig() map[string]interface{} {
	config := make(map[string]interface{})
	for k, v := range r.Config {
		config[k] = v
	}
	return config
}

// NamedParameters returns all parameters in this module.
func (r *Refine) NamedParameters() []primitives.NamedParameter {
	return r.Module.NamedParameters()
}

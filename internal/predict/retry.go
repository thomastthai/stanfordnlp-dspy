// Package predict provides prediction modules that interact with language models.
package predict

import (
	"context"
	"fmt"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// Retry wraps a module and retries on validation failure with feedback.
type Retry struct {
	*primitives.BaseModule

	// Module is the wrapped module to retry
	Module primitives.Module

	// MaxRetries is the maximum number of retry attempts
	MaxRetries int

	// Validator is a function that validates the output
	Validator func(map[string]interface{}) error

	// Config contains additional configuration
	Config map[string]interface{}
}

// NewRetry creates a new Retry module that wraps another module.
func NewRetry(module primitives.Module, maxRetries int, validator func(map[string]interface{}) error) *Retry {
	return &Retry{
		BaseModule: primitives.NewBaseModule(),
		Module:     module,
		MaxRetries: maxRetries,
		Validator:  validator,
		Config:     make(map[string]interface{}),
	}
}

// Forward executes the module with retry logic.
// It will retry up to MaxRetries times if the validator fails.
func (r *Retry) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	var lastErr error
	var lastPred *primitives.Prediction

	for attempt := 0; attempt <= r.MaxRetries; attempt++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Execute the module
		pred, err := r.Module.Forward(ctx, inputs)
		if err != nil {
			lastErr = err
			continue
		}

		lastPred = pred

		// Validate the output
		if r.Validator != nil {
			if err := r.Validator(pred.Fields()); err != nil {
				lastErr = err
				// Add feedback for next attempt
				inputs["feedback"] = fmt.Sprintf("Previous attempt failed: %v", err)
				inputs["past_outputs"] = pred.Fields()
				continue
			}
		}

		// Success!
		pred.SetMetadata("attempts", attempt+1)
		return pred, nil
	}

	// All retries exhausted
	if lastErr != nil {
		return lastPred, fmt.Errorf("retry exhausted after %d attempts, last error: %w", r.MaxRetries+1, lastErr)
	}

	return lastPred, fmt.Errorf("retry exhausted after %d attempts", r.MaxRetries+1)
}

// Copy creates a deep copy of the Retry module.
func (r *Retry) Copy() primitives.Module {
	return &Retry{
		BaseModule: primitives.NewBaseModule(),
		Module:     r.Module.Copy(),
		MaxRetries: r.MaxRetries,
		Validator:  r.Validator,
		Config:     r.copyConfig(),
	}
}

func (r *Retry) copyConfig() map[string]interface{} {
	config := make(map[string]interface{})
	for k, v := range r.Config {
		config[k] = v
	}
	return config
}

// NamedParameters returns all parameters in this module.
func (r *Retry) NamedParameters() []primitives.NamedParameter {
	return r.Module.NamedParameters()
}

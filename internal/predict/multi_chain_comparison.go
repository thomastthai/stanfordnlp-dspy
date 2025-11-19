// Package predict provides prediction modules that interact with language models.
package predict

import (
	"context"
	"fmt"

	"github.com/stanfordnlp/dspy/internal/primitives"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// MultiChainComparison generates and compares multiple reasoning chains.
type MultiChainComparison struct {
	*Predict

	// M is the number of reasoning chains to generate
	M int

	// Temperature for generating diverse chains
	Temperature float64
}

// NewMultiChainComparison creates a new MultiChainComparison module.
func NewMultiChainComparison(sig interface{}, m int) (*MultiChainComparison, error) {
	if m <= 0 {
		m = 3 // Default to 3 chains
	}

	var signature *signatures.Signature
	var err error

	switch s := sig.(type) {
	case string:
		signature, err = signatures.NewSignature(s)
		if err != nil {
			return nil, fmt.Errorf("failed to parse signature: %w", err)
		}
	case *signatures.Signature:
		signature = s
	default:
		return nil, fmt.Errorf("signature must be string or *Signature, got %T", sig)
	}

	// Add reasoning attempt fields for each chain
	for i := 0; i < m; i++ {
		attemptField := signatures.NewInputField(fmt.Sprintf("reasoning_attempt_%d", i+1))
		attemptField.Description = "Student reasoning attempt"
		attemptField.Prefix = fmt.Sprintf("Student Attempt #%d:", i+1)
		signature.InputFields = append(signature.InputFields, attemptField)
	}

	// Add rationale field to compare chains
	rationaleField := signatures.NewOutputField("rationale")
	rationaleField.Description = "Corrected reasoning comparing all attempts"
	rationaleField.Prefix = "Accurate Reasoning:"

	// Insert rationale at the beginning of outputs
	newOutputFields := make([]*signatures.Field, 0, len(signature.OutputFields)+1)
	newOutputFields = append(newOutputFields, rationaleField)
	newOutputFields = append(newOutputFields, signature.OutputFields...)
	signature.OutputFields = newOutputFields

	return &MultiChainComparison{
		Predict: &Predict{
			BaseModule: primitives.NewBaseModule(),
			Signature:  signature,
			Demos:      primitives.NewParameter(nil),
			Config:     make(map[string]interface{}),
		},
		M:           m,
		Temperature: 0.7,
	}, nil
}

// Forward generates M reasoning chains and compares them.
func (m *MultiChainComparison) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Extract completions (multiple chain results) from inputs
	completionsInterface, ok := inputs["completions"]
	if !ok {
		return nil, fmt.Errorf("missing 'completions' field - must provide multiple reasoning attempts")
	}

	completions, ok := completionsInterface.([]*primitives.Prediction)
	if !ok {
		return nil, fmt.Errorf("completions must be a slice of *primitives.Prediction")
	}

	if len(completions) != m.M {
		return nil, fmt.Errorf("expected %d completions, got %d", m.M, len(completions))
	}

	// Format the reasoning attempts
	for i, completion := range completions {
		outputs := completion.Fields()

		// Extract reasoning (look for "reasoning" or "rationale" fields)
		var reasoning string
		if r, ok := outputs["reasoning"]; ok {
			reasoning = fmt.Sprintf("%v", r)
		} else if r, ok := outputs["rationale"]; ok {
			reasoning = fmt.Sprintf("%v", r)
		} else {
			// Use first output as reasoning
			for _, v := range outputs {
				reasoning = fmt.Sprintf("%v", v)
				break
			}
		}

		// Add formatted attempt to inputs
		attemptKey := fmt.Sprintf("reasoning_attempt_%d", i+1)
		inputs[attemptKey] = fmt.Sprintf("«%s»", reasoning)
	}

	// Remove completions from inputs before validation
	delete(inputs, "completions")

	// Validate inputs
	if err := m.validateInputs(inputs); err != nil {
		return nil, err
	}

	// Create LM integration helper
	lmi, err := NewLMIntegration(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create LM integration: %w", err)
	}

	// Get demos if available
	var demos []map[string]interface{}
	if m.Demos != nil && m.Demos.Value() != nil {
		if demosSlice, ok := m.Demos.Value().([]map[string]interface{}); ok {
			demos = demosSlice
		}
	}

	// Generate output using LM (compares all reasoning attempts)
	output, err := lmi.Generate(ctx, m.Signature, inputs, demos)
	if err != nil {
		return nil, fmt.Errorf("multi-chain comparison failed: %w", err)
	}

	pred := primitives.NewPrediction(output)
	pred.SetMetadata("num_chains", m.M)
	pred.SetMetadata("comparison_method", "multi_chain")

	return pred, nil
}

// Copy creates a deep copy of the MultiChainComparison module.
func (m *MultiChainComparison) Copy() primitives.Module {
	newMCC := &MultiChainComparison{
		Predict: &Predict{
			BaseModule: primitives.NewBaseModule(),
			Signature:  m.Signature,
			Demos:      primitives.NewParameter(m.Demos.Value()),
			Config:     make(map[string]interface{}),
		},
		M:           m.M,
		Temperature: m.Temperature,
	}

	// Copy config
	for k, v := range m.Config {
		newMCC.Config[k] = v
	}

	return newMCC
}

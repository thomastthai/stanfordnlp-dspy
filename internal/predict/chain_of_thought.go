package predict

import (
	"context"
	"fmt"

	"github.com/stanfordnlp/dspy/internal/primitives"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// ChainOfThought extends Predict to include reasoning steps.
type ChainOfThought struct {
	*Predict

	// ReasoningField is the name of the reasoning field
	ReasoningField string
}

// NewChainOfThought creates a new ChainOfThought module.
// It automatically adds a reasoning field to the signature.
func NewChainOfThought(sig interface{}) (*ChainOfThought, error) {
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

	// Add reasoning field before the output fields
	reasoningField := signatures.NewOutputField("reasoning")
	reasoningField.Description = "Think step by step to solve this problem"
	reasoningField.Prefix = "Reasoning:"

	// Insert reasoning field at the beginning of outputs
	newOutputFields := make([]*signatures.Field, 0, len(signature.OutputFields)+1)
	newOutputFields = append(newOutputFields, reasoningField)
	newOutputFields = append(newOutputFields, signature.OutputFields...)

	signature.OutputFields = newOutputFields

	return &ChainOfThought{
		Predict: &Predict{
			BaseModule: primitives.NewBaseModule(),
			Signature:  signature,
			Demos:      primitives.NewParameter(nil),
			Config:     make(map[string]interface{}),
		},
		ReasoningField: "reasoning",
	}, nil
}

// Forward executes the chain of thought prediction.
func (c *ChainOfThought) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	// Validate inputs
	if err := c.validateInputs(inputs); err != nil {
		return nil, err
	}

	// Create LM integration helper
	lmi, err := NewLMIntegration(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create LM integration: %w", err)
	}

	// Get demos if available
	var demos []map[string]interface{}
	if c.Demos != nil && c.Demos.Value() != nil {
		if demosSlice, ok := c.Demos.Value().([]map[string]interface{}); ok {
			demos = demosSlice
		}
	}

	// Generate output using LM (includes reasoning field)
	output, err := lmi.Generate(ctx, c.Signature, inputs, demos)
	if err != nil {
		return nil, fmt.Errorf("chain of thought prediction failed: %w", err)
	}

	pred := primitives.NewPrediction(output)

	// Store reasoning in metadata if present
	if reasoning, ok := output[c.ReasoningField]; ok {
		pred.SetMetadata("reasoning", reasoning)
	}

	return pred, nil
}

// Copy creates a deep copy of the ChainOfThought module.
func (c *ChainOfThought) Copy() primitives.Module {
	newCoT := &ChainOfThought{
		Predict: &Predict{
			BaseModule: primitives.NewBaseModule(),
			Signature:  c.Signature,
			Demos:      primitives.NewParameter(c.Demos.Value()),
			Config:     make(map[string]interface{}),
		},
		ReasoningField: c.ReasoningField,
	}

	// Copy config
	for k, v := range c.Config {
		newCoT.Config[k] = v
	}

	return newCoT
}

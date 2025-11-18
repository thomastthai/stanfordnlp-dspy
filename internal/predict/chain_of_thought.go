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

	// TODO: Implement actual LM call with reasoning
	// For now, return a dummy prediction with reasoning
	output := make(map[string]interface{})

	// Add reasoning
	output[c.ReasoningField] = "[Step-by-step reasoning would go here]"

	// Add other outputs
	for _, field := range c.Signature.OutputFields {
		if field.Name != c.ReasoningField {
			output[field.Name] = fmt.Sprintf("[predicted %s after reasoning]", field.Name)
		}
	}

	pred := primitives.NewPrediction(output)

	// Store reasoning in metadata
	pred.SetMetadata("reasoning", output[c.ReasoningField])

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

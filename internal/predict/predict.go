// Package predict provides prediction modules that interact with language models.
package predict

import (
	"context"
	"fmt"
	
	"github.com/stanfordnlp/dspy/internal/primitives"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// Predict is the basic prediction module that calls an LM with a signature.
type Predict struct {
	*primitives.BaseModule
	
	// Signature defines the input and output structure
	Signature *signatures.Signature
	
	// Demos contains few-shot examples
	Demos *primitives.Parameter
	
	// Config contains additional configuration
	Config map[string]interface{}
}

// New creates a new Predict module with the given signature.
// The signature can be a string like "question -> answer" or a Signature object.
func New(sig interface{}) (*Predict, error) {
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
	
	return &Predict{
		BaseModule: primitives.NewBaseModule(),
		Signature:  signature,
		Demos:      primitives.NewParameter(nil),
		Config:     make(map[string]interface{}),
	}, nil
}

// Forward executes the prediction with the given inputs.
func (p *Predict) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	// Validate inputs
	if err := p.validateInputs(inputs); err != nil {
		return nil, err
	}
	
	// TODO: Implement actual LM call
	// For now, return a dummy prediction
	output := make(map[string]interface{})
	for _, field := range p.Signature.OutputFields {
		output[field.Name] = fmt.Sprintf("[predicted %s]", field.Name)
	}
	
	return primitives.NewPrediction(output), nil
}

// validateInputs checks if all required input fields are provided.
func (p *Predict) validateInputs(inputs map[string]interface{}) error {
	for _, field := range p.Signature.InputFields {
		if field.Required {
			if _, ok := inputs[field.Name]; !ok {
				return fmt.Errorf("required input field missing: %s", field.Name)
			}
		}
	}
	return nil
}

// Copy creates a deep copy of the Predict module.
func (p *Predict) Copy() primitives.Module {
	newPredict := &Predict{
		BaseModule: primitives.NewBaseModule(),
		Signature:  p.Signature, // Signatures are immutable, safe to share
		Demos:      primitives.NewParameter(p.Demos.Value()),
		Config:     make(map[string]interface{}),
	}
	
	// Copy config
	for k, v := range p.Config {
		newPredict.Config[k] = v
	}
	
	return newPredict
}

// NamedParameters returns all parameters in this module.
func (p *Predict) NamedParameters() []primitives.NamedParameter {
	return []primitives.NamedParameter{
		{Name: "demos", Param: p.Demos},
	}
}

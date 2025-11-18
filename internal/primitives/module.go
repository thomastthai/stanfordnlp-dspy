// Package primitives provides the core building blocks for DSPy modules.
package primitives

import "context"

// Module is the interface that all DSPy modules must implement.
// This is analogous to nn.Module in PyTorch.
type Module interface {
	// Forward executes the module with the given inputs.
	// Returns a Prediction containing the outputs.
	Forward(ctx context.Context, inputs map[string]interface{}) (*Prediction, error)
	
	// NamedParameters returns all parameters in this module and its submodules.
	NamedParameters() []NamedParameter
	
	// Reset resets the module's parameters to their initial state.
	Reset()
	
	// Copy creates a deep copy of the module.
	Copy() Module
	
	// Save serializes the module to JSON format.
	Save() ([]byte, error)
	
	// Load deserializes the module from JSON format.
	Load(data []byte) error
}

// NamedParameter represents a named parameter in a module hierarchy.
type NamedParameter struct {
	Name  string
	Param *Parameter
}

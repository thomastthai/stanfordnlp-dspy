// Package adapters provides format adapters for different LM APIs.
package adapters

import (
	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// Adapter defines the interface for format adapters.
// Adapters transform DSPy signatures and examples into LM-specific formats.
type Adapter interface {
	// Format converts a signature and inputs into an LM request.
	Format(sig *signatures.Signature, inputs map[string]interface{}, demos []map[string]interface{}) (*clients.Request, error)

	// Parse extracts outputs from an LM response.
	Parse(sig *signatures.Signature, response *clients.Response) (map[string]interface{}, error)

	// Name returns the adapter name.
	Name() string
}

// BaseAdapter provides common functionality for adapters.
type BaseAdapter struct {
	name string
}

// NewBaseAdapter creates a new base adapter.
func NewBaseAdapter(name string) *BaseAdapter {
	return &BaseAdapter{name: name}
}

// Name implements Adapter.Name.
func (a *BaseAdapter) Name() string {
	return a.name
}

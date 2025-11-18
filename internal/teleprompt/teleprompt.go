// Package teleprompt provides optimizers for DSPy modules.
package teleprompt

import (
	"context"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// Teleprompt is the base interface for all optimizers.
// Teleprompts optimize module parameters (prompts, demonstrations, etc.)
// to improve performance on a metric.
type Teleprompt interface {
	// Compile optimizes a module using the training set and metric.
	Compile(ctx context.Context, module primitives.Module, trainset []*primitives.Example, metric interface{}) (primitives.Module, error)

	// Name returns the optimizer name.
	Name() string
}

// BaseTeleprompt provides common functionality for optimizers.
type BaseTeleprompt struct {
	name string
}

// NewBaseTeleprompt creates a new base teleprompt.
func NewBaseTeleprompt(name string) *BaseTeleprompt {
	return &BaseTeleprompt{name: name}
}

// Name implements Teleprompt.Name.
func (t *BaseTeleprompt) Name() string {
	return t.name
}

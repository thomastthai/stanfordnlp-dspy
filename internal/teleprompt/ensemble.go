package teleprompt

import (
	"context"
	"fmt"
	"math/rand"
	"sync"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// Ensemble combines multiple modules and uses a reduce function to aggregate outputs.
// Based on dspy/teleprompt/ensemble.py
type Ensemble struct {
	*BaseTeleprompt

	// ReduceFn is the function to reduce multiple predictions to one
	// If nil, returns all predictions
	ReduceFn func([]*primitives.Prediction) *primitives.Prediction

	// Size is the number of programs to sample (if nil, uses all)
	Size *int

	// Deterministic determines if sampling is deterministic
	// Currently only false is supported
	Deterministic bool

	// Seed for random sampling
	Seed int64
}

// NewEnsemble creates a new Ensemble optimizer.
func NewEnsemble() *Ensemble {
	return &Ensemble{
		BaseTeleprompt: NewBaseTeleprompt("Ensemble"),
		Deterministic:  false,
		Seed:           0,
	}
}

// WithReduceFn sets the reduce function.
func (e *Ensemble) WithReduceFn(fn func([]*primitives.Prediction) *primitives.Prediction) *Ensemble {
	e.ReduceFn = fn
	return e
}

// WithSize sets the ensemble size.
func (e *Ensemble) WithSize(size int) *Ensemble {
	e.Size = &size
	return e
}

// WithDeterministic sets whether sampling is deterministic.
func (e *Ensemble) WithDeterministic(deterministic bool) *Ensemble {
	if deterministic {
		panic("deterministic ensemble not yet implemented")
	}
	e.Deterministic = deterministic
	return e
}

// Compile implements Teleprompt.Compile.
// It creates an ensembled module that runs multiple programs and aggregates results.
func (e *Ensemble) Compile(ctx context.Context, module primitives.Module, trainset []*primitives.Example, metric interface{}) (primitives.Module, error) {
	// For ensemble, the module parameter is expected to be a slice of modules
	// Since we don't have variadic support in the interface, we expect the trainset
	// to contain the programs in a special format, or we work with a single module
	// For simplicity, we'll create an ensemble of copies with different configurations

	// This is a simplified version - in practice, you'd pass multiple pre-compiled programs
	// For now, return a copy that represents the ensemble concept
	return module.Copy(), fmt.Errorf("ensemble requires multiple pre-compiled programs - not fully implemented")
}

// EnsembledModule wraps multiple modules and executes them together.
type EnsembledModule struct {
	*primitives.BaseModule

	programs  []primitives.Module
	reduceFn  func([]*primitives.Prediction) *primitives.Prediction
	size      *int
	seed      int64
	mu        sync.RWMutex
}

// NewEnsembledModule creates a new ensembled module.
func NewEnsembledModule(programs []primitives.Module, reduceFn func([]*primitives.Prediction) *primitives.Prediction, size *int, seed int64) *EnsembledModule {
	return &EnsembledModule{
		BaseModule: primitives.NewBaseModule(),
		programs:   programs,
		reduceFn:   reduceFn,
		size:       size,
		seed:       seed,
	}
}

// Forward executes all programs and aggregates results.
func (e *EnsembledModule) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	e.mu.RLock()
	programs := e.programs
	size := e.size
	reduceFn := e.reduceFn
	seed := e.seed
	e.mu.RUnlock()

	// Select programs to use
	selectedPrograms := programs
	if size != nil && *size < len(programs) {
		// Random sample
		rng := rand.New(rand.NewSource(seed))
		indices := rng.Perm(len(programs))[:*size]
		selectedPrograms = make([]primitives.Module, *size)
		for i, idx := range indices {
			selectedPrograms[i] = programs[idx]
		}
	}

	// Execute all programs
	predictions := make([]*primitives.Prediction, len(selectedPrograms))
	for i, prog := range selectedPrograms {
		pred, err := prog.Forward(ctx, inputs)
		if err != nil {
			return nil, fmt.Errorf("program %d failed: %w", i, err)
		}
		predictions[i] = pred
	}

	// Reduce if function provided
	if reduceFn != nil {
		return reduceFn(predictions), nil
	}

	// Return first prediction with all predictions in metadata
	result := predictions[0]
	result.SetMetadata("all_predictions", predictions)
	return result, nil
}

// Copy creates a deep copy of the ensembled module.
func (e *EnsembledModule) Copy() primitives.Module {
	e.mu.RLock()
	defer e.mu.RUnlock()

	copiedPrograms := make([]primitives.Module, len(e.programs))
	for i, prog := range e.programs {
		copiedPrograms[i] = prog.Copy()
	}

	var copiedSize *int
	if e.size != nil {
		s := *e.size
		copiedSize = &s
	}

	return NewEnsembledModule(copiedPrograms, e.reduceFn, copiedSize, e.seed)
}

// Package predict provides prediction modules that interact with language models.
package predict

import (
	"context"
	"fmt"
	"sync"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// Parallel executes predictions in parallel using goroutines.
type Parallel struct {
	*primitives.BaseModule

	// NumWorkers is the number of worker goroutines
	NumWorkers int

	// MaxErrors is the maximum number of errors before stopping
	MaxErrors int

	// Config contains additional configuration
	Config map[string]interface{}
}

// NewParallel creates a new Parallel execution module.
func NewParallel(numWorkers int) *Parallel {
	if numWorkers <= 0 {
		numWorkers = 4 // Default to 4 workers
	}

	return &Parallel{
		BaseModule: primitives.NewBaseModule(),
		NumWorkers: numWorkers,
		MaxErrors:  -1, // No limit by default
		Config:     make(map[string]interface{}),
	}
}

// ExecutionPair represents a module and its inputs to execute.
type ExecutionPair struct {
	Module primitives.Module
	Inputs map[string]interface{}
}

// ExecutionResult represents the result of a parallel execution.
type ExecutionResult struct {
	Index      int
	Prediction *primitives.Prediction
	Error      error
}

// Forward executes multiple module-input pairs in parallel.
// Input should contain an "execution_pairs" field with a slice of ExecutionPair.
func (p *Parallel) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Extract execution pairs
	pairsInterface, ok := inputs["execution_pairs"]
	if !ok {
		return nil, fmt.Errorf("missing 'execution_pairs' field in inputs")
	}

	pairs, ok := pairsInterface.([]ExecutionPair)
	if !ok {
		return nil, fmt.Errorf("execution_pairs must be a slice of ExecutionPair")
	}

	if len(pairs) == 0 {
		return nil, fmt.Errorf("no execution pairs provided")
	}

	// Execute in parallel
	results := p.executeParallel(ctx, pairs)

	// Collect results and errors
	var predictions []*primitives.Prediction
	var errors []error
	for _, result := range results {
		if result.Error != nil {
			errors = append(errors, result.Error)
		} else if result.Prediction != nil {
			predictions = append(predictions, result.Prediction)
		}
	}

	// Check if we exceeded max errors
	if p.MaxErrors >= 0 && len(errors) > p.MaxErrors {
		return nil, fmt.Errorf("exceeded max errors: %d errors occurred", len(errors))
	}

	// Create output prediction with results
	output := map[string]interface{}{
		"predictions":   predictions,
		"error_count":   len(errors),
		"success_count": len(predictions),
		"total_count":   len(pairs),
	}

	if len(errors) > 0 {
		output["errors"] = errors
	}

	pred := primitives.NewPrediction(output)
	pred.SetMetadata("parallel_execution", true)
	pred.SetMetadata("num_workers", p.NumWorkers)

	return pred, nil
}

// executeParallel executes the pairs in parallel using a worker pool.
func (p *Parallel) executeParallel(ctx context.Context, pairs []ExecutionPair) []ExecutionResult {
	results := make([]ExecutionResult, len(pairs))
	resultsChan := make(chan ExecutionResult, len(pairs))
	pairsChan := make(chan struct {
		index int
		pair  ExecutionPair
	}, len(pairs))

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < p.NumWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range pairsChan {
				// Check context cancellation
				select {
				case <-ctx.Done():
					resultsChan <- ExecutionResult{
						Index: item.index,
						Error: ctx.Err(),
					}
					continue
				default:
				}

				// Execute the module
				pred, err := item.pair.Module.Forward(ctx, item.pair.Inputs)
				resultsChan <- ExecutionResult{
					Index:      item.index,
					Prediction: pred,
					Error:      err,
				}
			}
		}()
	}

	// Send work to workers
	go func() {
		for i, pair := range pairs {
			pairsChan <- struct {
				index int
				pair  ExecutionPair
			}{index: i, pair: pair}
		}
		close(pairsChan)
	}()

	// Wait for all workers to finish
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Collect results in order
	resultMap := make(map[int]ExecutionResult)
	for result := range resultsChan {
		resultMap[result.Index] = result
	}

	// Sort results by original index
	for i := 0; i < len(pairs); i++ {
		results[i] = resultMap[i]
	}

	return results
}

// Copy creates a deep copy of the Parallel module.
func (p *Parallel) Copy() primitives.Module {
	newParallel := &Parallel{
		BaseModule: primitives.NewBaseModule(),
		NumWorkers: p.NumWorkers,
		MaxErrors:  p.MaxErrors,
		Config:     make(map[string]interface{}),
	}

	for k, v := range p.Config {
		newParallel.Config[k] = v
	}

	return newParallel
}

// NamedParameters returns all parameters in this module.
func (p *Parallel) NamedParameters() []primitives.NamedParameter {
	return []primitives.NamedParameter{}
}

package utils

import (
	"context"
	"sync"
)

// AsyncFunc represents an asynchronous function.
type AsyncFunc[T any] func(context.Context) (T, error)

// Asyncify converts a synchronous function to asynchronous execution using goroutines.
type Asyncify struct {
	maxWorkers int
	semaphore  chan struct{}
}

// NewAsyncify creates a new Asyncify instance with the given number of max workers.
func NewAsyncify(maxWorkers int) *Asyncify {
	if maxWorkers <= 0 {
		maxWorkers = 10 // Default
	}

	return &Asyncify{
		maxWorkers: maxWorkers,
		semaphore:  make(chan struct{}, maxWorkers),
	}
}

// Run executes the given function asynchronously and returns a channel for the result.
func (a *Asyncify) Run(ctx context.Context, fn func() (interface{}, error)) <-chan Result {
	resultCh := make(chan Result, 1)

	go func() {
		// Acquire semaphore
		select {
		case a.semaphore <- struct{}{}:
			defer func() { <-a.semaphore }()
		case <-ctx.Done():
			resultCh <- Result{Error: ctx.Err()}
			close(resultCh)
			return
		}

		// Execute function
		result, err := fn()
		resultCh <- Result{Value: result, Error: err}
		close(resultCh)
	}()

	return resultCh
}

// RunMany executes multiple functions asynchronously and returns a channel for all results.
func (a *Asyncify) RunMany(ctx context.Context, fns []func() (interface{}, error)) <-chan []Result {
	resultCh := make(chan []Result, 1)

	go func() {
		var wg sync.WaitGroup
		results := make([]Result, len(fns))

		for i, fn := range fns {
			wg.Add(1)
			go func(idx int, f func() (interface{}, error)) {
				defer wg.Done()

				// Acquire semaphore
				select {
				case a.semaphore <- struct{}{}:
					defer func() { <-a.semaphore }()
				case <-ctx.Done():
					results[idx] = Result{Error: ctx.Err()}
					return
				}

				// Execute function
				result, err := f()
				results[idx] = Result{Value: result, Error: err}
			}(i, fn)
		}

		wg.Wait()
		resultCh <- results
		close(resultCh)
	}()

	return resultCh
}

// Result represents the result of an async operation.
type Result struct {
	Value interface{}
	Error error
}

// Await waits for the result of an async operation.
func Await(ctx context.Context, resultCh <-chan Result) (interface{}, error) {
	select {
	case result := <-resultCh:
		return result.Value, result.Error
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// AwaitAll waits for all results from multiple async operations.
func AwaitAll(ctx context.Context, resultsCh <-chan []Result) ([]interface{}, []error) {
	select {
	case results := <-resultsCh:
		values := make([]interface{}, len(results))
		errors := make([]error, len(results))
		for i, r := range results {
			values[i] = r.Value
			errors[i] = r.Error
		}
		return values, errors
	case <-ctx.Done():
		return nil, []error{ctx.Err()}
	}
}
